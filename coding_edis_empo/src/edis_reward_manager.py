"""
Configurable oversampled EMPO reward manager.

Active behavior:
1. Generate M responses.
2. Run EMPO once on the original M responses to produce first-round rewards and
   vote artifacts.
3. Select K responses using:
   - `random_topk`
   - `balanced_largest_vs_rest`
4. Assign training rewards directly from the first-round artifact:
   - `use_empo_reward`
   - `use_binary_reward`
5. Optionally apply entropy-based advantage weighting after GRPO.

No reward is recomputed after selection. Discarded responses are masked so GRPO
ignores them.
"""

from collections import defaultdict
import numpy as np
import torch
import traceback
import asyncio
from concurrent.futures import ProcessPoolExecutor

import psutil
from tqdm import tqdm

from verl import DataProto
from verl.workers.reward_manager.empo import (
    EMPORewardManager,
)

BALANCED_SELECTION_METHOD = "D"


def compute_balanced_counts(
    total_selected: int,
    largest_count: int | None = None,
    rest_count: int | None = None,
) -> tuple[int, int]:
    if total_selected % 2 != 0:
        raise ValueError(
            f"Balanced largest-vs-rest selection requires an even K, got {total_selected}"
        )

    if largest_count is None and rest_count is None:
        largest_count = total_selected // 2
        rest_count = total_selected - largest_count
    elif largest_count is None:
        largest_count = total_selected - int(rest_count)
    elif rest_count is None:
        rest_count = total_selected - int(largest_count)

    if largest_count < 0 or rest_count < 0:
        raise ValueError(
            f"largest_count and rest_count must be non-negative, got {largest_count} and {rest_count}"
        )
    if largest_count + rest_count != total_selected:
        raise ValueError(
            f"largest_count + rest_count must equal K={total_selected}, got "
            f"{largest_count} + {rest_count} = {largest_count + rest_count}"
        )
    return int(largest_count), int(rest_count)


def sample_without_replacement(pool_indices: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if len(pool_indices) < count:
        raise ValueError(f"Pool has only {len(pool_indices)} items but needs {count}")

    chosen = np.random.choice(pool_indices, size=count, replace=False)
    return sorted(int(idx) for idx in chosen.tolist())


def select_balanced_largest_vs_rest_indices(
    in_largest_cluster: list[bool],
    largest_cluster_id: int,
    total_selected: int,
    largest_count: int | None,
    rest_count: int | None,
    underfilled_policy: str = "skip_prompt",
) -> tuple[list[int] | None, dict]:
    if underfilled_policy != "skip_prompt":
        raise ValueError(f"Unsupported underfilled_policy: {underfilled_policy}")

    majority_count, rest_count = compute_balanced_counts(
        total_selected,
        largest_count=largest_count,
        rest_count=rest_count,
    )
    if not in_largest_cluster:
        return None, {
            "skip_reason": "empty_group",
            "majority_count": majority_count,
            "rest_count": rest_count,
        }
    majority_pool = [idx for idx, is_largest in enumerate(in_largest_cluster) if is_largest]
    rest_pool = [idx for idx, is_largest in enumerate(in_largest_cluster) if not is_largest]
    largest_cluster_size = len(majority_pool)

    if len(majority_pool) < majority_count or len(rest_pool) < rest_count:
        return None, {
            "skip_reason": "underfilled",
            "largest_cluster_id": largest_cluster_id,
            "largest_cluster_size": largest_cluster_size,
            "majority_pool_size": len(majority_pool),
            "rest_pool_size": len(rest_pool),
            "majority_count": majority_count,
            "rest_count": rest_count,
        }

    selected_majority = sample_without_replacement(majority_pool, majority_count)
    selected_rest = sample_without_replacement(rest_pool, rest_count)
    selected_indices = sorted(selected_majority + selected_rest)

    return selected_indices, {
        "skip_reason": None,
        "largest_cluster_id": largest_cluster_id,
        "largest_cluster_size": largest_cluster_size,
        "majority_pool_size": len(majority_pool),
        "rest_pool_size": len(rest_pool),
        "majority_count": majority_count,
        "rest_count": rest_count,
        "selected_majority_indices": selected_majority,
        "selected_rest_indices": selected_rest,
    }


def metric_values(metrics: dict, key: str, count: int, default):
    values = metrics.pop(key, None)
    if values is None:
        return [default] * count
    return list(values)


def slice_by_indices(values: list, selected_indices: list[int]) -> list:
    return [values[idx] for idx in selected_indices]


def build_binary_tag_rewards(
    selected_local_indices: list[int],
    vote_supports_winner: list[bool],
) -> list[float]:
    return [1.0 if vote_supports_winner[idx] else 0.0 for idx in selected_local_indices]


class EdisEmpoRewardManager(EMPORewardManager):
    """
    Configurable EMPO-binary reward manager.

    Active 2x2 controls:
      - selection_family: random_topk
      - reward_method: use_binary_reward
      - adv_weight_mode: off | entropy_high_token_ratio

    The original M-sample EMPO artifact is always computed first and logged as
    `pre_selection_*`. Training rewards are always reused directly from that
    first-round artifact; no selected-K reward recomputation exists.

    The balanced largest-vs-rest branch is retained only for legacy runs. New
    experiments should use random_topk and control behavior with oversampled
    plus weighted script switches.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        n_votes_per_prompt=1,
        n_samples_per_prompt=1,
        mode="eval",
        eval_n_samples=1,
        reward_type=None,
        entropy_thres=None,
        use_xverify=True,
        use_eval_math=False,
        gt_reward_coef: float = 0.0,
        selected_top_k: int = 8,
        selection_family: str = "random_topk",
        balanced_selection_method: str = BALANCED_SELECTION_METHOD,
        largest_group_count: int | None = None,
        rest_group_count: int | None = None,
        underfilled_policy: str = "skip_prompt",
        reward_method: str = "use_binary_reward",
        adv_weight_mode: str = "off",
        entropy_weight_top_fraction: float = 0.2,
        adv_weight_clip_min: float = 0.5,
        adv_weight_clip_max: float = 1.5,
        min_response_length: int = 0,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=overlong_buffer_cfg,
            n_votes_per_prompt=n_votes_per_prompt,
            n_samples_per_prompt=n_samples_per_prompt,
            mode=mode,
            eval_n_samples=eval_n_samples,
            reward_type=reward_type,
            entropy_thres=entropy_thres,
            use_xverify=use_xverify,
            use_eval_math=use_eval_math,
            gt_reward_coef=gt_reward_coef,
        )

        if selection_family not in {"random_topk", "balanced_largest_vs_rest"}:
            raise ValueError(f"Unsupported selection_family: {selection_family}")
        if balanced_selection_method != BALANCED_SELECTION_METHOD:
            raise ValueError(
                f"Unsupported balanced_selection_method: {balanced_selection_method}"
            )
        if reward_method not in {"use_empo_reward", "use_binary_reward"}:
            raise ValueError(f"Unsupported reward_method: {reward_method}")
        if adv_weight_mode not in {"off", "entropy_high_token_ratio"}:
            raise ValueError(
                f"Unsupported adv_weight_mode: {adv_weight_mode}"
            )
        if underfilled_policy != "skip_prompt":
            raise ValueError(f"Unsupported underfilled_policy: {underfilled_policy}")
        if not (0.0 < entropy_weight_top_fraction < 1.0):
            raise ValueError(
                "entropy_weight_top_fraction must be between 0 and 1, got "
                f"{entropy_weight_top_fraction}"
            )
        if adv_weight_clip_min <= 0 or adv_weight_clip_max <= 0 or adv_weight_clip_min > adv_weight_clip_max:
            raise ValueError(
                "Invalid advantage weight clip range: "
                f"{adv_weight_clip_min} to {adv_weight_clip_max}"
            )
        if selected_top_k > n_votes_per_prompt:
            raise ValueError(
                f"selected_top_k ({selected_top_k}) must be <= n_votes_per_prompt ({n_votes_per_prompt})"
            )
        if selection_family == "balanced_largest_vs_rest" and selected_top_k % 2 != 0:
            raise ValueError(
                f"Balanced largest-vs-rest selection requires an even K, got {selected_top_k}"
            )
        if selection_family != "balanced_largest_vs_rest" and (
            largest_group_count is not None or rest_group_count is not None
        ):
            raise ValueError(
                "largest_group_count / rest_group_count only apply when "
                "selection_family='balanced_largest_vs_rest'"
            )
        if selection_family == "balanced_largest_vs_rest":
            compute_balanced_counts(
                selected_top_k,
                largest_count=largest_group_count,
                rest_count=rest_group_count,
            )

        self.selected_top_k = selected_top_k
        self.selection_family = selection_family
        self.balanced_selection_method = balanced_selection_method
        self.largest_group_count = largest_group_count
        self.rest_group_count = rest_group_count
        self.underfilled_policy = underfilled_policy
        self.reward_method = reward_method
        self.adv_weight_mode = adv_weight_mode
        self.entropy_weight_top_fraction = entropy_weight_top_fraction
        self.adv_weight_clip_min = adv_weight_clip_min
        self.adv_weight_clip_max = adv_weight_clip_max
        self.min_response_length = min_response_length

        if selection_family == "balanced_largest_vs_rest":
            resolved_largest_count, resolved_rest_count = compute_balanced_counts(
                selected_top_k,
                largest_count=largest_group_count,
                rest_count=rest_group_count,
            )
            balanced_count_desc = (
                f"largest_group_count={resolved_largest_count}, "
                f"rest_group_count={resolved_rest_count}"
            )
        else:
            balanced_count_desc = "largest_group_count=n/a, rest_group_count=n/a"

        print(
            "EmpoBinaryRewardManager initialized: "
            f"M={n_votes_per_prompt}, K={selected_top_k}, "
            f"selection_family={selection_family}, "
            f"balanced_selection_method={balanced_selection_method}, "
            f"reward_method={reward_method}, "
            f"adv_weight_mode={adv_weight_mode}, "
            f"min_response_length={min_response_length}, "
            f"{balanced_count_desc}."
        )

    def _select_local_indices(
        self,
        original_in_largest_cluster: list[bool],
        original_largest_cluster_id: list[int],
        eligible_mask: list[bool] | None = None,
    ) -> tuple[list[int] | None, dict]:
        total_selected = self.selected_top_k

        if self.selection_family == "random_topk":
            all_indices = np.arange(len(original_in_largest_cluster))
            if eligible_mask is not None:
                all_indices = all_indices[[eligible_mask[i] for i in all_indices]]
            np.random.shuffle(all_indices)
            selected_local_indices = sorted(all_indices[:total_selected].tolist())
            return selected_local_indices, {
                "skip_reason": None,
                "selected_topk_indices": selected_local_indices,
            }

        return select_balanced_largest_vs_rest_indices(
            in_largest_cluster=original_in_largest_cluster,
            largest_cluster_id=original_largest_cluster_id[0] if original_largest_cluster_id else -1,
            total_selected=total_selected,
            largest_count=self.largest_group_count,
            rest_count=self.rest_group_count,
            underfilled_policy=self.underfilled_policy,
        )

    def _selected_pool_label(self, context: dict, local_i: int, is_selected: bool) -> str:
        if self.selection_family == "balanced_largest_vs_rest":
            if local_i in context["selected_majority_indices"]:
                return "majority"
            if local_i in context["selected_rest_indices"]:
                return "rest"
            return "unselected"
        return "selected" if is_selected else "unselected"

    async def _compute_empo_reward_async(self, data: "DataProto"):
        M = self.n_votes_per_prompt
        K = self.selected_top_k

        assert len(data) % M == 0, \
            f"Data length {len(data)} must be divisible by n_votes_per_prompt {M}"

        prompt_num = len(data) // M
        reward_tensor = torch.zeros_like(
            data.batch["responses"][:prompt_num * self.n_samples_per_prompt],
            dtype=torch.float32,
        )
        scores = [0.0] * len(data)
        prompts_str_set = set()
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        discard_global_indices = []
        selected_global_indices = []
        skipped_prompt_count = 0
        num_short_filtered = 0

        with ProcessPoolExecutor(max_workers=64) as executor:
            prompt_contexts = []

            for prompt_i in range(prompt_num):
                group_pred_outputs, group_labels, group_extra_info = [], [], []
                results_template = []

                for local_i in range(M):
                    global_i = prompt_i * M + local_i
                    data_item = data[global_i]

                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]
                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if isinstance(ground_truth, list):
                        ground_truth = ground_truth[-1]

                    group_labels.append(ground_truth)
                    group_pred_outputs.append(response_str)
                    group_extra_info.append(extra_info)
                    prompts_str_set.add(prompt_str)

                    results_template.append({
                        "prompt": prompt_str,
                        "response": response_str,
                        "extracted_pred": "",
                        "extracted_gt": ground_truth,
                        "estimated_label": ground_truth,
                        "score": 0.0,
                        "data_source": data_source,
                        "scored_by": "not_scored",
                        "index": extra_info.get("index", "N/A"),
                        "valid_response_length": valid_response_length,
                        "label_accuracy": 0.0,
                        "majority_ratio": 0.0,
                        f"pass@{K}": 0.0,
                    })

                original_vote_task = asyncio.create_task(
                    self._single_empo_task(
                        {
                            "group_pred_outputs": group_pred_outputs,
                            "group_labels": group_labels,
                            "group_extra_info": group_extra_info,
                            "reward_type": self.reward_type,
                            "entropy_thres": self.entropy_thres,
                            "vote_mode": "majority",
                        },
                        executor,
                    )
                )

                prompt_contexts.append({
                    "prompt_i": prompt_i,
                    "group_pred_outputs": group_pred_outputs,
                    "group_labels": group_labels,
                    "group_extra_info": group_extra_info,
                    "results_template": results_template,
                    "original_vote_task": original_vote_task,
                    "response_lengths": [rt["valid_response_length"] for rt in results_template],
                    "selected_local_indices": [],
                    "selected_majority_indices": [],
                    "selected_rest_indices": [],
                    "skip_prompt": False,
                    "selection_skip_reason": "",
                    "original_rewards": [0.0] * M,
                    "original_vote_supports_winner": [False] * M,
                    "original_vote_tag": ["incorrect"] * M,
                    "original_vote_winner_label": [""] * M,
                    "original_vote_winner_score": [0.0] * M,
                    "original_vote_mode": ["majority"] * M,
                    "original_cluster_id": [-1] * M,
                    "original_cluster_size": [0] * M,
                    "original_cluster_frequency": [0.0] * M,
                    "original_cluster_vote_score": [0.0] * M,
                    "original_sample_vote_weight": [1.0] * M,
                    "original_in_largest_cluster": [False] * M,
                    "original_largest_cluster_id": [-1] * M,
                    "original_majority_ratio": [0.0] * M,
                    "original_label_accuracy": [0.0] * M,
                    "original_extracted_answers": [""] * M,
                })

            for context in prompt_contexts:
                group_index = context["group_extra_info"][0].get("index", "N/A")
                try:
                    original_rewards, original_metrics = await asyncio.wait_for(
                        context["original_vote_task"], timeout=600.0
                    )
                except asyncio.TimeoutError:
                    print(f"[Timeout] Original vote task for group {group_index} timed out.")
                    original_rewards = [0.0] * M
                    original_metrics = {}
                except Exception as exc:
                    print(f"[Error] Original vote task for group {group_index} failed: {exc}")
                    original_rewards = [0.0] * M
                    original_metrics = {}

                original_rewards = list(original_rewards or [])
                if len(original_rewards) < M:
                    original_rewards.extend([0.0] * (M - len(original_rewards)))
                elif len(original_rewards) > M:
                    original_rewards = original_rewards[:M]
                context["original_rewards"] = [float(v) for v in original_rewards]
                original_metrics = dict(original_metrics or {})
                context["original_cluster_id"] = metric_values(original_metrics, "cluster_id", M, -1)
                context["original_cluster_size"] = metric_values(original_metrics, "cluster_size", M, 0)
                context["original_cluster_frequency"] = metric_values(
                    original_metrics, "cluster_frequency", M, 0.0
                )
                context["original_cluster_vote_score"] = metric_values(
                    original_metrics, "cluster_vote_score", M, 0.0
                )
                context["original_sample_vote_weight"] = metric_values(
                    original_metrics, "sample_vote_weight", M, 1.0
                )
                context["original_vote_supports_winner"] = metric_values(
                    original_metrics, "vote_supports_winner", M, False
                )
                context["original_vote_tag"] = metric_values(
                    original_metrics, "vote_tag", M, "incorrect"
                )
                context["original_vote_winner_label"] = metric_values(
                    original_metrics, "vote_winner_label", M, ""
                )
                context["original_vote_winner_score"] = metric_values(
                    original_metrics, "vote_winner_score", M, 0.0
                )
                context["original_vote_mode"] = metric_values(
                    original_metrics, "vote_mode", M, "majority"
                )
                context["original_in_largest_cluster"] = metric_values(
                    original_metrics, "in_largest_cluster", M, False
                )
                context["original_largest_cluster_id"] = metric_values(
                    original_metrics, "largest_cluster_id", M, -1
                )
                context["original_majority_ratio"] = metric_values(
                    original_metrics, "majority_ratio", M, 0.0
                )
                context["original_label_accuracy"] = metric_values(
                    original_metrics, "label_accuracy", M, 0.0
                )
                context["original_extracted_answers"] = metric_values(
                    original_metrics, "extracted_answers", M, ""
                )

                selected_local_indices, selection_meta = self._select_local_indices(
                    original_in_largest_cluster=context["original_in_largest_cluster"],
                    original_largest_cluster_id=context["original_largest_cluster_id"],
                    eligible_mask=[
                        int(context["response_lengths"][i]) >= self.min_response_length
                        for i in range(M)
                    ] if self.min_response_length > 0 else None,
                )
                if self.min_response_length > 0:
                    num_short_filtered += sum(
                        1 for l in context["response_lengths"] if int(l) < self.min_response_length
                    )

                if selected_local_indices is None:
                    context["skip_prompt"] = True
                    context["selection_skip_reason"] = selection_meta["skip_reason"] or "underfilled"
                    skipped_prompt_count += 1
                    for local_i in range(M):
                        discard_global_indices.append(context["prompt_i"] * M + local_i)
                    continue

                selected_set = set(selected_local_indices)
                context["selected_local_indices"] = selected_local_indices
                if self.selection_family == "balanced_largest_vs_rest":
                    majority_pool_set = {
                        idx
                        for idx, is_largest in enumerate(context["original_in_largest_cluster"])
                        if is_largest
                    }
                    context["selected_majority_indices"] = [
                        idx for idx in selected_local_indices if idx in majority_pool_set
                    ]
                    context["selected_rest_indices"] = [
                        idx for idx in selected_local_indices if idx not in majority_pool_set
                    ]

                if context["prompt_i"] < 2:
                    if self.selection_family == "random_topk":
                        print(
                            f"[RANDOM-TOPK] Prompt {context['prompt_i']}: "
                            f"selected indices {selected_local_indices} out of {M}"
                        )
                    else:
                        print(
                            f"[BALANCED] Prompt {context['prompt_i']}: "
                            f"selected majority={len(context['selected_majority_indices'])}, "
                            f"rest={len(context['selected_rest_indices'])}"
                        )

                for local_i in range(M):
                    global_i = context["prompt_i"] * M + local_i
                    if local_i in selected_set:
                        selected_global_indices.append(global_i)
                    else:
                        discard_global_indices.append(global_i)

            try:
                for context in tqdm(prompt_contexts, desc="Computing configurable oversampled EMPO rewards"):
                    prompt_i = context["prompt_i"]
                    selected_positions = {
                        local_i: pos for pos, local_i in enumerate(context["selected_local_indices"])
                    }
                    selected_count = len(context["selected_local_indices"])

                    selected_empo_rewards = slice_by_indices(
                        context["original_rewards"],
                        context["selected_local_indices"],
                    )
                    selected_binary_rewards = build_binary_tag_rewards(
                        context["selected_local_indices"],
                        context["original_vote_supports_winner"],
                    )

                    for local_i in range(M):
                        result_template = context["results_template"][local_i]
                        is_selected = local_i in selected_positions
                        selected_pool = self._selected_pool_label(context, local_i, is_selected)

                        result_template.update({
                            "selection_family": self.selection_family,
                            "balanced_selection_method": (
                                self.balanced_selection_method
                                if self.selection_family == "balanced_largest_vs_rest" else ""
                            ),
                            "reward_method": self.reward_method,
                            "adv_weight_mode": self.adv_weight_mode,
                            "pre_selection_cluster_id": context["original_cluster_id"][local_i],
                            "pre_selection_cluster_size": context["original_cluster_size"][local_i],
                            "pre_selection_cluster_frequency": context["original_cluster_frequency"][local_i],
                            "pre_selection_cluster_vote_score": context["original_cluster_vote_score"][local_i],
                            "pre_selection_sample_vote_weight": context["original_sample_vote_weight"][local_i],
                            "pre_selection_vote_supports_winner": context["original_vote_supports_winner"][local_i],
                            "pre_selection_vote_tag": context["original_vote_tag"][local_i],
                            "pre_selection_vote_winner_label": context["original_vote_winner_label"][local_i],
                            "pre_selection_vote_winner_score": context["original_vote_winner_score"][local_i],
                            "first_round_vote_mode": context["original_vote_mode"][local_i],
                            "pre_selection_in_largest_cluster": context["original_in_largest_cluster"][local_i],
                            "pre_selection_largest_cluster_id": context["original_largest_cluster_id"][local_i],
                            "pre_selection_majority_ratio": context["original_majority_ratio"][local_i],
                            "pre_selection_label_accuracy": context["original_label_accuracy"][local_i],
                            "pre_selection_extracted_pred": context["original_extracted_answers"][local_i],
                            "first_round_reward": context["original_rewards"][local_i],
                            "first_round_binary_reward": float(
                                1.0 if context["original_vote_supports_winner"][local_i] else 0.0
                            ),
                            "selected_for_training": is_selected or (
                                self.min_response_length > 0
                                and int(context["response_lengths"][local_i]) < self.min_response_length
                            ),
                            "selected_pool": selected_pool,
                            "selection_status": (
                                "skipped" if context["skip_prompt"]
                                else ("selected" if is_selected else "discarded")
                            ),
                            "selection_skip_reason": context["selection_skip_reason"],
                            "reward_source": self.reward_method,
                        })

                        if context["skip_prompt"]:
                            reward = 0.0
                            reward_weight_factor = 1.0
                            estimated_label = ""
                            model_answer = ""
                            label_accuracy = 0.0
                            majority_ratio = 0.0
                            cluster_id = -1
                            cluster_size = 0
                            cluster_frequency = 0.0
                            cluster_vote_score = 0.0
                            sample_vote_weight = 1.0
                            supports_winner = False
                            vote_tag = "incorrect"
                            vote_winner_label = ""
                            vote_winner_score = 0.0
                            current_vote_mode = "majority"
                            largest_cluster_id = -1
                            is_in_largest_cluster = False
                        elif is_selected:
                            selected_pos = selected_positions[local_i]
                            if self.reward_method == "use_empo_reward":
                                reward = float(selected_empo_rewards[selected_pos])
                            else:
                                reward = float(selected_binary_rewards[selected_pos])
                            reward_weight_factor = 1.0
                            estimated_label = context["original_vote_winner_label"][local_i]
                            model_answer = context["original_extracted_answers"][local_i]
                            label_accuracy = context["original_label_accuracy"][local_i]
                            majority_ratio = context["original_majority_ratio"][local_i]
                            cluster_id = context["original_cluster_id"][local_i]
                            cluster_size = context["original_cluster_size"][local_i]
                            cluster_frequency = context["original_cluster_frequency"][local_i]
                            cluster_vote_score = context["original_cluster_vote_score"][local_i]
                            sample_vote_weight = context["original_sample_vote_weight"][local_i]
                            supports_winner = context["original_vote_supports_winner"][local_i]
                            vote_tag = context["original_vote_tag"][local_i]
                            vote_winner_label = context["original_vote_winner_label"][local_i]
                            vote_winner_score = context["original_vote_winner_score"][local_i]
                            current_vote_mode = context["original_vote_mode"][local_i]
                            largest_cluster_id = context["original_largest_cluster_id"][local_i]
                            is_in_largest_cluster = context["original_in_largest_cluster"][local_i]
                        else:
                            reward = 0.0
                            reward_weight_factor = 1.0
                            estimated_label = ""
                            model_answer = ""
                            label_accuracy = 0.0
                            majority_ratio = 0.0
                            cluster_id = -1
                            cluster_size = 0
                            cluster_frequency = 0.0
                            cluster_vote_score = 0.0
                            sample_vote_weight = 1.0
                            supports_winner = False
                            vote_tag = "incorrect"
                            vote_winner_label = ""
                            vote_winner_score = 0.0
                            current_vote_mode = "majority"
                            largest_cluster_id = -1
                            is_in_largest_cluster = False

                        result_template.update({
                            "estimated_label": estimated_label,
                            "extracted_pred": model_answer,
                            "label_accuracy": label_accuracy,
                            "majority_ratio": majority_ratio,
                            "cluster_id": cluster_id,
                            "cluster_size": cluster_size,
                            "cluster_frequency": cluster_frequency,
                            "cluster_vote_score": cluster_vote_score,
                            "sample_vote_weight": sample_vote_weight,
                            "vote_supports_winner": supports_winner,
                            "vote_tag": vote_tag,
                            "vote_winner_label": vote_winner_label,
                            "vote_winner_score": vote_winner_score,
                            "vote_mode": current_vote_mode,
                            "largest_cluster_id": largest_cluster_id,
                            "in_largest_cluster": is_in_largest_cluster,
                            "reward_weight_factor": reward_weight_factor,
                            "score": str(float(reward)),
                        })

                        hybrid_added = 0.0
                        if self.gt_reward_coef > 0:
                            hybrid_added = self.gt_reward_coef * float(label_accuracy)
                            reward = float(reward) + hybrid_added
                        result_template["hybrid_gt_component"] = hybrid_added
                        result_template["hybrid_reward"] = float(reward)

                        for key, value in result_template.items():
                            if key != "valid_response_length":
                                reward_extra_info[key].append(value)

                        score_index = prompt_i * M + local_i
                        scores[score_index] = reward

                        if local_i < self.n_samples_per_prompt:
                            tensor_index = prompt_i * self.n_samples_per_prompt + local_i
                            reward_tensor[tensor_index, result_template["valid_response_length"] - 1] = reward

                        data_source = result_template["data_source"]
                        if data_source not in already_print_data_sources:
                            already_print_data_sources[data_source] = 0
                        if already_print_data_sources[data_source] < self.num_examine:
                            already_print_data_sources[data_source] += 1
                            print(
                                f"[{self.selection_family.upper()} {'SELECTED' if is_selected else 'DISCARDED'}] "
                                f"pool={selected_pool} "
                                f"first_round_tag={result_template['pre_selection_vote_tag']} "
                                f"reward={reward:.4f}"
                            )

                print("[Success] All configurable oversampled EMPO tasks gathered.")
            except Exception as exc:
                print(f"[Exception] in processing configurable oversampled EMPO futures: {exc}")
                traceback.print_exc()
            finally:
                print("[Shutdown] Cleaning up remaining configurable oversampled EMPO subprocesses...")
                terminated_count, total_count = 0, 0
                for pid, proc in list(executor._processes.items()):
                    try:
                        process = psutil.Process(pid)
                        if process.is_running():
                            process.terminate()
                            try:
                                process.wait(timeout=600.0)
                            except psutil.TimeoutExpired:
                                print(f"Process {pid} did not terminate, killing...")
                                process.kill()
                            terminated_count += 1
                    except psutil.NoSuchProcess:
                        pass
                    except Exception as cleanup_exc:
                        print(f"Error cleaning up process {pid}: {cleanup_exc}")
                    finally:
                        total_count += 1
                print(f"[Shutdown] {terminated_count} of {total_count} subprocess(es) terminated.")

        mask_count = 0
        response_length = data.batch["responses"].shape[1]
        for global_i in discard_global_indices:
            if "response_mask" in data.batch:
                data.batch["response_mask"][global_i, :] = 0
            if "loss_mask" in data.batch:
                data.batch["loss_mask"][global_i, -response_length:] = 0
            if "uid" in data.non_tensor_batch:
                data.non_tensor_batch["uid"][global_i] = (
                    str(data.non_tensor_batch["uid"][global_i]) + "_discarded"
                )
            mask_count += 1

        print(
            f"[{self.selection_family.upper()}] Masked {mask_count} of {len(data)} responses; "
            f"selected={len(selected_global_indices)}, skipped_prompts={skipped_prompt_count}/{prompt_num}, "
            f"short_filtered={num_short_filtered}."
        )

        device = data.batch["prompts"].device
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=device)
        data.batch["vote_supports_winner"] = torch.tensor(
            [bool(v) for v in reward_extra_info["vote_supports_winner"]],
            dtype=torch.bool,
            device=device,
        )
        data.batch["pre_selection_vote_supports_winner"] = torch.tensor(
            [bool(v) for v in reward_extra_info["pre_selection_vote_supports_winner"]],
            dtype=torch.bool,
            device=device,
        )

        empo_info = {
            "skipped_prompt_count": skipped_prompt_count,
            "selected_response_count": len(selected_global_indices),
            "num_short_filtered": num_short_filtered,
        }

        assert len(reward_extra_info["prompt"]) == len(data), \
            f"Incomplete reward info, expect {len(data)} but got {len(reward_extra_info['prompt'])}"

        if len(prompts_str_set) < prompt_num:
            print(
                f"[Warning] Expected {prompt_num} distinct prompts but got "
                f"{len(prompts_str_set)}, abnormal except for TTT setting."
            )

        return reward_tensor, reward_extra_info, empo_info
