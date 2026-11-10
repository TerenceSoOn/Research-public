# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
import json
import os
import shutil
from collections import defaultdict, Counter
from copy import deepcopy
from pprint import pprint
import wandb
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_empo_metrics,
    compute_overlong_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.utils.debug import marked_timer


def apply_optional_advantage_weighting(
    batch: DataProto,
    adv_weight_mode: str = "off",
    entropy_weight_top_fraction: float = 0.2,
    adv_weight_clip_min: float = 0.5,
    adv_weight_clip_max: float = 1.5,
):
    """
    Apply optional per-sample scaling to already-computed GRPO advantages.

    The entropy mode is reward-aware:
      - largest-cluster / binary-positive samples: lower entropy => higher weight
      - rest-cluster / binary-zero samples: lower entropy => lower weight

    This keeps EMPO reward computation unchanged and only rescales the actor
    update signal after GRPO computes advantages.
    """
    if adv_weight_mode != "entropy_high_token_ratio":
        return {}

    selected_for_training = batch.non_tensor_batch.get("selected_for_training")
    in_largest_cluster = batch.non_tensor_batch.get("pre_selection_in_largest_cluster")
    token_level_entropys = batch.batch.get("token_level_entropys")
    response_mask = batch.batch.get("response_mask")
    entropy_reference_mask = batch.batch.get("entropy_reference_response_mask", response_mask)
    if selected_for_training is None or token_level_entropys is None or response_mask is None:
        return {}
    if in_largest_cluster is None:
        raise ValueError(
            "entropy_high_token_ratio advantage weighting requires "
            "pre_selection_in_largest_cluster metadata"
        )

    def _to_bool(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).lower() in {"1", "true", "yes", "on"}

    apply_mask = np.array([_to_bool(value) for value in selected_for_training], dtype=bool)
    largest_mask_np = np.array([_to_bool(value) for value in in_largest_cluster], dtype=bool)
    if apply_mask.shape != largest_mask_np.shape:
        raise ValueError(
            "selected_for_training and pre_selection_in_largest_cluster must have "
            f"the same length, got {apply_mask.shape[0]} and {largest_mask_np.shape[0]}"
        )
    if not np.any(apply_mask):
        return {}

    advantages = batch.batch["advantages"]
    entropies = token_level_entropys.to(device=advantages.device, dtype=advantages.dtype)
    response_mask = response_mask.to(device=advantages.device).bool()
    entropy_reference_mask = entropy_reference_mask.to(device=advantages.device).bool()
    if entropies.shape != response_mask.shape:
        raise ValueError(
            "token_level_entropys must have the same shape as response_mask, got "
            f"{tuple(entropies.shape)} vs {tuple(response_mask.shape)}"
        )
    if entropies.shape != entropy_reference_mask.shape:
        raise ValueError(
            "token_level_entropys must have the same shape as entropy_reference_response_mask, got "
            f"{tuple(entropies.shape)} vs {tuple(entropy_reference_mask.shape)}"
        )

    valid_entropies = entropies[entropy_reference_mask]
    if valid_entropies.numel() == 0:
        return {}

    quantile = float(np.clip(1.0 - entropy_weight_top_fraction, 0.0, 1.0))
    threshold = torch.quantile(valid_entropies, quantile)
    high_entropy_mask = (entropies >= threshold) & entropy_reference_mask

    valid_token_counts = entropy_reference_mask.sum(dim=-1)
    high_entropy_counts = high_entropy_mask.sum(dim=-1)
    ratios = torch.zeros(entropies.shape[0], device=advantages.device, dtype=advantages.dtype)
    non_empty = valid_token_counts > 0
    ratios[non_empty] = (
        high_entropy_counts[non_empty].to(advantages.dtype)
        / valid_token_counts[non_empty].to(advantages.dtype)
    )

    selected_mask = torch.as_tensor(apply_mask, device=advantages.device)
    largest_mask = torch.as_tensor(largest_mask_np, device=advantages.device)
    selected_non_empty = selected_mask & non_empty
    selected_largest_mask = selected_non_empty & largest_mask
    selected_rest_mask = selected_non_empty & ~largest_mask

    stable_weights = torch.ones_like(ratios)
    unstable_weights = torch.ones_like(ratios)
    if torch.any(non_empty):
        mean_ratio = ratios[non_empty].mean()
        if mean_ratio.item() > 0:
            stable_weights[non_empty] = mean_ratio / (ratios[non_empty] + 1e-8)
            unstable_weights[non_empty] = ratios[non_empty] / mean_ratio
    stable_weights = torch.clamp(stable_weights, adv_weight_clip_min, adv_weight_clip_max)
    unstable_weights = torch.clamp(unstable_weights, adv_weight_clip_min, adv_weight_clip_max)

    effective_weights = torch.ones_like(ratios)
    effective_weights[selected_largest_mask] = stable_weights[selected_largest_mask]
    effective_weights[selected_rest_mask] = unstable_weights[selected_rest_mask]
    batch.batch["advantages"] = advantages * effective_weights.unsqueeze(-1)

    selected_weights = effective_weights[selected_mask]
    selected_ratios = ratios[selected_mask]

    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if not torch.any(mask):
            return 0.0
        return values[mask].mean().item()

    return {
        "adv_weight/mean": selected_weights.mean().item(),
        "adv_weight/min": selected_weights.min().item(),
        "adv_weight/max": selected_weights.max().item(),
        "adv_weight/num_weighted": float(selected_weights.numel()),
        "adv_weight/high_entropy_ratio_mean": selected_ratios.mean().item(),
        "adv_weight/high_entropy_threshold": threshold.item(),
        "adv_weight/largest_mean": _masked_mean(effective_weights, selected_largest_mask),
        "adv_weight/rest_mean": _masked_mean(effective_weights, selected_rest_mask),
        "adv_weight/largest_high_entropy_ratio_mean": _masked_mean(ratios, selected_largest_mask),
        "adv_weight/rest_high_entropy_ratio_mean": _masked_mean(ratios, selected_rest_mask),
        "adv_weight/largest_count": float(selected_largest_mask.sum().item()),
        "adv_weight/rest_count": float(selected_rest_mask.sum().item()),
    }


def apply_selected_training_mask(batch: DataProto):
    """
    Restrict training masks to selected samples while preserving a reference mask
    over all generated responses for entropy-threshold calculation.
    """
    selected_for_training = batch.non_tensor_batch.get("selected_for_training")
    response_mask = batch.batch.get("response_mask")
    if selected_for_training is None or response_mask is None:
        return {}

    def _to_bool(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).lower() in {"1", "true", "yes", "on"}

    selected_mask = torch.as_tensor(
        [_to_bool(value) for value in selected_for_training],
        device=response_mask.device,
        dtype=torch.bool,
    )
    batch.batch["entropy_reference_response_mask"] = response_mask.clone()
    batch.batch["response_mask"] = response_mask * selected_mask.unsqueeze(-1).to(response_mask.dtype)
    return {
        "selection/num_training_responses": float(selected_mask.sum().item()),
        "selection/num_masked_responses": float((~selected_mask).sum().item()),
    }


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _topk_checkpoint_config(self):
        trainer_cfg = self.config.trainer
        raw_enabled = trainer_cfg.get("topk_checkpoint_enable", False)
        if isinstance(raw_enabled, str):
            enabled = raw_enabled.lower() in {"1", "true", "yes", "on"}
        else:
            enabled = bool(raw_enabled)
        k = int(trainer_cfg.get("topk_checkpoint_k", 3))
        metric_key = trainer_cfg.get("topk_checkpoint_metric", "val-core/amc/acc/mean@16")
        dir_name = trainer_cfg.get("topk_checkpoint_dir_name", "top_amc_checkpoints")
        return enabled, k, metric_key, dir_name

    def _topk_checkpoint_paths(self, dir_name: str):
        root_dir = os.path.join(self.config.trainer.default_local_dir, dir_name)
        metadata_path = os.path.join(root_dir, "topk_checkpoints.json")
        return root_dir, metadata_path

    def _load_topk_checkpoint_records(self, dir_name: str):
        root_dir, metadata_path = self._topk_checkpoint_paths(dir_name)
        if not os.path.exists(metadata_path):
            return []

        try:
            with open(metadata_path, "r") as f:
                records = json.load(f)
        except Exception as exc:
            print(f"[TopKCheckpoint] Failed to read {metadata_path}: {exc}. Starting from empty top-k state.")
            return []

        valid_records = []
        for record in records:
            path = record.get("path")
            if path and os.path.exists(path):
                valid_records.append(record)
        if len(valid_records) != len(records):
            self._write_topk_checkpoint_records(dir_name, valid_records)
        return valid_records

    def _write_topk_checkpoint_records(self, dir_name: str, records: list[dict]):
        root_dir, metadata_path = self._topk_checkpoint_paths(dir_name)
        os.makedirs(root_dir, exist_ok=True)
        tmp_path = f"{metadata_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(records, f, indent=2, sort_keys=True)
        os.replace(tmp_path, metadata_path)

    def _safe_remove_topk_checkpoint(self, root_dir: str, checkpoint_path: str):
        root_real = os.path.realpath(root_dir)
        path_real = os.path.realpath(checkpoint_path)
        if os.path.commonpath([root_real, path_real]) != root_real:
            raise ValueError(f"Refusing to remove checkpoint outside top-k root: {checkpoint_path}")
        if os.path.exists(path_real):
            shutil.rmtree(path_real)
            print(f"[TopKCheckpoint] Removed old top-k checkpoint: {path_real}")

    def _maybe_save_topk_checkpoint(self, val_metrics: dict):
        enabled, k, metric_key, dir_name = self._topk_checkpoint_config()
        if not enabled or k <= 0:
            return {}

        if metric_key not in val_metrics:
            print(
                f"[TopKCheckpoint] Metric {metric_key!r} not found at step {self.global_steps}; "
                "skipping top-k checkpoint."
            )
            return {
                "topk_checkpoint/enabled": 1.0,
                "topk_checkpoint/metric_available": 0.0,
            }

        try:
            score = float(val_metrics[metric_key])
        except (TypeError, ValueError):
            print(f"[TopKCheckpoint] Metric {metric_key!r} is not numeric: {val_metrics[metric_key]!r}")
            return {
                "topk_checkpoint/enabled": 1.0,
                "topk_checkpoint/metric_available": 0.0,
            }
        if not np.isfinite(score):
            print(f"[TopKCheckpoint] Metric {metric_key!r} is not finite: {score}")
            return {
                "topk_checkpoint/enabled": 1.0,
                "topk_checkpoint/metric_available": 0.0,
            }

        root_dir, _ = self._topk_checkpoint_paths(dir_name)
        records = self._load_topk_checkpoint_records(dir_name)
        same_step_records = [
            record for record in records
            if int(record.get("step", -1)) == int(self.global_steps)
        ]
        for record in same_step_records:
            self._safe_remove_topk_checkpoint(root_dir, record["path"])
        records = [
            record for record in records
            if int(record.get("step", -1)) != int(self.global_steps)
        ]
        records.sort(key=lambda record: (float(record["score"]), int(record["step"])), reverse=True)

        should_save = len(records) < k or score > float(records[-1]["score"])
        metrics = {
            "topk_checkpoint/enabled": 1.0,
            "topk_checkpoint/metric_available": 1.0,
            "topk_checkpoint/current_score": score,
            "topk_checkpoint/saved": 0.0,
        }
        if not should_save:
            metrics["topk_checkpoint/worst_kept_score"] = float(records[-1]["score"])
            print(
                f"[TopKCheckpoint] Step {self.global_steps} score={score:.6f} did not enter "
                f"top-{k}; worst kept={float(records[-1]['score']):.6f}."
            )
            return metrics

        folder_name = os.path.join(dir_name, f"global_step_{self.global_steps}_score_{score:.6f}")
        local_checkpoint_path = os.path.join(self.config.trainer.default_local_dir, folder_name)
        if os.path.exists(local_checkpoint_path):
            self._safe_remove_topk_checkpoint(root_dir, local_checkpoint_path)

        print(
            f"[TopKCheckpoint] Saving step {self.global_steps} with {metric_key}={score:.6f} "
            f"to {local_checkpoint_path}"
        )
        self._save_checkpoint(checkpoint_folder_name=folder_name, update_latest=False)

        records.append(
            {
                "step": int(self.global_steps),
                "score": score,
                "metric": metric_key,
                "path": local_checkpoint_path,
            }
        )
        records.sort(key=lambda record: (float(record["score"]), int(record["step"])), reverse=True)
        kept_records = records[:k]
        pruned_records = records[k:]
        for record in pruned_records:
            self._safe_remove_topk_checkpoint(root_dir, record["path"])
        self._write_topk_checkpoint_records(dir_name, kept_records)

        rank = next(
            idx + 1
            for idx, record in enumerate(kept_records)
            if int(record["step"]) == int(self.global_steps)
        )
        metrics.update(
            {
                "topk_checkpoint/saved": 1.0,
                "topk_checkpoint/rank": float(rank),
                "topk_checkpoint/best_score": float(kept_records[0]["score"]),
                "topk_checkpoint/worst_kept_score": float(kept_records[-1]["score"]),
                "topk_checkpoint/num_kept": float(len(kept_records)),
            }
        )
        return metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        
        # load checkpoint before doing anything
        self._load_checkpoint()
            

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate(logger)
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        is_best_step = False
        best_val_metrics = 0

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        prompt_uid2scores = defaultdict(lambda: {'scores': [], 'sources': str})
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in (self.config.trainer.profile_steps or [])
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # empo batch info
                new_batch.meta_info["do_vote"] = False
                if self.use_empo:
                    self.config.actor_rollout_ref.rollout.n = self.n_votes_per_prompt
                    new_batch.meta_info["do_vote"] = True

                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                n_samples = self.config.actor_rollout_ref.rollout.n
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=False)
                gen_batch.meta_info['is_train'] = True
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    skip_empty_training_batch = False
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        indices = torch.arange(len(gen_batch_output)).reshape(self.config.actor_rollout_ref.rollout.n, -1).T.reshape(-1)
                        gen_batch_output.reorder(indices)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                            empo_info = reward_result.get("empo_info") or {}
                            if "num_short_filtered" in empo_info:
                                metrics["selection/num_short_filtered"] = float(empo_info["num_short_filtered"])
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # recorde the scores(rollout n times) for each prompt uid
                        for uid, score, source in zip(new_batch.non_tensor_batch['uid'], new_batch.batch['token_level_scores'].sum(-1), new_batch.non_tensor_batch['data_source']):
                            prompt_uid2scores[uid]['scores'].append(score)
                            prompt_uid2scores[uid]['sources'] = source
                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        prompt_uid2metric_max = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            numeric_metric_vals = [float(val) for val in metric_vals]
                            prompt_uid2metric_std[prompt_uid] = np.std(numeric_metric_vals)
                            prompt_uid2metric_max[prompt_uid] = np.max(numeric_metric_vals)
                        
                        lower = self.config.algorithm.filter_groups.confidence_lower_bound
                        upper = self.config.algorithm.filter_groups.confidence_upper_bound
                        kept_prompt_uids = [uid for uid, max in prompt_uid2metric_max.items() if (max >= lower and max <= upper) or len(prompt_uid2metric_vals[uid]) == 1]
                        
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"num_prompt_in_batch={num_prompt_in_batch}, {num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]
                    
                    sample_size = min(len(batch), 8)

                    if self.use_empo:
                        table_data = [
                            [
                                batch.non_tensor_batch['index'].tolist()[i],
                                batch.non_tensor_batch['prompt'].tolist()[i],
                                batch.non_tensor_batch['response'].tolist()[i],
                                batch.non_tensor_batch['extracted_pred'].tolist()[i],
                                batch.non_tensor_batch['estimated_label'].tolist()[i],
                                batch.non_tensor_batch['extracted_gt'].tolist()[i], 
                                batch.non_tensor_batch['score'].tolist()[i],
                            ]
                            for i in range(sample_size)
                        ]
                    
                        logger.log({
                            "generated_examples": wandb.Table(
                                columns=['index', 'prompt', 'response', 'extracted_answer', 'pseudo_label', 'gold_answer', 'raw_reward'],
                                data=table_data,
                            )
                        }, step=self.global_steps)
                    else:
                        table_data = [
                            [
                                batch.non_tensor_batch['prompt'].tolist()[i],
                                batch.non_tensor_batch['response'].tolist()[i],
                                batch.non_tensor_batch['extracted_pred'].tolist()[i],
                                batch.non_tensor_batch['extracted_gt'].tolist()[i],
                                batch.non_tensor_batch['score'][i],
                                batch[i].non_tensor_batch['data_source'],
                                batch[i].non_tensor_batch['scored_by']
                            ]
                            for i in np.random.choice(len(batch), sample_size, replace=False)
                        ]
                        logger.log({
                            "generated_examples": wandb.Table(
                                columns=['prompt', 'completion', 'extracted_answer', 'gold_answer', 'raw_reward', 'data_source', 'scored_by'],
                                data=table_data,
                            )
                        }, step=self.global_steps, backend='wandb')
                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    metrics.update(apply_selected_training_mask(batch))
                    if batch.batch["response_mask"].sum().item() == 0:
                        skip_empty_training_batch = True
                        metrics["selection/empty_training_batch"] = 1.0
                        print(
                            "[Trainer] No selected response tokens remain after applying the "
                            "training mask; skipping this generated batch without an optimizer "
                            "update. For balanced largest-vs-rest runs, this usually means every "
                            "prompt in the batch was skipped by the underfilled policy."
                        )
                    else:
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        # TODO: Decouple the DP balancing and mini-batching.
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        # Determine whether we can drop unselected sequences
                        # *before* old_log_prob (early drop).  This is safe when:
                        #   - entropy-aware advantage weighting is OFF (no need
                        #     for token_level_entropys over all M responses), AND
                        #   - KL-in-reward is OFF (apply_kl_penalty already ran
                        #     or was skipped), AND
                        #   - no reference policy (ref_log_prob not needed for
                        #     advantage computation).
                        # When any of these are active the full batch must survive
                        # through old_log_prob / ref / advantage, so we fall back
                        # to the late-drop path (before update_actor only).
                        _adv_weight_mode = OmegaConf.select(
                            self.config, "reward_model.reward_kwargs.adv_weight_mode", default="off"
                        )
                        _can_early_drop = (
                            _adv_weight_mode not in ("entropy_high_token_ratio",)
                            and not self.config.algorithm.get("use_kl_in_reward", False)
                            and not self.use_reference_policy
                            and not self.use_critic
                        )

                        def _drop_unselected(batch, label=""):
                            """Remove unselected sequences from *batch* in-place.
                            No-op when M == K (all selected) or when
                            selected_for_training is absent."""
                            sel = batch.non_tensor_batch.get("selected_for_training")
                            if sel is None:
                                return batch
                            _to_bool = lambda v: (
                                bool(v) if isinstance(v, (bool, np.bool_))
                                else str(v).lower() in {"1", "true", "yes", "on"}
                            )
                            mask = np.array([_to_bool(v) for v in sel], dtype=bool)
                            if mask.sum() < len(mask):
                                idxs = np.where(mask)[0].tolist()
                                pre = len(batch)
                                batch = batch[idxs]
                                print(
                                    f"[Trainer] Dropped {pre - len(batch)} unselected "
                                    f"sequences{label}; training on {len(batch)}"
                                )
                            return batch

                        if _can_early_drop:
                            # --- Early-drop path (oversampling, no entropy weighting) ---
                            # 1. Compute advantages on the FULL batch (GRPO needs
                            #    all M responses per prompt for mean/std).
                            with marked_timer("adv", timing_raw, "brown"):
                                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                                batch = compute_advantage(
                                    batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                )
                                # adv weighting is off in this branch, but call
                                # for safety (returns {} when mode is "off").
                                metrics.update(
                                    apply_optional_advantage_weighting(
                                        batch,
                                        adv_weight_mode=_adv_weight_mode,
                                        entropy_weight_top_fraction=OmegaConf.select(
                                            self.config, "reward_model.reward_kwargs.entropy_weight_top_fraction", default=0.2
                                        ),
                                        adv_weight_clip_min=OmegaConf.select(
                                            self.config, "reward_model.reward_kwargs.adv_weight_clip_min", default=0.5
                                        ),
                                        adv_weight_clip_max=OmegaConf.select(
                                            self.config, "reward_model.reward_kwargs.adv_weight_clip_max", default=1.5
                                        ),
                                    )
                                )

                            # 2. Drop unselected sequences.
                            batch = _drop_unselected(batch, " (early, before old_log_prob)")

                            # 3. Recompute old_log_probs on the reduced batch.
                            with marked_timer("old_log_prob", timing_raw, "blue"):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                batch.batch["token_level_entropys"] = entropys
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)

                        else:
                            # --- Late-drop path (original flow) ---
                            # old_log_prob on full batch (entropy weighting or
                            # KL penalty needs entropys/log_probs from all M).
                            with marked_timer("old_log_prob", timing_raw, "blue"):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                batch.batch["token_level_entropys"] = entropys
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)

                            if self.use_reference_policy:
                                with marked_timer("ref", timing_raw, "olive"):
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                    batch = batch.union(ref_log_prob)

                            if self.use_critic:
                                with marked_timer("values", timing_raw, "cyan"):
                                    values = self.critic_wg.compute_values(batch)
                                    batch = batch.union(values)

                            with marked_timer("adv", timing_raw, "brown"):
                                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                                batch = compute_advantage(
                                    batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                )
                                metrics.update(
                                    apply_optional_advantage_weighting(
                                        batch,
                                        adv_weight_mode=_adv_weight_mode,
                                        entropy_weight_top_fraction=OmegaConf.select(
                                            self.config, "reward_model.reward_kwargs.entropy_weight_top_fraction", default=0.2
                                        ),
                                        adv_weight_clip_min=OmegaConf.select(
                                            self.config, "reward_model.reward_kwargs.adv_weight_clip_min", default=0.5
                                        ),
                                        adv_weight_clip_max=OmegaConf.select(
                                            self.config, "reward_model.reward_kwargs.adv_weight_clip_max", default=1.5
                                        ),
                                    )
                                )

                            # Drop unselected sequences before parameter updates.
                            batch = _drop_unselected(batch, " (late, before update)")

                        # update critic
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, "pink"):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, "red"):
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        # validate
                        if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                            with marked_timer("testing", timing_raw, "green"):
                                val_metrics: dict = self._validate(logger)
                            metrics.update(val_metrics)
                            with marked_timer("save_topk_checkpoint", timing_raw, "green"):
                                metrics.update(self._maybe_save_topk_checkpoint(val_metrics))

                        if is_best_step:
                            with marked_timer("save_best_checkpoint", timing_raw, "green"):
                                self._save_checkpoint(is_best_step=True)
                                is_best_step = False

                        if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                            with marked_timer("save_checkpoint", timing_raw, "green"):
                                # self._save_checkpoint()
                                self._save_checkpoint(is_best_step=False)

                if skip_empty_training_batch:
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    metrics["train/num_gen_batches"] = num_gen_batches
                    logger.log(data=metrics, step=self.global_steps)
                    timing_raw = defaultdict(float)
                    batch = None
                    num_prompt_in_batch = 0
                    num_gen_batches = 0

                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                    continue

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                if self.use_empo:
                    metrics.update(compute_empo_metrics(batch=batch))

                if "overlong_reward" in batch.non_tensor_batch.keys():
                    metrics.update(compute_overlong_metrics(batch=batch))
                
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
