"""
EMPO binary experiment main entry point.

Research-facing custom surface:
- `reward_manager=oversampled_empo` for the active binary 2x2 experiments

The active 2x2 surface uses first-round binary vote tags for reward and
switches only oversampling plus optional cluster-aware entropy advantage
weighting. Legacy EMPO and balanced-selection branches remain importable for
old checkpoints/scripts, but are not the active experiment surface.
"""

from recipe.empo.src.dapo_ray_trainer import RayDAPOTrainer

import os
import ray
import hydra


def _is_truthy(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path='config', config_name='edis_empo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN',
                    'VLLM_LOGGING_LEVEL': 'WARN'
                }
            }
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Some multimodal chat templates (e.g., Gemma-3) can emit image
        # placeholder positions even on text-only data, which breaks vLLM
        # request processing when no multimodal payload is provided.
        # Default to text-only tokenization for Gemma-3 unless explicitly
        # overridden.
        force_disable_processor = _is_truthy(os.environ.get("DISABLE_HF_PROCESSOR")) or _is_truthy(config.data.get("disable_processor", False))
        force_enable_processor = _is_truthy(os.environ.get("FORCE_ENABLE_HF_PROCESSOR"))
        auto_disable_gemma3_processor = _is_truthy(os.environ.get("AUTO_DISABLE_GEMMA3_PROCESSOR", "1"))
        model_path_lower = str(local_path).lower()

        if force_disable_processor:
            processor = None
            print("[Info] HF processor disabled by config/env (DISABLE_HF_PROCESSOR or data.disable_processor).")
        elif auto_disable_gemma3_processor and not force_enable_processor and "gemma-3" in model_path_lower:
            processor = None
            print(
                "[Info] Auto-disabled HF processor for Gemma-3 text-only training "
                "(set FORCE_ENABLE_HF_PROCESSOR=1 to override)."
            )

        # Worker classes (same as EMPO)
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # --- Reward manager selection ---
        reward_manager_name = config.reward_model.get("reward_manager", "oversampled_empo")

        if reward_manager_name in {"oversampled_empo", "edis_empo"}:
            from src.edis_reward_manager import EdisEmpoRewardManager
            reward_manager_cls = EdisEmpoRewardManager
        elif reward_manager_name == "empo":
            from verl.workers.reward_manager import EMPORewardManager
            reward_manager_cls = EMPORewardManager
        elif reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'dapo':
            from verl.workers.reward_manager import DAPORewardManager
            reward_manager_cls = DAPORewardManager
        else:
            raise NotImplementedError(
                f"Unknown reward_manager: {reward_manager_name}. "
                "Use `oversampled_empo` for the active EMPO-binary 2x2 "
                "experiments."
            )

        compute_score = get_custom_reward_fn(config)

        # Build reward_fn kwargs based on manager type
        if reward_manager_name in ("oversampled_empo", "edis_empo", "empo"):
            common_kwargs = dict(
                tokenizer=tokenizer,
                num_examine=1,
                compute_score=compute_score,
                n_votes_per_prompt=config.reward_model.reward_kwargs.n_votes_per_prompt,
                n_samples_per_prompt=config.reward_model.reward_kwargs.n_samples_per_prompt,
                reward_fn_key=config.data.reward_fn_key,
                eval_n_samples=config.actor_rollout_ref.rollout.val_kwargs.n,
                reward_type=config.reward_model.reward_kwargs.reward_type,
                entropy_thres=config.reward_model.reward_kwargs.entropy_thres,
                use_xverify=config.reward_model.use_xverify,
                use_eval_math=config.reward_model.get("use_eval_math", False),
            )

            if reward_manager_name in {"oversampled_empo", "edis_empo"}:
                reward_kwargs = config.reward_model.reward_kwargs
                common_kwargs["selected_top_k"] = reward_kwargs.selected_top_k
                common_kwargs["selection_family"] = reward_kwargs.selection_family
                common_kwargs["balanced_selection_method"] = reward_kwargs.get(
                    "balanced_selection_method", "D"
                )
                common_kwargs["reward_method"] = reward_kwargs.reward_method
                common_kwargs["adv_weight_mode"] = reward_kwargs.adv_weight_mode
                common_kwargs["entropy_weight_top_fraction"] = reward_kwargs.entropy_weight_top_fraction
                common_kwargs["adv_weight_clip_min"] = reward_kwargs.adv_weight_clip_min
                common_kwargs["adv_weight_clip_max"] = reward_kwargs.adv_weight_clip_max
                common_kwargs["largest_group_count"] = reward_kwargs.largest_group_count
                common_kwargs["rest_group_count"] = reward_kwargs.rest_group_count
                common_kwargs["underfilled_policy"] = reward_kwargs.underfilled_policy
                common_kwargs["min_response_length"] = reward_kwargs.get("min_response_length", 0)

            reward_fn = reward_manager_cls(mode="train", **common_kwargs)
            val_reward_fn = reward_manager_cls(mode="eval", **common_kwargs)
        else:
            reward_fn = reward_manager_cls(
                tokenizer=tokenizer, num_examine=1, compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.overlong_buffer,
                use_xverify=config.reward_model.use_xverify,
            )
            val_reward_fn = reward_manager_cls(
                tokenizer=tokenizer, num_examine=1, compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.overlong_buffer,
                use_xverify=config.reward_model.use_xverify,
            )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()
