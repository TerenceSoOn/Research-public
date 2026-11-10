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
A unified tracking interface that supports logging data to different backend
"""
import random
import dataclasses
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Dict, Any, Optional


def _get_nested_config(config: Optional[Dict[str, Any]], *keys, default=None):
    current = config or {}
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


def _wandb_run_id_file_candidates(config: Optional[Dict[str, Any]]) -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("WANDB_RUN_ID_FILE")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    default_local_dir = _get_nested_config(config, "trainer", "default_local_dir")
    if default_local_dir:
        default_local_dir = Path(default_local_dir).expanduser()
        candidates.append(default_local_dir.parent / "wandb_run_id.txt")

    resume_from_path = _get_nested_config(config, "trainer", "resume_from_path")
    if resume_from_path:
        resume_from_path = Path(resume_from_path).expanduser()
        # .../output_dir/checkpoints/global_step_x -> .../output_dir/wandb_run_id.txt
        if len(resume_from_path.parents) >= 2:
            candidates.append(resume_from_path.parents[1] / "wandb_run_id.txt")

    deduped: list[Path] = []
    seen = set()
    for path in candidates:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _resolve_wandb_resume(config: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str], Optional[Path]]:
    trainer_cfg = _get_nested_config(config, "trainer", default={}) or {}
    resume_mode = trainer_cfg.get("resume_mode", "disable")
    run_id_files = _wandb_run_id_file_candidates(config)
    write_path = run_id_files[0] if run_id_files else None

    if resume_mode not in {"resume_path", "auto"}:
        return None, None, write_path

    for path in run_id_files:
        try:
            if not path.exists():
                continue
            run_id = path.read_text(encoding="utf-8").strip()
            if run_id:
                return run_id, "must", write_path
        except OSError:
            continue

    return None, None, write_path


def _persist_wandb_run_id(run_id_file: Optional[Path], run_id: str) -> None:
    if run_id_file is None:
        return
    run_id_file.parent.mkdir(parents=True, exist_ok=True)
    run_id_file.write_text(f"{run_id}\n", encoding="utf-8")


class Tracking(object):
    supported_backend = ["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console"]

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None):
        self.cached_tables = {}
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"
        self.cached_tables = {}
        self.logger = {}

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import wandb
            wandb_mode = os.environ.get("WANDB_MODE", "online")
            wandb_dir = os.environ.get("WANDB_DIR", "./wandb")
            wandb_group = (
                os.environ.get("WANDB_GROUP")
                or os.environ.get("WANDB_RUN_GROUP")
                or _get_nested_config(config, "trainer", "wandb_group")
            )
            wandb_run_id, wandb_resume, wandb_run_id_file = _resolve_wandb_resume(config)
            init_kwargs = dict(
                project=project_name,
                name=experiment_name,
                config=config,
                mode=wandb_mode,
                dir=wandb_dir,
            )
            if wandb_group:
                init_kwargs["group"] = wandb_group
            if wandb_run_id:
                init_kwargs["id"] = wandb_run_id
                init_kwargs["resume"] = wandb_resume
                print(f"Resuming wandb run id {wandb_run_id} from {wandb_run_id_file}.")

            wandb.init(**init_kwargs)
            if getattr(wandb, "run", None) is not None and getattr(wandb.run, "id", None):
                _persist_wandb_run_id(wandb_run_id_file, wandb.run.id)
            self.logger['wandb'] = wandb
            # self.table = wandb.Table(columns=["step", "question", "answer", "extracted_answer", "pred", "extracted_pred", "acc"])
            # self.test_table = wandb.Table(columns=["step", "question", "answer", "extracted_answer", "pred", "extracted_pred", "acc"])

        if 'mlflow' in default_backend:
            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger['mlflow'] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "/mnt/shared-storage-user/p1-shared/zhangqingyang/swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "local")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten
            swanlab.init(project=project_name,
                         experiment_name=experiment_name,
                         config={
                             "FRAMEWORK": "veRL",
                             **config
                         },
                         logdir=SWANLAB_LOG_DIR,
                         mode=SWANLAB_MODE)
            self.logger["swanlab"] = swanlab

        if 'vemlp_wandb' in default_backend:
            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb
            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger['vemlp_wandb'] = vemlp_wandb

        if 'tensorboard' in default_backend:
            self.logger['tensorboard'] = _TensorboardAdapter()

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger

    '''
    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                if False:
                    import wandb
                    processed_data = {}
                    for key, value in data.items():
                        if 'distribution' in key and isinstance(value, dict):
                            current_columns = ["step"] + [k for k in value.keys()]
                            columns = current_columns
                            row = [step] + [v for v in value.values()]
                            if key not in self.cached_tables:
                                self.cached_tables[key] = {"columns": current_columns, "data": []}

                            if current_columns != self.cached_tables[key]["columns"]:
                                print(f"Warning: Column mismatch for {key}")
                                print(f"Existing columns: {self.cached_tables[key]['columns']}")
                                print(f"Current columns: {current_columns}")
                                
                                if len(current_columns) > len(self.cached_tables[key]["columns"]):
                                    new_columns = set(current_columns) - set(self.cached_tables[key]["columns"])
                                    old_data = self.cached_tables[key]["data"]
                                    updated_data = []
                                    
                                    for old_row in old_data:
                                        new_row = list(old_row)
                                        for _ in new_columns:
                                            new_row.append(0.0)
                                        updated_data.append(new_row)
                                    
                                    self.cached_tables[key] = {
                                        "columns": current_columns,
                                        "data": updated_data
                                    }
                                    
                                    self.cached_tables[key]["data"].append(row)
                                    
                                else:
                                    original_columns = self.cached_tables[key]["columns"]
                                    columns = original_columns
                                    reorganized_row = [step]
                                    
                                    for col in original_columns[1:]:
                                        reorganized_row.append(value.get(col, 0.0))
                                    
                                    self.cached_tables[key]["data"].append(reorganized_row)
                            else:
                                self.cached_tables[key]["data"].append(row)
                            try:
                                processed_data[key] = wandb.Table(columns=columns, data=self.cached_tables[key]["data"])
                            except:
                                pass
                        else:
                            processed_data[key] = value
                    print("log processed_data")
                    print(processed_data)
                    logger_instance.log(data=processed_data, step=step)
                else:
                    print("log row_data")
                    logger_instance.log(data=data, step=step)
    '''
    
    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)
    

    def __del__(self):
        if 'wandb' in self.logger:
            self.logger['wandb'].finish(exit_code=0)
        if 'swanlab' in self.logger:
            self.logger['swanlab'].finish()
        if 'vemlp_wandb' in self.logger:
            self.logger['vemlp_wandb'].finish(exit_code=0)
        if 'tensorboard' in self.logger:
            self.logger['tensorboard'].finish()


class _TensorboardAdapter:

    def __init__(self):
        from torch.utils.tensorboard import SummaryWriter
        import os
        # Use the current working directory's tensorboard_log folder by default
        # This ensures logs are kept within the project structure
        default_dir = os.path.join(os.getcwd(), "tensorboard_log")
        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        import numbers
        import numpy as np
        import torch
        for key, value in data.items():
            try:
                if isinstance(value, (numbers.Number,)):
                    self.writer.add_scalar(key, float(value), step)
                elif isinstance(value, np.ndarray) and value.ndim == 0:
                    self.writer.add_scalar(key, float(value), step)
                elif isinstance(value, torch.Tensor) and value.dim() == 0:
                    self.writer.add_scalar(key, float(value.item()), step)
                else:
                    # Skip non-scalars (e.g., wandb.Table) for TensorBoard backend
                    continue
            except Exception:
                # Be robust: ignore logging errors for individual keys
                continue

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:

    def log(self, data, step):
        import mlflow
        results = {k.replace('@', '_at_'): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep='/')


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {'list_len': len(x)} | {f'{i}': _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient='records')[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:

    def log(self, loggers, samples, step):
        if 'wandb' in loggers:
            self.log_generations_to_wandb(samples, step)
        if 'swanlab' in loggers:
            self.log_generations_to_swanlab(samples, step)
        if 'mlflow' in loggers:
            self.log_generations_to_mlflow(samples, step)

    def log_generations_to_wandb(self, samples, step):
        """Log samples to wandb as a table"""
        import wandb

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i+1}"))

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_text_list}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        #https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import mlflow
        import tempfile
        import json
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")
