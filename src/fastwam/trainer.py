import ctypes
import copy
import gc
import logging
import hashlib
import json
import inspect
import os
import random
import re
import shutil
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.pin_memory = bool(cfg.get("pin_memory", False))
        train_data_cfg = cfg.get("data", {}).get("train", {})
        train_concat_multi_camera = str(train_data_cfg.get("concat_multi_camera", "")).strip().lower()
        self.use_robotwin_loader_branch = train_concat_multi_camera == "robotwin"
        self.loader_prefetch_factor = 1 if self.use_robotwin_loader_branch and self.num_workers > 0 else None
        self.num_epochs = int(cfg.num_epochs)
        self.configured_num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.configured_max_steps = int(max_steps) if max_steps is not None else None
        self.max_steps = self.configured_max_steps
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.eval_num_samples = int(cfg.get("eval_num_samples", 1))
        if self.eval_num_samples <= 0:
            raise ValueError(f"`eval_num_samples` must be >= 1, got {self.eval_num_samples}")
        self.eval_enable_video = bool(cfg.get("eval_enable_video", True))
        self.eval_save_video = bool(cfg.get("eval_save_video", True))
        self.eval_enable_action_metrics = bool(cfg.get("eval_enable_action_metrics", True))
        self.pre_save_cleanup = bool(cfg.get("pre_save_cleanup", True))
        self.pre_save_cleanup_sleep_seconds = float(cfg.get("pre_save_cleanup_sleep_seconds", 5.0))
        self.pre_save_cleanup_malloc_trim = bool(cfg.get("pre_save_cleanup_malloc_trim", True))
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.lr_scheduler_type = str(cfg.lr_scheduler_type).strip().lower()
        
        self.resume = cfg.resume
        self.init_checkpoint = cfg.get("init_checkpoint", None)
        self.resume_training_state = cfg.get("resume_training_state", None)
        checkpoint_cfg = cfg.get("checkpoint", {})
        self.checkpoint_policy = str(checkpoint_cfg.get("policy", "auto")).strip().lower()
        self.checkpoint_save_latest = bool(checkpoint_cfg.get("save_latest", True))
        self.checkpoint_save_best_action_l1 = bool(checkpoint_cfg.get("save_best_action_l1", True))
        self.checkpoint_save_best_action_l2 = bool(checkpoint_cfg.get("save_best_action_l2", True))
        self.checkpoint_lightweight_resume_backend = str(
            checkpoint_cfg.get("lightweight_resume_backend", "trainable_only")
        ).strip().lower()
        self.checkpoint_trainable_only_include_optimizer_state = bool(
            checkpoint_cfg.get("trainable_only_include_optimizer_state", False)
        )
        if self.checkpoint_lightweight_resume_backend not in {"trainable_only", "full_state"}:
            raise ValueError(
                "`checkpoint.lightweight_resume_backend` must be one of "
                "['trainable_only', 'full_state'], "
                f"got {self.checkpoint_lightweight_resume_backend!r}."
            )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        ds_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage = (
            ds_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
            if ds_plugin is not None
            else "disabled"
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        if self.use_robotwin_loader_branch:
            logger.info(
                "Using robotwin loader branch: num_workers=%d prefetch_factor=%d",
                self.num_workers,
                self.loader_prefetch_factor,
            )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        self.local_eval_indices = self._build_local_eval_indices()
        if self.val_dataset is not None:
            logger.info(
                "Evaluation subset: per_rank_samples=%d local_indices=%s",
                self.eval_num_samples,
                self.local_eval_indices[: min(8, len(self.local_eval_indices))],
            )
        else:
            logger.info("Evaluation subset: val_dataset is None; skipping eval index initialization.")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps only the stage-selected modules trainable when ZeRO builds optimizer state.
        self._apply_training_stage_train_mode(self.model)
        trainable_params = list(self._get_trainable_params(self.model))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.total_train_steps = int(total_train_steps)
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.warmup_steps = int(warmup_steps)
        self.scheduler = self._build_scheduler(
            scheduler_type=self.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.best_state_root = os.path.join(self.state_dir, "best_refs")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        self.latest_training_path = os.path.join(self.checkpoint_root, "latest_training.pt")
        self.latest_state_dir = os.path.join(self.state_dir, "latest")
        self.best_action_l1_path = os.path.join(self.checkpoint_root, "best_action_l1.pt")
        self.best_action_l2_path = os.path.join(self.checkpoint_root, "best_action_l2.pt")
        self.best_action_l1 = None
        self.best_action_l2 = None
        self.last_eval_metrics = None
        self.base_checkpoint_path = None

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.best_state_root)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": bool(self.pin_memory and torch.cuda.is_available()),
            "worker_init_fn": worker_init_fn,
        }
        if self.loader_prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.loader_prefetch_factor
        return DataLoader(**loader_kwargs)

    @staticmethod
    def _best_effort_malloc_trim() -> None:
        try:
            libc = ctypes.CDLL("libc.so.6")
        except OSError:
            return

        trim_fn = getattr(libc, "malloc_trim", None)
        if trim_fn is None:
            return

        try:
            trim_fn(0)
        except Exception:
            logger.debug("best-effort malloc_trim failed", exc_info=True)

    def _run_pre_save_cleanup(self) -> None:
        if not self.pre_save_cleanup:
            return

        self.accelerator.wait_for_everyone()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.pre_save_cleanup_malloc_trim:
            self._best_effort_malloc_trim()
        self.accelerator.wait_for_everyone()
        if self.pre_save_cleanup_sleep_seconds > 0:
            time.sleep(self.pre_save_cleanup_sleep_seconds)
        self.accelerator.wait_for_everyone()

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_eval_infer_kwargs(
        self,
        *,
        sample: dict,
        prompt: str | None,
        input_image: torch.Tensor,
        num_frames: int,
        action: torch.Tensor | None,
        proprio: torch.Tensor | None,
    ) -> dict:
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample["action_horizon"],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt
        return infer_kwargs

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _build_local_eval_indices(self) -> list[int]:
        if self.val_dataset is None:
            return []

        val_len = len(self.val_dataset)
        if val_len <= 0:
            raise ValueError("`val_dataset` must contain at least one sample for evaluation.")

        world_size = max(int(self.accelerator.num_processes), 1)
        rng = torch.Generator(device="cpu").manual_seed(self.seed)
        shuffled = torch.randperm(val_len, generator=rng).tolist()
        local_indices = shuffled[self.accelerator.process_index::world_size][: self.eval_num_samples]
        if len(local_indices) < self.eval_num_samples:
            raise ValueError(
                "Evaluation subset is too small for the current world size. "
                f"val_len={val_len}, world_size={world_size}, "
                f"eval_num_samples={self.eval_num_samples}, rank={self.accelerator.process_index}, "
                f"local_count={len(local_indices)}"
            )
        return local_indices

    @staticmethod
    def _resolve_checkpoint_path(path_like):
        return str(Path(str(path_like)).expanduser().resolve())

    @staticmethod
    def _normalize_for_serialization(value):
        if isinstance(value, dict):
            return {str(k): Wan22Trainer._normalize_for_serialization(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Wan22Trainer._normalize_for_serialization(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _drop_resume_dynamic_keys(value):
        drop_keys = {"hydra", "output_dir", "resume", "resume_training_state", "init_checkpoint", "wandb"}
        if isinstance(value, dict):
            normalized = {}
            for k, v in value.items():
                key = str(k)
                if key in drop_keys:
                    continue
                if key == "checkpoint" and isinstance(v, dict):
                    checkpoint_cfg = {
                        str(inner_k): Wan22Trainer._drop_resume_dynamic_keys(inner_v)
                        for inner_k, inner_v in v.items()
                        if str(inner_k) != "lightweight_resume_backend"
                    }
                    normalized[key] = checkpoint_cfg
                    continue
                normalized[key] = Wan22Trainer._drop_resume_dynamic_keys(v)
            return normalized
        if isinstance(value, list):
            return [Wan22Trainer._drop_resume_dynamic_keys(v) for v in value]
        return value

    @staticmethod
    def _drop_resume_runtime_only_keys(value, prefix: str = ""):
        drop_paths = {"pin_memory"}
        if isinstance(value, dict):
            normalized = {}
            for k, v in value.items():
                key = str(k)
                key_path = f"{prefix}.{key}" if prefix else key
                if key_path in drop_paths:
                    continue
                normalized[key] = Wan22Trainer._drop_resume_runtime_only_keys(v, prefix=key_path)
            return normalized
        if isinstance(value, list):
            return [Wan22Trainer._drop_resume_runtime_only_keys(v, prefix=prefix) for v in value]
        return value

    @staticmethod
    def _flatten_nested_dict(value, prefix: str = "") -> dict[str, object]:
        flat: dict[str, object] = {}
        if isinstance(value, dict):
            for key in sorted(value.keys()):
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                flat.update(Wan22Trainer._flatten_nested_dict(value[key], prefix=next_prefix))
            return flat
        flat[prefix] = value
        return flat

    def _build_dataset_resume_meta(self, dataset) -> dict | None:
        if dataset is None:
            return None
        return {
            "class_name": type(dataset).__name__,
            "length": int(len(dataset)),
        }

    def _capture_rng_state(self) -> dict:
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return rng_state

    def _restore_rng_state(self, rng_state: dict | None, *, source: str):
        if not rng_state:
            logger.warning("Resume metadata from %s has no `rng_state`; RNG restore is skipped.", source)
            return

        random_state = rng_state.get("python")
        if random_state is not None:
            random.setstate(random_state)

        numpy_state = rng_state.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)

        torch_cpu_state = rng_state.get("torch_cpu")
        if torch_cpu_state is not None:
            torch.set_rng_state(torch_cpu_state)

        torch_cuda_state = rng_state.get("torch_cuda")
        if torch_cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda_state)

    def _build_resume_meta(self, *, include_rng_state: bool) -> dict:
        model = self.accelerator.unwrap_model(self.model)
        resolved_cfg = self._normalize_for_serialization(
            OmegaConf.to_container(self.cfg, resolve=True)
        )
        sanitized_cfg = self._drop_resume_dynamic_keys(resolved_cfg)
        digest_cfg = self._drop_resume_runtime_only_keys(sanitized_cfg)
        config_bytes = json.dumps(digest_cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        critical_fingerprint = {
            "batch_size": int(self.batch_size),
            "num_workers": int(self.num_workers),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "max_grad_norm": float(self.max_grad_norm),
            "lr_scheduler_type": str(self.lr_scheduler_type),
            "configured_num_epochs": int(self.configured_num_epochs),
            "configured_max_steps": self.configured_max_steps,
            "effective_max_steps": int(self.max_steps),
            "warmup_steps": int(self.warmup_steps),
            "mixed_precision": str(self.mixed_precision),
            "seed": int(self.seed),
            "world_size": int(self.accelerator.num_processes),
            "checkpoint_policy": str(self.checkpoint_policy),
            "train_dataset": self._build_dataset_resume_meta(self.train_dataset),
            "val_dataset": self._build_dataset_resume_meta(self.val_dataset),
            "pfd_enabled": bool(getattr(model, "pfd_enabled", False)),
            "pfd_stage": str(getattr(model, "pfd_stage", "none")),
            "pfd_training_mode": str(getattr(model, "pfd_training_mode", "none")),
        }
        if str(getattr(model, "pfd_training_mode", "")).lower() == "action512_partial":
            critical_fingerprint["partial_unfreeze"] = {
                "action_last_layers": int(getattr(model, "pfd_partial_action_last_layers", 0)),
                "video_last_layers": int(getattr(model, "pfd_partial_video_last_layers", 0)),
            }

        meta = {
            "schema_version": 2,
            "config_digest": hashlib.sha256(config_bytes).hexdigest(),
            "resolved_cfg": sanitized_cfg,
            "critical_fingerprint": critical_fingerprint,
        }
        if include_rng_state:
            meta["rng_state"] = self._capture_rng_state()
        return meta

    def _validate_resume_meta(self, resume_meta: dict | None, *, source: str):
        if resume_meta is None:
            logger.warning(
                "Resume source %s has no `resume_meta`; strict config validation is skipped.",
                source,
            )
            return

        current_meta = self._build_resume_meta(include_rng_state=False)
        saved_digest = resume_meta.get("config_digest")
        current_digest = current_meta["config_digest"]
        saved_resolved_cfg = resume_meta.get("resolved_cfg")
        current_resolved_cfg = current_meta.get("resolved_cfg")
        saved_fingerprint = resume_meta.get("critical_fingerprint")
        current_fingerprint = current_meta["critical_fingerprint"]

        config_diffs: list[str] = []
        if isinstance(saved_resolved_cfg, dict) and isinstance(current_resolved_cfg, dict):
            flat_saved_cfg = self._flatten_nested_dict(
                self._drop_resume_runtime_only_keys(saved_resolved_cfg)
            )
            flat_current_cfg = self._flatten_nested_dict(
                self._drop_resume_runtime_only_keys(current_resolved_cfg)
            )
            for key in sorted(set(flat_saved_cfg) | set(flat_current_cfg)):
                if flat_saved_cfg.get(key) != flat_current_cfg.get(key):
                    config_diffs.append(
                        f"{key}: saved={flat_saved_cfg.get(key)!r}, current={flat_current_cfg.get(key)!r}"
                    )

        fingerprint_diffs: list[str] = []
        if isinstance(saved_fingerprint, dict):
            flat_saved = self._flatten_nested_dict(saved_fingerprint)
            flat_current = self._flatten_nested_dict(current_fingerprint)
            for key in sorted(set(flat_saved) | set(flat_current)):
                if flat_saved.get(key) != flat_current.get(key):
                    fingerprint_diffs.append(
                        f"{key}: saved={flat_saved.get(key)!r}, current={flat_current.get(key)!r}"
                    )

        if config_diffs:
            details = config_diffs + fingerprint_diffs
            raise ValueError(
                "Strict resume config validation failed for %s. %s"
                % (source, " | ".join(details[:20]))
            )

        if saved_digest is not None and saved_digest != current_digest:
            if not isinstance(saved_resolved_cfg, dict) or not isinstance(current_resolved_cfg, dict):
                raise ValueError(
                    "Strict resume validation failed for %s. "
                    "Current config does not match the saved training state. "
                    "config_digest: saved=%s, current=%s"
                    % (source, saved_digest, current_digest)
                )
            logger.warning(
                "Resume config digest mismatch for %s is limited to allowed runtime-only fields; "
                "continuing. saved=%s current=%s",
                source,
                saved_digest,
                current_digest,
            )

        if fingerprint_diffs:
            raise ValueError(
                "Strict resume fingerprint validation failed for %s. %s"
                % (source, " | ".join(fingerprint_diffs[:20]))
            )

    def _uses_lightweight_pfd_checkpoints(self, model=None) -> bool:
        if self.checkpoint_policy == "full":
            return False

        model = self.accelerator.unwrap_model(self.model) if model is None else model
        if self.checkpoint_policy == "lightweight":
            return True

        return (
            bool(getattr(model, "pfd_enabled", False))
            and str(getattr(model, "pfd_stage", "none")).lower() == "s1"
        )

    def _uses_trainable_only_lightweight_resume(self, model=None) -> bool:
        return (
            self._uses_lightweight_pfd_checkpoints(model=model)
            and self._is_deepspeed_training()
            and self.checkpoint_lightweight_resume_backend == "trainable_only"
        )

    def _is_deepspeed_training(self) -> bool:
        return str(self.accelerator.distributed_type).lower().endswith("deepspeed")

    def _get_underlying_optimizer(self):
        return getattr(self.optimizer, "optimizer", self.optimizer)

    def _sync_optimizer_lrs_from_scheduler(self, *, source: str) -> None:
        last_lrs: list[float] = []
        try:
            last_lrs = [float(lr) for lr in self.scheduler.get_last_lr()]
        except Exception:
            state_dict = self.scheduler.state_dict() if hasattr(self.scheduler, "state_dict") else {}
            raw_last_lrs = state_dict.get("_last_lr", [])
            last_lrs = [float(lr) for lr in raw_last_lrs]

        if len(last_lrs) == 0:
            logger.warning("Scheduler restored from %s but no lr values were available to sync.", source)
            return

        updated = False
        candidate_optimizers = [self.optimizer]
        underlying_optimizer = self._get_underlying_optimizer()
        if underlying_optimizer is not self.optimizer:
            candidate_optimizers.append(underlying_optimizer)

        for optimizer in candidate_optimizers:
            param_groups = getattr(optimizer, "param_groups", None)
            if param_groups is None:
                continue
            if len(param_groups) != len(last_lrs):
                logger.warning(
                    "Skipping optimizer lr sync for %s because param_group count mismatched: groups=%d lrs=%d",
                    source,
                    len(param_groups),
                    len(last_lrs),
                )
                continue
            for group, lr in zip(param_groups, last_lrs):
                group["lr"] = float(lr)
            updated = True

        if updated:
            logger.info("Synchronized optimizer lr(s) from scheduler after %s: %s", source, last_lrs)
        else:
            logger.warning("Failed to synchronize optimizer lr(s) from scheduler after %s.", source)

    def _should_include_optimizer_state_in_trainable_resume(self) -> bool:
        return (
            self._uses_trainable_only_lightweight_resume()
            and self.checkpoint_trainable_only_include_optimizer_state
        )

    def _load_init_checkpoint(self, checkpoint_path: str):
        resolved_path = self._resolve_checkpoint_path(checkpoint_path)
        if not Path(resolved_path).exists():
            raise FileNotFoundError(f"Init checkpoint not found: {checkpoint_path}")
        logger.info("Loading init checkpoint weights only: %s", resolved_path)
        self.accelerator.unwrap_model(self.model).load_checkpoint(resolved_path, optimizer=None)
        self.base_checkpoint_path = resolved_path

    def _build_pfd_training_payload(
        self,
        metrics: dict | None = None,
        *,
        full_state_path: str | None = None,
        trainable_state_path: str | None = None,
        include_model_payload: bool = True,
        include_optimizer_state: bool | None = None,
    ) -> dict:
        model = self.accelerator.unwrap_model(self.model)
        if include_optimizer_state is None:
            include_optimizer_state = not self._is_deepspeed_training()
        payload = model.build_pfd_training_payload() if include_model_payload else {}
        payload.update(
            {
                "checkpoint_format": "pfd_training_state_v2",
                "global_step": int(self.global_step),
                "epoch": int(self.epoch),
                "batch_in_epoch": int(self.batch_in_epoch),
                "scheduler": self._materialize_checkpoint_value(self.scheduler.state_dict()),
                "base_checkpoint_path": self.base_checkpoint_path,
                "best_action_l1": self.best_action_l1,
                "best_action_l2": self.best_action_l2,
                "last_eval_metrics": dict(self.last_eval_metrics) if self.last_eval_metrics is not None else None,
                "current_eval_metrics": dict(metrics) if metrics is not None else None,
                "resume_meta": self._build_resume_meta(include_rng_state=True),
            }
        )
        if include_optimizer_state:
            payload["optimizer"] = self._materialize_checkpoint_value(self.optimizer.state_dict())
        if full_state_path is not None:
            payload["full_state_path"] = self._resolve_checkpoint_path(full_state_path)
        if trainable_state_path is not None:
            payload["trainable_state_path"] = self._resolve_checkpoint_path(trainable_state_path)
        return payload

    def _save_pfd_training_payload(
        self,
        path: str,
        metrics: dict | None = None,
        *,
        full_state_path: str | None = None,
        trainable_state_path: str | None = None,
        include_model_payload: bool = True,
        include_optimizer_state: bool | None = None,
        payload: dict | None = None,
    ) -> str:
        if not self.accelerator.is_main_process:
            return path
        if payload is None:
            payload = self._build_pfd_training_payload(
                metrics=metrics,
                full_state_path=full_state_path,
                trainable_state_path=trainable_state_path,
                include_model_payload=include_model_payload,
                include_optimizer_state=include_optimizer_state,
            )
        self._atomic_torch_save(payload, path)
        return path

    @staticmethod
    def _atomic_torch_save(payload: dict, path: str) -> None:
        path_obj = Path(path)
        ensure_dir(str(path_obj.parent))
        tmp_path = path_obj.with_name(f"{path_obj.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path_obj)

    @staticmethod
    def _materialize_checkpoint_value(value: object):
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.device.type == "cpu":
                return tensor.clone()
            return tensor.to(device="cpu", copy=True)
        if isinstance(value, dict):
            materialized = value.__class__()
            metadata = getattr(value, "_metadata", None)
            if metadata is not None:
                materialized._metadata = copy.deepcopy(metadata)
            for key, item in value.items():
                materialized[key] = Wan22Trainer._materialize_checkpoint_value(item)
            return materialized
        if isinstance(value, list):
            return [Wan22Trainer._materialize_checkpoint_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(Wan22Trainer._materialize_checkpoint_value(item) for item in value)
        return copy.deepcopy(value)

    @staticmethod
    def _replace_with_link_or_copy(src: str, dst: str) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        if src_path.resolve() == dst_path.resolve():
            return

        ensure_dir(str(dst_path.parent))
        tmp_path = dst_path.with_name(f"{dst_path.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            os.link(src_path, tmp_path)
        except OSError:
            shutil.copy2(src_path, tmp_path)
        os.replace(tmp_path, dst_path)

    def _build_externalized_pfd_training_payload(
        self,
        state_path: str,
        *,
        metrics: dict | None = None,
        payload_path: str | None = None,
    ) -> dict:
        payload = self._build_pfd_training_payload(
            metrics=metrics,
            trainable_state_path=state_path,
            include_model_payload=False,
            include_optimizer_state=False,
        )
        payload["model_payload_format"] = "pfd_training_payload_external_v1"
        payload["model_payload_dir"] = "model_components"
        if payload_path is not None:
            payload["linked_payload_path"] = self._resolve_checkpoint_path(payload_path)
        return payload

    def _save_externalized_pfd_training_payload(
        self,
        payload_path: str,
        state_path: str,
        *,
        metrics: dict | None = None,
    ) -> str:
        payload = self._build_externalized_pfd_training_payload(
            state_path=state_path,
            metrics=metrics,
            payload_path=payload_path,
        )
        return self._save_pfd_training_payload(payload_path, payload=payload)

    def _make_best_state_dir(self) -> str:
        state_id = f"step_{self.global_step:06d}_{time.time_ns()}"
        return os.path.join(self.best_state_root, state_id)

    @classmethod
    def _clone_tree_with_links(cls, src_dir: str, dst_dir: str) -> None:
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        if not src_path.is_dir():
            raise FileNotFoundError(f"Cannot clone missing checkpoint tree: {src_dir}")
        if dst_path.exists():
            shutil.rmtree(dst_path)
        for root, dirs, files in os.walk(src_path):
            rel_root = Path(root).relative_to(src_path)
            target_root = dst_path / rel_root
            target_root.mkdir(parents=True, exist_ok=True)
            for dirname in dirs:
                (target_root / dirname).mkdir(parents=True, exist_ok=True)
            for filename in files:
                cls._replace_with_link_or_copy(str(Path(root) / filename), str(target_root / filename))

    def _rewrite_externalized_payload_references(
        self,
        state_path: str,
        *,
        linked_payload_path: str | None = None,
        metrics: dict | None = None,
    ) -> None:
        model_payload_path = os.path.join(state_path, "model_payload.pt")
        payload = torch.load(model_payload_path, map_location="cpu", weights_only=False)
        if payload.get("model_payload_format") == "pfd_training_payload_external_v1":
            payload["trainable_state_path"] = self._resolve_checkpoint_path(state_path)
            if linked_payload_path is not None:
                payload["linked_payload_path"] = self._resolve_checkpoint_path(linked_payload_path)
            if metrics is not None:
                payload["current_eval_metrics"] = dict(metrics)
            self._save_pfd_training_payload(model_payload_path, payload=payload)

    def _should_exclude_frozen_parameters_from_full_state(self) -> bool:
        return (
            self._is_deepspeed_training()
            and self._uses_lightweight_pfd_checkpoints()
            and bool(self.base_checkpoint_path)
        )

    def _save_pfd_full_training_state(
        self,
        state_path: str,
        *,
        metrics: dict | None = None,
        linked_payload_path: str | None = None,
    ) -> str:
        state_path = self._resolve_checkpoint_path(state_path)
        self.accelerator.wait_for_everyone()
        ensure_dir(state_path)
        exclude_frozen_parameters = self._should_exclude_frozen_parameters_from_full_state()
        save_state_kwargs = {}
        if exclude_frozen_parameters:
            save_state_kwargs["exclude_frozen_parameters"] = True
        self.accelerator.save_state(output_dir=state_path, **save_state_kwargs)
        if self.accelerator.is_main_process:
            self._save_trainer_state(
                state_path,
                metrics=metrics,
                linked_payload_path=linked_payload_path,
                deepspeed_exclude_frozen_parameters=exclude_frozen_parameters,
            )
        self.accelerator.wait_for_everyone()
        return state_path

    def _save_pfd_eval_checkpoints(self, metrics: dict | None) -> dict[str, str]:
        updates: dict[str, str] = {}
        if metrics is None:
            return updates

        best_checkpoint_targets: list[tuple[str, str]] = []
        if self.checkpoint_save_best_action_l1 and "action_l1" in metrics:
            action_l1 = float(metrics["action_l1"])
            if self.best_action_l1 is None or action_l1 < self.best_action_l1:
                self.best_action_l1 = action_l1
                best_checkpoint_targets.append(("best_action_l1", self.best_action_l1_path))

        if self.checkpoint_save_best_action_l2 and "action_l2" in metrics:
            action_l2 = float(metrics["action_l2"])
            if self.best_action_l2 is None or action_l2 < self.best_action_l2:
                self.best_action_l2 = action_l2
                best_checkpoint_targets.append(("best_action_l2", self.best_action_l2_path))

        latest_state_path = None
        if self.checkpoint_save_latest:
            if self._uses_trainable_only_lightweight_resume():
                latest_state_path = self._save_pfd_trainable_resume_state(
                    self.latest_state_dir,
                    metrics=metrics,
                    linked_payload_path=self.latest_training_path,
                )
            else:
                latest_state_path = self._save_pfd_full_training_state(
                    self.latest_state_dir,
                    metrics=metrics,
                    linked_payload_path=self.latest_training_path,
                )
            if self.accelerator.is_main_process:
                if self._uses_trainable_only_lightweight_resume():
                    self._save_externalized_pfd_training_payload(
                        self.latest_training_path,
                        latest_state_path,
                        metrics=metrics,
                    )
                else:
                    self._save_pfd_training_payload(
                        self.latest_training_path,
                        metrics=metrics,
                        full_state_path=latest_state_path,
                        include_model_payload=False,
                    )
            updates["latest"] = self.latest_training_path
            updates["state"] = latest_state_path

        if best_checkpoint_targets:
            primary_name, primary_path = best_checkpoint_targets[0]
            if self._uses_trainable_only_lightweight_resume():
                best_state_path = self._make_best_state_dir()
                if latest_state_path is not None:
                    if self.accelerator.is_main_process:
                        self._clone_tree_with_links(latest_state_path, best_state_path)
                        self._rewrite_externalized_payload_references(
                            best_state_path,
                            linked_payload_path=primary_path,
                            metrics=metrics,
                        )
                    self.accelerator.wait_for_everyone()
                else:
                    best_state_path = self._save_pfd_trainable_resume_state(
                        best_state_path,
                        metrics=metrics,
                        linked_payload_path=primary_path,
                    )
            else:
                best_state_path = self._make_best_state_dir()
                if latest_state_path is not None:
                    if self.accelerator.is_main_process:
                        self._clone_tree_with_links(latest_state_path, best_state_path)
                    self.accelerator.wait_for_everyone()
                else:
                    best_state_path = self._save_pfd_full_training_state(
                        best_state_path,
                        metrics=metrics,
                        linked_payload_path=primary_path,
                    )

            if self.accelerator.is_main_process:
                if self._uses_trainable_only_lightweight_resume():
                    self._save_externalized_pfd_training_payload(
                        primary_path,
                        best_state_path,
                        metrics=metrics,
                    )
                else:
                    self._save_pfd_training_payload(
                        primary_path,
                        metrics=metrics,
                        full_state_path=best_state_path,
                        include_model_payload=False,
                    )
            updates[primary_name] = primary_path
            for name, path in best_checkpoint_targets[1:]:
                if self.accelerator.is_main_process:
                    if self._uses_trainable_only_lightweight_resume():
                        self._save_externalized_pfd_training_payload(
                            path,
                            best_state_path,
                            metrics=metrics,
                        )
                    else:
                        self._save_pfd_training_payload(
                            path,
                            metrics=metrics,
                            full_state_path=best_state_path,
                            include_model_payload=False,
                        )
                updates[name] = path

        return updates

    def _save_pfd_latest_checkpoint(self, metrics: dict | None = None) -> str | None:
        if not self.checkpoint_save_latest:
            return None
        if self._uses_trainable_only_lightweight_resume():
            latest_state_path = self._save_pfd_trainable_resume_state(
                self.latest_state_dir,
                metrics=metrics,
                linked_payload_path=self.latest_training_path,
            )
        else:
            latest_state_path = self._save_pfd_full_training_state(
                self.latest_state_dir,
                metrics=metrics,
                linked_payload_path=self.latest_training_path,
            )
        if not self.accelerator.is_main_process:
            return self.latest_training_path
        if self._uses_trainable_only_lightweight_resume():
            self._save_externalized_pfd_training_payload(
                self.latest_training_path,
                latest_state_path,
                metrics=metrics,
            )
        else:
            self._save_pfd_training_payload(
                self.latest_training_path,
                metrics=metrics,
                full_state_path=latest_state_path,
                include_model_payload=False,
            )
        return self.latest_training_path

    def _resolve_embedded_state_path(self, state_path: str, *, payload_path: str) -> str:
        path = Path(str(state_path)).expanduser()
        if not path.is_absolute():
            path = Path(payload_path).resolve().parent / path
        return str(path.resolve())

    def _resolve_state_dir_entry_path(self, state_dir: str, entry_path: str) -> str:
        path = Path(str(entry_path)).expanduser()
        if not path.is_absolute():
            path = Path(state_dir).resolve() / path
        return str(path.resolve())

    @staticmethod
    def _checkpoint_tmp_dir(state_path: str) -> str:
        return f"{state_path}.tmp"

    def _write_rank_trainable_resume_shard(
        self,
        shard_path: str,
        *,
        include_optimizer_state: bool,
    ) -> None:
        shard_payload = {
            "format": "pfd_trainable_resume_rank_v2" if not include_optimizer_state else "pfd_trainable_resume_rank_v1",
            "process_index": int(self.accelerator.process_index),
            "world_size": int(self.accelerator.num_processes),
            "optimizer_state_included": bool(include_optimizer_state),
            "rng_state": self._capture_rng_state(),
        }
        if include_optimizer_state:
            shard_payload["optimizer_state_dict"] = self._materialize_checkpoint_value(self.optimizer.state_dict())
        self._atomic_torch_save(shard_payload, shard_path)

    def _save_pfd_trainable_resume_state(
        self,
        state_path: str,
        *,
        metrics: dict | None = None,
        linked_payload_path: str | None = None,
    ) -> str:
        state_path = self._resolve_checkpoint_path(state_path)
        tmp_state_path = self._checkpoint_tmp_dir(state_path)

        if self.accelerator.is_main_process:
            if os.path.isdir(tmp_state_path):
                shutil.rmtree(tmp_state_path)
            ensure_dir(tmp_state_path)
            ensure_dir(os.path.join(tmp_state_path, "model_components"))
            ensure_dir(os.path.join(tmp_state_path, "optimizer_shards"))
        self.accelerator.wait_for_everyone()

        include_optimizer_state = self._should_include_optimizer_state_in_trainable_resume()

        shard_path = os.path.join(
            tmp_state_path,
            "optimizer_shards",
            f"optimizer_rank_{self.accelerator.process_index:03d}.pt",
        )
        self._write_rank_trainable_resume_shard(
            shard_path,
            include_optimizer_state=include_optimizer_state,
        )

        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).save_pfd_training_payload_components(
                os.path.join(tmp_state_path, "model_components")
            )
            model_payload_path = os.path.join(tmp_state_path, "model_payload.pt")
            self._save_externalized_pfd_training_payload(
                payload_path=model_payload_path,
                state_path=state_path,
                metrics=metrics,
            )
            scheduler_path = os.path.join(tmp_state_path, "scheduler.pt")
            torch.save(self._materialize_checkpoint_value(self.scheduler.state_dict()), scheduler_path)
            self._save_trainer_state(
                tmp_state_path,
                metrics=metrics,
                linked_payload_path=linked_payload_path,
                deepspeed_exclude_frozen_parameters=self._should_exclude_frozen_parameters_from_full_state(),
                extra_payload={
                    "resume_backend": "trainable_only",
                    "resume_state_format": "pfd_trainable_resume_v3",
                    "model_payload_path": "model_payload.pt",
                    "model_payload_format": "pfd_training_payload_external_v1",
                    "model_payload_dir": "model_components",
                    "scheduler_path": "scheduler.pt",
                    "optimizer_shard_dir": "optimizer_shards",
                    "optimizer_state_included": bool(include_optimizer_state),
                    "optimizer_shard_count": int(self.accelerator.num_processes),
                    "optimizer_shard_pattern": "optimizer_rank_{rank:03d}.pt",
                },
            )
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            if os.path.isdir(state_path):
                shutil.rmtree(state_path)
            os.replace(tmp_state_path, state_path)
            self._rewrite_externalized_payload_references(
                state_path,
                linked_payload_path=linked_payload_path,
                metrics=metrics,
            )
        self.accelerator.wait_for_everyone()
        return state_path

    def _load_pfd_trainable_resume_state(self, state_dir: str, trainer_state_payload: dict | None = None):
        state_dir = self._resolve_checkpoint_path(state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if trainer_state_payload is None:
            if not state_file.exists():
                raise FileNotFoundError(f"Trainable resume state is missing {state_file}")
            with open(state_file, "r", encoding="utf-8") as f:
                trainer_state_payload = json.load(f)

        self._validate_resume_meta(trainer_state_payload.get("resume_meta"), source=str(state_file))

        expected_world_size = int(trainer_state_payload.get("optimizer_shard_count", self.accelerator.num_processes))
        if expected_world_size != int(self.accelerator.num_processes):
            raise ValueError(
                "Trainable resume shard-count mismatch: "
                f"saved={expected_world_size}, current={self.accelerator.num_processes}."
            )

        model_payload_path = self._resolve_state_dir_entry_path(
            state_dir,
            trainer_state_payload.get("model_payload_path", "model_payload.pt"),
        )
        model_payload = torch.load(model_payload_path, map_location="cpu", weights_only=False)

        base_checkpoint_path = model_payload.get(
            "base_checkpoint_path",
            trainer_state_payload.get("base_checkpoint_path", self.base_checkpoint_path),
        )
        if not base_checkpoint_path:
            raise ValueError(
                "Trainable resume state requires `base_checkpoint_path`, "
                f"but neither {model_payload_path} nor {state_file} provides one."
            )
        self._load_init_checkpoint(base_checkpoint_path)
        self.accelerator.unwrap_model(self.model).load_pfd_training_payload(model_payload)

        shard_dir = self._resolve_state_dir_entry_path(
            state_dir,
            trainer_state_payload.get("optimizer_shard_dir", "optimizer_shards"),
        )
        shard_pattern = str(
            trainer_state_payload.get("optimizer_shard_pattern", "optimizer_rank_{rank:03d}.pt")
        )
        shard_entries = []
        shard_sources = []
        for rank in range(expected_world_size):
            shard_path = os.path.join(shard_dir, shard_pattern.format(rank=rank))
            if not os.path.exists(shard_path):
                raise FileNotFoundError(
                    f"Expected trainable-resume shard for rank {rank} at {shard_path}, but it does not exist."
                )
            shard_entries.append(torch.load(shard_path, map_location="cpu", weights_only=False))
            shard_sources.append(shard_path)

        optimizer_state_included = trainer_state_payload.get("optimizer_state_included")
        shard_optimizer_state_available = all("optimizer_state_dict" in entry for entry in shard_entries)
        if optimizer_state_included is None:
            optimizer_state_included = shard_optimizer_state_available

        if optimizer_state_included and not shard_optimizer_state_available:
            missing_ranks = [
                rank for rank, entry in enumerate(shard_entries) if "optimizer_state_dict" not in entry
            ]
            raise KeyError(
                "Trainable resume expected optimizer state, but the following rank shards are missing "
                f"`optimizer_state_dict`: {missing_ranks}"
            )

        if optimizer_state_included:
            optimizer_state_list = [entry["optimizer_state_dict"] for entry in shard_entries]
            underlying_optimizer = self._get_underlying_optimizer()
            if self._is_deepspeed_training():
                underlying_optimizer.load_state_dict(
                    optimizer_state_list,
                    load_optimizer_states=True,
                    load_from_fp32_weights=True,
                )
            else:
                self.optimizer.load_state_dict(optimizer_state_list[self.accelerator.process_index])
        else:
            logger.warning(
                "Trainable resume state %s does not include optimizer state; optimizer will cold-start.",
                state_dir,
            )

        if "scheduler" in model_payload:
            self.scheduler.load_state_dict(model_payload["scheduler"])
            self._sync_optimizer_lrs_from_scheduler(source=f"{model_payload_path}::scheduler")
        else:
            scheduler_path = trainer_state_payload.get("scheduler_path")
            if scheduler_path is not None:
                resolved_scheduler_path = self._resolve_state_dir_entry_path(state_dir, scheduler_path)
                self.scheduler.load_state_dict(torch.load(resolved_scheduler_path, map_location="cpu", weights_only=False))
                self._sync_optimizer_lrs_from_scheduler(source=resolved_scheduler_path)
            else:
                logger.warning(
                    "Trainable resume payload %s has no `scheduler` state; scheduler was not restored.",
                    model_payload_path,
                )

        self.global_step = int(model_payload.get("global_step", trainer_state_payload["global_step"]))
        self.base_checkpoint_path = base_checkpoint_path
        self.best_action_l1 = model_payload.get("best_action_l1", trainer_state_payload.get("best_action_l1", self.best_action_l1))
        self.best_action_l2 = model_payload.get("best_action_l2", trainer_state_payload.get("best_action_l2", self.best_action_l2))
        self.last_eval_metrics = model_payload.get("last_eval_metrics", trainer_state_payload.get("last_eval_metrics", self.last_eval_metrics))

        self.epoch = int(model_payload.get("epoch", trainer_state_payload["epoch"]))
        self.batch_in_epoch = int(model_payload.get("batch_in_epoch", trainer_state_payload["batch_in_epoch"]))
        self.train_sampler.set_epoch(self.epoch)
        self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)

        local_rank_shard = shard_entries[self.accelerator.process_index]
        self._restore_rng_state(
            local_rank_shard.get("rng_state"),
            source=shard_sources[self.accelerator.process_index],
        )
        self.accelerator.wait_for_everyone()
        logger.info(
            "Restored trainable-only PFD resume state from %s at step=%d epoch=%d batch_in_epoch=%d",
            state_dir,
            self.global_step,
            self.epoch,
            self.batch_in_epoch,
        )

    def _load_pfd_training_state(self, checkpoint_path: str, init_checkpoint_override: str | None = None):
        resolved_path = self._resolve_checkpoint_path(checkpoint_path)
        checkpoint = Path(resolved_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"PFD training-state checkpoint not found: {checkpoint_path}")
        if checkpoint.is_dir():
            self.load_training_state(resolved_path)
            return

        payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
        self._validate_resume_meta(payload.get("resume_meta"), source=resolved_path)
        trainable_state_path = payload.get("trainable_state_path")
        if trainable_state_path:
            self._load_pfd_trainable_resume_state(
                self._resolve_embedded_state_path(trainable_state_path, payload_path=resolved_path)
            )
            return
        full_state_path = payload.get("full_state_path")
        if full_state_path:
            self.load_training_state(
                self._resolve_embedded_state_path(full_state_path, payload_path=resolved_path)
            )
            return
        if self._is_deepspeed_training():
            raise ValueError(
                "Cannot resume Deepspeed/ZeRO training from single-file PFD checkpoint "
                f"{resolved_path}: it has no `full_state_path`. Older lightweight PFD "
                "checkpoints only contain one rank's optimizer shard and are not a complete "
                "ZeRO resume artifact. Resume from a `checkpoints/state/...` directory or "
                "from a new `latest_training.pt` that references one."
            )

        base_checkpoint_path = init_checkpoint_override or payload.get("base_checkpoint_path")
        if not base_checkpoint_path:
            raise ValueError(
                "PFD training-state checkpoint is missing `base_checkpoint_path`; "
                "set `init_checkpoint=...` explicitly to resume."
            )
        self._load_init_checkpoint(base_checkpoint_path)

        self.accelerator.unwrap_model(self.model).load_pfd_training_payload(payload)

        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        else:
            logger.warning("PFD training-state checkpoint has no `optimizer` state; optimizer was not restored.")
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
            self._sync_optimizer_lrs_from_scheduler(source=resolved_path)
        else:
            logger.warning("PFD training-state checkpoint has no `scheduler` state; scheduler was not restored.")

        self.global_step = int(payload.get("global_step", 0))
        self.epoch = int(payload.get("epoch", 0))
        self.batch_in_epoch = int(payload.get("batch_in_epoch", 0))
        self.best_action_l1 = payload.get("best_action_l1")
        self.best_action_l2 = payload.get("best_action_l2")
        self.last_eval_metrics = payload.get("last_eval_metrics")
        self._restore_rng_state(payload.get("resume_meta", {}).get("rng_state"), source=resolved_path)
        self.train_sampler.set_epoch(self.epoch)
        self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
        self.accelerator.wait_for_everyone()
        logger.info(
            "Restored PFD training state from %s at step=%d epoch=%d batch_in_epoch=%d",
            resolved_path,
            self.global_step,
            self.epoch,
            self.batch_in_epoch,
        )

    def _resume_or_load_checkpoint(self):
        if self.resume is not None and (self.init_checkpoint is not None or self.resume_training_state is not None):
            logger.warning(
                "Legacy `resume` is ignored because `init_checkpoint` or `resume_training_state` was provided."
            )

        if self.resume_training_state:
            if not self._uses_lightweight_pfd_checkpoints():
                raise ValueError(
                    "`resume_training_state` is only supported when lightweight PFD checkpointing is enabled."
                )
            logger.info("Resuming lightweight PFD training state from file: %s", self.resume_training_state)
            self._load_pfd_training_state(
                checkpoint_path=str(self.resume_training_state),
                init_checkpoint_override=(
                    None if self.init_checkpoint in (None, "", "null") else str(self.init_checkpoint)
                ),
            )
            return

        if self.init_checkpoint:
            self._load_init_checkpoint(str(self.init_checkpoint))
            return

        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")
        self.base_checkpoint_path = self._resolve_checkpoint_path(resume_path)

    @staticmethod
    def _get_trainable_params(model):
        if hasattr(model, "get_trainable_parameters"):
            return model.get_trainable_parameters()

        trainable_params = list(model.dit.parameters())
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        return trainable_params

    @staticmethod
    def _apply_training_stage_train_mode(model):
        if hasattr(model, "apply_training_stage_train_mode"):
            logger.info("Applying model-defined training stage mode.")
            model.apply_training_stage_train_mode()
            return

        logger.info("Setting DiT to train mode and freezing other model components.")
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _get_optimizer_trainable_params(optimizer):
        params = []
        seen = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                if not isinstance(param, torch.Tensor) or not param.requires_grad:
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                params.append(param)
        return params

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        model.eval()

        local_metrics_rows = []
        video_path = None
        video_metrics_enabled = bool(self.eval_enable_video)
        action_metrics_enabled = bool(self.eval_enable_action_metrics)

        for eval_index in self.local_eval_indices:
            sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

            with self.accelerator.autocast():
                val_loss, _ = model.training_loss(sample)
                val_loss = val_loss.float().item()

            prompt = sample["prompt"][0]
            video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
            action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
            proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None
            input_image = video0[:, 0].unsqueeze(0)
            _, num_frames, _, _ = video0.shape
            infer_kwargs = self._build_eval_infer_kwargs(
                sample=sample,
                prompt=prompt,
                input_image=input_image,
                num_frames=num_frames,
                action=action,
                proprio=proprio,
            )

            sample_action_metrics_enabled = bool(self.eval_enable_action_metrics and action is not None)
            pred_action = None
            gt_video_tensor = None
            sample_video_path = None
            psnr_rollout_vs_gt = 0.0
            ssim_rollout_vs_gt = 0.0
            psnr_decode_vs_gt = 0.0
            ssim_decode_vs_gt = 0.0
            psnr_rollout_vs_decode = 0.0
            ssim_rollout_vs_decode = 0.0

            if video_metrics_enabled:
                pred = model.infer(**infer_kwargs)
                pred_video = pred["video"]
                pred_action = pred.get("action", None)

                pred_video_tensor = pil_frames_to_video_tensor(pred_video)
                gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

                assert pred_video_tensor.shape == gt_video_tensor.shape, (
                    "Eval infer prediction/GT shape mismatch: "
                    f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
                )

                psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
                ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)
            elif sample_action_metrics_enabled:
                pred = model.infer_action(
                    prompt=infer_kwargs["prompt"],
                    input_image=input_image,
                    action_horizon=infer_kwargs["action_horizon"],
                    proprio=infer_kwargs["proprio"],
                    context=infer_kwargs.get("context"),
                    context_mask=infer_kwargs.get("context_mask"),
                    text_cfg_scale=infer_kwargs["text_cfg_scale"],
                    num_inference_steps=infer_kwargs["num_inference_steps"],
                    seed=infer_kwargs["seed"],
                    tiled=infer_kwargs["tiled"],
                )
                pred_action = pred.get("action", None)

            action_l1 = None
            action_l2 = None
            if sample_action_metrics_enabled and pred_action is not None:
                if sample["proprio"] is None:
                    raise ValueError("Eval sample must contain `proprio` for action denormalization.")
                proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

                processor = self.val_dataset.lerobot_dataset.processor
                denorm_actions = {}
                action_meta = processor.shape_meta["action"]
                state_meta = processor.shape_meta["state"]
                for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                    if not isinstance(raw_action, torch.Tensor):
                        raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                    if raw_action.ndim == 2:
                        action_btd = raw_action.unsqueeze(0)
                    elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                        action_btd = raw_action
                    else:
                        raise ValueError(
                            f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                        )
                    action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                    batch = {
                        "action": action_btd,
                        "state": proprio,
                    }
                    batch = processor.action_state_merger.backward(batch)
                    batch = processor.normalizer.backward(batch)
                    merged_batch = {
                        "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                        "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                    }
                    merged_batch = processor.action_state_merger.forward(merged_batch)
                    denorm_action = merged_batch["action"].unsqueeze(0)
                    if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                        raise ValueError(
                            f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                        )
                    denorm_actions[action_name] = denorm_action

                pred_action_denorm = denorm_actions["pred"]
                gt_action_denorm = denorm_actions["gt"]

                if pred_action_denorm.shape != gt_action_denorm.shape:
                    raise ValueError(
                        "Predicted action/GT action shape mismatch after denormalization: "
                        f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                    )
                action_diff = pred_action_denorm - gt_action_denorm
                action_l1 = action_diff.abs().mean().item()
                action_l2 = action_diff.pow(2).mean().item()

            if video_metrics_enabled:
                gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
                vae_recon_video = model._decode_latents(vae_latents, tiled=False)
                vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

                assert vae_video_tensor.shape == gt_video_tensor.shape, (
                    "Eval VAE reconstruction/GT shape mismatch: "
                    f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
                )

                psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
                ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

                psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
                ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

                if self.eval_save_video and video_path is None:
                    stitched_video_tensor = torch.cat(
                        [pred_video_tensor, vae_video_tensor, gt_video_tensor],
                        dim=2,
                    ).contiguous()
                    stitched_frames = []
                    for t in range(stitched_video_tensor.shape[1]):
                        frame = (
                            stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
                        ).astype(np.uint8)
                        stitched_frames.append(Image.fromarray(frame))

                    sample_video_path = os.path.join(
                        self.eval_dir,
                        f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
                    )
                    save_mp4(stitched_frames, sample_video_path, fps=8)
                    video_path = sample_video_path

            local_metrics_rows.append(
                torch.tensor(
                    [
                        float(val_loss),
                        float(psnr_rollout_vs_gt),
                        float(ssim_rollout_vs_gt),
                        float(psnr_rollout_vs_decode),
                        float(ssim_rollout_vs_decode),
                        float(psnr_decode_vs_gt),
                        float(ssim_decode_vs_gt),
                        float(action_l2) if action_l2 is not None else -1.0,
                        float(action_l1) if action_l1 is not None else -1.0,
                    ],
                    device=self.accelerator.device,
                    dtype=torch.float32,
                )
            )
            action_metrics_enabled = action_metrics_enabled and sample_action_metrics_enabled

        if len(local_metrics_rows) == 0:
            raise RuntimeError("Evaluation produced no local metric rows.")

        local_metrics = torch.stack(local_metrics_rows, dim=0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        local_flags = torch.tensor(
            [
                1.0 if video_metrics_enabled else 0.0,
                1.0 if action_metrics_enabled else 0.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_flags = self.accelerator.gather_for_metrics(local_flags)
        flag_consistent = torch.all(gathered_flags == gathered_flags[0]).item()
        if not flag_consistent:
            raise RuntimeError(
                "Eval metric availability mismatch across ranks: "
                f"{gathered_flags.cpu().tolist()}"
            )

        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        has_video_metrics = bool(gathered_flags[0, 0].item() > 0.5)
        has_action_metrics = bool(gathered_flags[0, 1].item() > 0.5)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if has_action_metrics else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if has_action_metrics else None

        self._apply_training_stage_train_mode(model)

        result = {"val_loss": float(mean_metrics[0].item())}
        if has_video_metrics:
            result.update(
                {
                    "psnr_rg": float(mean_metrics[1].item()),
                    "ssim_rg": float(mean_metrics[2].item()),
                    "psnr_rd": float(mean_metrics[3].item()),
                    "ssim_rd": float(mean_metrics[4].item()),
                    "psnr_dg": float(mean_metrics[5].item()),
                    "ssim_dg": float(mean_metrics[6].item()),
                }
            )
            if video_path is not None:
                result["video_path"] = video_path
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(
        self,
        state_path: str,
        *,
        metrics: dict | None = None,
        linked_payload_path: str | None = None,
        deepspeed_exclude_frozen_parameters: bool = False,
        extra_payload: dict | None = None,
    ):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "base_checkpoint_path": self.base_checkpoint_path,
            "best_action_l1": self.best_action_l1,
            "best_action_l2": self.best_action_l2,
            "last_eval_metrics": dict(self.last_eval_metrics) if self.last_eval_metrics is not None else None,
            "current_eval_metrics": dict(metrics) if metrics is not None else None,
            "linked_payload_path": linked_payload_path,
            "deepspeed_exclude_frozen_parameters": bool(deepspeed_exclude_frozen_parameters),
            "resume_meta": self._build_resume_meta(include_rng_state=False),
        }
        if extra_payload:
            payload.update(extra_payload)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        trainer_state_payload = None
        load_state_kwargs = {}
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                trainer_state_payload = json.load(f)
            if trainer_state_payload.get("resume_backend") == "trainable_only":
                self._load_pfd_trainable_resume_state(state_dir, trainer_state_payload=trainer_state_payload)
                return
            self._validate_resume_meta(trainer_state_payload.get("resume_meta"), source=str(state_file))
            if bool(trainer_state_payload.get("deepspeed_exclude_frozen_parameters", False)):
                base_checkpoint_path = trainer_state_payload.get("base_checkpoint_path", self.base_checkpoint_path)
                if not base_checkpoint_path:
                    raise ValueError(
                        "Reduced DeepSpeed full-state resume requires `base_checkpoint_path`, "
                        f"but {state_file} does not provide one."
                    )
                self._load_init_checkpoint(base_checkpoint_path)
                load_state_kwargs["load_module_strict"] = False

        self.accelerator.load_state(input_dir=state_dir, **load_state_kwargs)
        if trainer_state_payload is not None:
            self.global_step = int(trainer_state_payload["global_step"])
            self.base_checkpoint_path = trainer_state_payload.get("base_checkpoint_path", self.base_checkpoint_path)
            self.best_action_l1 = trainer_state_payload.get("best_action_l1", self.best_action_l1)
            self.best_action_l2 = trainer_state_payload.get("best_action_l2", self.best_action_l2)
            self.last_eval_metrics = trainer_state_payload.get("last_eval_metrics", self.last_eval_metrics)

            if "epoch" in trainer_state_payload and "batch_in_epoch" in trainer_state_payload:
                self.epoch = int(trainer_state_payload["epoch"])
                self.batch_in_epoch = int(trainer_state_payload["batch_in_epoch"])
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.set_epoch(self.epoch)
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._apply_training_stage_train_mode(self.accelerator.unwrap_model(self.model))

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        self.train_sampler.set_epoch(self.epoch)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                self.train_sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self._get_optimizer_trainable_params(self.optimizer),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    did_eval_checkpoint = False
                    did_latest_checkpoint = False
                    latest_path_for_step = None

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        self.last_eval_metrics = dict(metrics) if metrics is not None else None
                        if metrics is not None and self.accelerator.is_main_process:
                            description_parts = [
                                f"[eval] step={self.global_step}",
                                f"val_loss={metrics['val_loss']:.4f}",
                            ]
                            if "psnr_rd" in metrics:
                                description_parts.append(f"infer_psnr={metrics['psnr_rd']:.4f}")
                            if "ssim_rd" in metrics:
                                description_parts.append(f"infer_ssim={metrics['ssim_rd']:.4f}")
                            if "action_l2" in metrics:
                                description_parts.append(f"action_l2={metrics['action_l2']:.4f}")
                            if "action_l1" in metrics:
                                description_parts.append(f"action_l1={metrics['action_l1']:.4f}")
                            logger.info(" ".join(description_parts))
                            eval_payload = {"eval/val_loss": float(metrics["val_loss"])}
                            for key in ("psnr_rg", "ssim_rg", "psnr_rd", "ssim_rd", "psnr_dg", "ssim_dg"):
                                if key in metrics:
                                    eval_payload[f"eval/{key}"] = float(metrics[key])
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                        if self._uses_lightweight_pfd_checkpoints(unwrapped_model):
                            self._run_pre_save_cleanup()
                            ckpt_updates = self._save_pfd_eval_checkpoints(metrics)
                            did_eval_checkpoint = True
                            latest_path_for_step = ckpt_updates.get("latest")
                            did_latest_checkpoint = latest_path_for_step is not None
                            self.accelerator.wait_for_everyone()
                            if ckpt_updates and self.accelerator.is_main_process:
                                logger.info(
                                    "[ckpt] step=%d %s",
                                    self.global_step,
                                    " ".join(f"{name}={path}" for name, path in sorted(ckpt_updates.items())),
                                )

                    if (
                        self.save_every > 0
                        and self.global_step % self.save_every == 0
                    ):
                        if self._uses_lightweight_pfd_checkpoints(unwrapped_model):
                            if not did_eval_checkpoint:
                                self._run_pre_save_cleanup()
                                latest_path = self._save_pfd_latest_checkpoint(metrics=self.last_eval_metrics)
                                latest_path_for_step = latest_path
                                did_latest_checkpoint = latest_path is not None
                                self.accelerator.wait_for_everyone()
                                if self.accelerator.is_main_process:
                                    logger.info(
                                        "[ckpt] step=%d latest=%s state=%s",
                                        self.global_step,
                                        latest_path,
                                        self.latest_state_dir,
                                    )
                        else:
                            ckpt_info = self.save_checkpoint()
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[ckpt] step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )

                    if self.global_step >= self.max_steps:
                        if self._uses_lightweight_pfd_checkpoints(unwrapped_model):
                            if did_latest_checkpoint:
                                latest_path = latest_path_for_step or self.latest_training_path
                            else:
                                self._run_pre_save_cleanup()
                                latest_path = self._save_pfd_latest_checkpoint(metrics=self.last_eval_metrics)
                                latest_path_for_step = latest_path
                                did_latest_checkpoint = latest_path is not None
                                self.accelerator.wait_for_everyone()
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[done] max_steps reached step=%d latest=%s",
                                    self.global_step,
                                    latest_path,
                                )
                        else:
                            ckpt_info = self.save_checkpoint()
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[done] max_steps reached step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )
                        return

        if self._uses_lightweight_pfd_checkpoints(unwrapped_model):
            latest_path = self._save_pfd_latest_checkpoint(metrics=self.last_eval_metrics)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] training finished step=%d latest=%s",
                    self.global_step,
                    latest_path,
                )
        else:
            ckpt_info = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] training finished step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
        
