#!/usr/bin/env python3
"""Reproducible Action-Expert LoRA workflow for the PARC pi0.5 submission.

The workflow deliberately does not consume PARC evaluation observations or
results.  It selects a task-balanced set of curated LIBERO demonstrations,
splits whole episodes into train/validation sets, invokes LeRobot's official
PEFT training path, merges the adapter into ordinary pi0.5 weights, and can
measure a deterministic held-out flow-matching loss.

Run this script with the Python 3.10 environment prepared by
``pi05_parc_colab_setup.py``.  See ``pi05_parc_colab.ipynb`` for the Colab
sequence and conservative NVIDIA L4 defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections import deque
from pathlib import Path
from typing import Any


FULL_LIBERO_REPO = "lerobot/libero"
FULL_LIBERO_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
DEFAULT_DATASET_REPO = FULL_LIBERO_REPO
DEFAULT_DATASET_REVISION = FULL_LIBERO_REVISION

# Only the separate action transformer and its action/time projections are
# adapted.  No PaliGemma vision or language module can match this expression.
ACTION_EXPERT_LORA_TARGETS = (
    r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|"
    r".*\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))"
)
LIGHT_COLOR_TRANSFORMS = {
    "brightness": {
        "weight": 1.0,
        "type": "ColorJitter",
        "kwargs": {"brightness": [0.9, 1.1]},
    },
    "contrast": {
        "weight": 1.0,
        "type": "ColorJitter",
        "kwargs": {"contrast": [0.9, 1.1]},
    },
    "saturation": {
        "weight": 1.0,
        "type": "ColorJitter",
        "kwargs": {"saturation": [0.9, 1.1]},
    },
}

CHECKPOINT_REQUIRED_FILES = (
    "pretrained_model/config.json",
    "pretrained_model/train_config.json",
    "pretrained_model/adapter_model.safetensors",
    "training_state/training_step.json",
    "training_state/rng_state.safetensors",
    "training_state/optimizer_state.safetensors",
    "training_state/optimizer_param_groups.json",
    "training_state/scheduler_state.json",
)


def recommended_batch_size(total_vram_gib: float) -> int:
    """Conservative physical batch for a single Colab GPU."""
    if total_vram_gib >= 35.0:
        return 4
    if total_vram_gib >= 22.0:
        return 2
    raise RuntimeError(
        f"pi0.5 LoRA requires an L4-class 24GB GPU or better; got {total_vram_gib:.1f} GiB"
    )


def checkpoint_step(checkpoint_dir: Path) -> int | None:
    """Return a checkpoint step only after every resume-critical file exists."""
    if not checkpoint_dir.is_dir() or not checkpoint_dir.name.isdigit():
        return None
    if any(not (checkpoint_dir / relative).is_file() for relative in CHECKPOINT_REQUIRED_FILES):
        return None
    state = json.loads(
        (checkpoint_dir / "training_state" / "training_step.json").read_text(
            encoding="utf-8"
        )
    )
    step = int(state["step"])
    if step != int(checkpoint_dir.name):
        raise ValueError(
            f"checkpoint name/metadata mismatch: {checkpoint_dir.name} != {step}"
        )
    return step


def discover_checkpoints(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    if not checkpoints_dir.is_dir():
        return []
    discovered: list[tuple[int, Path]] = []
    for path in checkpoints_dir.iterdir():
        if path.is_symlink():
            continue
        step = checkpoint_step(path)
        if step is not None:
            discovered.append((step, path))
    return sorted(discovered)


def sync_checkpoints(local_dir: Path, backup_dir: Path) -> list[Path]:
    """Atomically copy newly completed LoRA checkpoints to persistent storage."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for step, source in discover_checkpoints(local_dir):
        destination = backup_dir / source.name
        if checkpoint_step(destination) == step:
            continue
        partial_root = backup_dir / f".{source.name}.partial"
        partial = partial_root / source.name
        shutil.rmtree(partial_root, ignore_errors=True)
        print(f"CHECKPOINT_BACKUP_START step={step} -> {destination}", flush=True)
        started = time.monotonic()
        shutil.copytree(source, partial)
        if checkpoint_step(partial) != step:
            raise RuntimeError(f"incomplete checkpoint copy: {partial}")
        shutil.rmtree(destination, ignore_errors=True)
        partial.replace(destination)
        shutil.rmtree(partial_root, ignore_errors=True)
        elapsed = time.monotonic() - started
        print(f"CHECKPOINT_BACKUP_DONE step={step} elapsed={elapsed:.1f}s", flush=True)
        copied.append(destination)
    return copied


def restore_checkpoints(backup_dir: Path, local_dir: Path) -> list[Path]:
    """Restore every persistent checkpoint missing from the fast local run dir."""
    restored: list[Path] = []
    local_dir.mkdir(parents=True, exist_ok=True)
    for step, source in discover_checkpoints(backup_dir):
        destination = local_dir / source.name
        if checkpoint_step(destination) == step:
            continue
        partial_root = local_dir / f".{source.name}.restore"
        partial = partial_root / source.name
        shutil.rmtree(partial_root, ignore_errors=True)
        print(f"CHECKPOINT_RESTORE_START step={step} <- {source}", flush=True)
        shutil.copytree(source, partial)
        if checkpoint_step(partial) != step:
            raise RuntimeError(f"incomplete restored checkpoint: {partial}")
        shutil.rmtree(destination, ignore_errors=True)
        partial.replace(destination)
        shutil.rmtree(partial_root, ignore_errors=True)
        restored.append(destination)
        print(f"CHECKPOINT_RESTORE_DONE step={step}", flush=True)
    return restored


def _task_name(value: Any) -> str:
    """Normalize the task field stored in LeRobot episode metadata."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("episode has an empty task list")
        value = value[0]
    task = str(value).strip()
    if not task or task.lower() == "nan":
        raise ValueError(f"invalid episode task: {value!r}")
    return task


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_rows(episodes: Any):
    """Yield metadata rows from LeRobot's HF Dataset (or older DataFrame)."""
    if hasattr(episodes, "iterrows"):
        yield from episodes.iterrows()
        return
    for row_index in range(len(episodes)):
        yield row_index, episodes[row_index]


def balanced_episode_split(
    episodes_by_task: dict[str, list[int]],
    *,
    seed: int,
    validation_per_task: int,
    max_per_task: int | None,
) -> tuple[list[int], list[int], dict[str, dict[str, list[int]]]]:
    """Select equal episode counts per task and split without frame leakage."""
    if not episodes_by_task:
        raise ValueError("no episodes were found")
    if validation_per_task < 1:
        raise ValueError("validation_per_task must be at least 1")

    counts = {task: len(set(indices)) for task, indices in episodes_by_task.items()}
    smallest = min(counts.values())
    selected_per_task = smallest if max_per_task is None else min(smallest, max_per_task)
    if selected_per_task <= validation_per_task:
        raise ValueError(
            "not enough episodes per task: "
            f"selected={selected_per_task}, validation={validation_per_task}"
        )

    train: list[int] = []
    validation: list[int] = []
    split_by_task: dict[str, dict[str, list[int]]] = {}
    for task_index, task in enumerate(sorted(episodes_by_task)):
        indices = sorted(set(int(index) for index in episodes_by_task[task]))
        task_rng = random.Random(seed + task_index * 1_000_003)
        task_rng.shuffle(indices)
        selected = indices[:selected_per_task]
        task_validation = sorted(selected[:validation_per_task])
        task_train = sorted(selected[validation_per_task:])
        train.extend(task_train)
        validation.extend(task_validation)
        split_by_task[task] = {
            "train": task_train,
            "validation": task_validation,
        }

    train = sorted(train)
    validation = sorted(validation)
    if set(train) & set(validation):
        raise AssertionError("train and validation episodes overlap")
    return train, validation, split_by_task


def _metadata_groups(repo_id: str, revision: str) -> tuple[Any, dict[str, list[int]]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id, revision=revision)
    groups: dict[str, list[int]] = defaultdict(list)
    for row_index, row in _episode_rows(metadata.episodes):
        episode_index = int(row.get("episode_index", row_index))
        groups[_task_name(row.get("tasks"))].append(episode_index)
    return metadata, dict(groups)


def prepare_manifest(args: argparse.Namespace) -> Path:
    metadata, groups = _metadata_groups(args.dataset_repo, args.dataset_revision)
    max_per_task = args.max_episodes_per_task or None
    train, validation, split_by_task = balanced_episode_split(
        groups,
        seed=args.seed,
        validation_per_task=args.validation_per_task,
        max_per_task=max_per_task,
    )

    lengths: list[int] = []
    invalid_length_episodes: list[int] = []
    selected_episodes = set(train + validation)
    for row_index, row in _episode_rows(metadata.episodes):
        episode_index = int(row.get("episode_index", row_index))
        length = int(row.get("length", 0))
        if episode_index in selected_episodes:
            lengths.append(length)
            if length <= 0:
                invalid_length_episodes.append(episode_index)
    if invalid_length_episodes:
        raise ValueError(f"non-positive episode lengths: {invalid_length_episodes}")

    task_counts = {
        task: {
            "available": len(set(groups[task])),
            "train": len(parts["train"]),
            "validation": len(parts["validation"]),
        }
        for task, parts in split_by_task.items()
    }
    manifest = {
        "schema_version": 1,
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.dataset_revision,
        "dataset_license": "Apache-2.0",
        "selection_seed": args.seed,
        "selection_basis": (
            "task-balanced curated LIBERO demonstrations; no PARC evaluation "
            "observation or result used"
        ),
        "split_unit": "episode",
        "train_episodes": train,
        "validation_episodes": validation,
        "task_counts": task_counts,
        "selected_episode_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {args.output}")
    print(f"tasks: {len(task_counts)}")
    print(f"train episodes: {len(train)}")
    print(f"validation episodes: {len(validation)}")
    for task, counts in task_counts.items():
        print(
            f"  {task}: available={counts['available']} "
            f"train={counts['train']} validation={counts['validation']}"
        )
    return args.output


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_repo",
        "dataset_revision",
        "train_episodes",
        "validation_episodes",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"manifest is missing fields: {sorted(missing)}")
    if set(manifest["train_episodes"]) & set(manifest["validation_episodes"]):
        raise ValueError("manifest train/validation episodes overlap")
    return manifest


def build_train_command(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    if args.steps < 1 or args.batch_size < 1 or args.lora_rank < 1:
        raise ValueError("steps, batch_size, and lora_rank must be positive")
    save_freq = min(args.save_freq, args.steps)
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={manifest['dataset_repo']}",
        f"--dataset.revision={manifest['dataset_revision']}",
        f"--dataset.episodes={json.dumps(manifest['train_episodes'])}",
        "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=true",
        "--dataset.image_transforms.max_num_transforms=1",
        "--dataset.image_transforms.tfs="
        + json.dumps(LIGHT_COLOR_TRANSFORMS, separators=(",", ":")),
        f"--policy.path={args.base_model}",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.gradient_checkpointing=true",
        "--policy.compile_model=false",
        f"--policy.optimizer_lr={args.learning_rate}",
        f"--policy.scheduler_decay_lr={args.decay_learning_rate}",
        f"--policy.scheduler_warmup_steps={min(args.warmup_steps, args.steps)}",
        f"--policy.scheduler_decay_steps={args.steps}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        "--log_freq=10",
        f"--save_freq={save_freq}",
        "--save_checkpoint=true",
        "--peft.method_type=LORA",
        f"--peft.r={args.lora_rank}",
        f"--peft.target_modules={ACTION_EXPERT_LORA_TARGETS}",
        "--peft.full_training_modules=[]",
        f"--wandb.enable={str(args.wandb).lower()}",
        f"--wandb.project={args.wandb_project}",
        "--wandb.notes=pi0.5 Action Expert LoRA; no PARC evaluation data used",
    ]


def build_resume_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        "--resume=true",
        f"--config_path={config_path}",
    ]


def _training_environment(use_wandb: bool) -> dict[str, str]:
    environment = os.environ.copy()
    environment["TOKENIZERS_PARALLELISM"] = "false"
    if use_wandb:
        environment.pop("WANDB_DISABLED", None)
        environment["WANDB_MODE"] = "online"
    else:
        environment["WANDB_MODE"] = "disabled"
        environment["WANDB_DISABLED"] = "true"
    return environment


def _run_streamed(command: list[str], environment: dict[str, str]) -> tuple[int, str]:
    """Run a short probe while retaining enough text to identify CUDA OOM."""
    recent: deque[str] = deque(maxlen=300)
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        recent.append(line)
    return process.wait(), "".join(recent)


def _preflight_batch(args: argparse.Namespace, manifest: dict[str, Any], initial_batch: int) -> int:
    """Try a few real optimizer steps and reduce batch only for a confirmed OOM."""
    candidate = initial_batch
    while candidate >= 1:
        probe_dir = args.output_dir.parent / f"preflight_batch_{candidate}"
        shutil.rmtree(probe_dir, ignore_errors=True)
        probe_args = argparse.Namespace(**vars(args))
        probe_args.output_dir = probe_dir
        probe_args.steps = args.preflight_steps
        probe_args.save_freq = args.preflight_steps
        probe_args.batch_size = candidate
        probe_args.wandb = False
        print(
            f"PREFLIGHT_START batch={candidate} steps={args.preflight_steps}",
            flush=True,
        )
        probe_command = build_train_command(probe_args, manifest)
        probe_command[probe_command.index("--save_checkpoint=true")] = (
            "--save_checkpoint=false"
        )
        returncode, recent = _run_streamed(
            probe_command, _training_environment(False)
        )
        shutil.rmtree(probe_dir, ignore_errors=True)
        if returncode == 0:
            print(f"PREFLIGHT_PASS batch={candidate}", flush=True)
            return candidate
        oom = "out of memory" in recent.lower() or "cuda error: memory" in recent.lower()
        if not oom:
            raise RuntimeError(
                f"preflight failed for a reason other than CUDA OOM (exit={returncode})"
            )
        if candidate == 1:
            raise RuntimeError("pi0.5 LoRA is out of memory even with batch 1")
        next_candidate = max(1, candidate // 2)
        print(
            f"PREFLIGHT_OOM batch={candidate}; retrying batch={next_candidate}",
            flush=True,
        )
        candidate = next_candidate
    raise AssertionError("unreachable batch preflight state")


def _patch_resume_config(
    config_path: Path,
    *,
    batch_size: int,
    output_dir: Path,
    num_workers: int,
    steps: int,
    save_freq: int,
) -> None:
    """Allow an A100 checkpoint to resume safely on an L4, or vice versa."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["batch_size"] = batch_size
    config["output_dir"] = str(output_dir)
    config["num_workers"] = num_workers
    config["steps"] = steps
    config["save_freq"] = min(save_freq, steps)
    policy_config = config.get("policy")
    if isinstance(policy_config, dict):
        policy_config["scheduler_decay_steps"] = steps
        policy_config["scheduler_warmup_steps"] = min(
            int(policy_config.get("scheduler_warmup_steps", steps)), steps
        )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def _monitor_backups(
    stop_event: threading.Event,
    backup_lock: threading.Lock,
    local_checkpoints: Path,
    backup_checkpoints: Path,
    interval_seconds: float,
) -> None:
    while not stop_event.wait(interval_seconds):
        try:
            with backup_lock:
                sync_checkpoints(local_checkpoints, backup_checkpoints)
        except Exception as exc:
            # Do not kill a healthy multi-hour training run for a transient
            # Drive issue.  The final sync still fails loudly if it persists.
            print(f"CHECKPOINT_BACKUP_WARNING: {exc}", flush=True)


def train(args: argparse.Namespace) -> None:
    import torch

    manifest = _load_manifest(args.manifest)
    if args.steps < 1 or args.preflight_steps < 1 or args.save_freq < 1:
        raise ValueError("steps, preflight_steps, and save_freq must be positive")
    if args.backup_interval <= 0:
        raise ValueError("backup_interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pi0.5 LoRA training")
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    detected_batch = recommended_batch_size(total_vram_gib)
    initial_batch = args.batch_size or detected_batch
    print(
        f"GPU={torch.cuda.get_device_name(0)} VRAM={total_vram_gib:.1f}GiB "
        f"initial_batch={initial_batch}",
        flush=True,
    )
    selected_batch = _preflight_batch(args, manifest, initial_batch)
    args.batch_size = selected_batch
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    local_checkpoints = args.output_dir / "checkpoints"
    if args.backup_dir is not None:
        restore_checkpoints(args.backup_dir, local_checkpoints)
        args.backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.manifest, args.backup_dir.parent / "data_split.json")

    checkpoints = discover_checkpoints(local_checkpoints)
    if checkpoints:
        step, checkpoint_dir = checkpoints[-1]
        if step >= args.steps:
            print(f"TRAINING_ALREADY_COMPLETE step={step}", flush=True)
            if args.backup_dir is not None:
                sync_checkpoints(local_checkpoints, args.backup_dir)
            return
        config_path = checkpoint_dir / "pretrained_model" / "train_config.json"
        _patch_resume_config(
            config_path,
            batch_size=selected_batch,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            steps=args.steps,
            save_freq=args.save_freq,
        )
        command = build_resume_command(config_path)
        print(f"RESUME_FROM step={step} batch={selected_batch}", flush=True)
    else:
        if args.output_dir.exists():
            # A failure before the first periodic checkpoint leaves no resumable
            # state. Preserve its logs, then start the real run cleanly.
            abandoned = args.output_dir.with_name(
                f"{args.output_dir.name}.incomplete_{int(time.time())}"
            )
            args.output_dir.replace(abandoned)
            print(f"Moved incomplete run to {abandoned}", flush=True)
        command = build_train_command(args, manifest)

    print("Action Expert LoRA training command:")
    print(" ".join(command))
    stop_event = threading.Event()
    backup_lock = threading.Lock()
    monitor = None
    if args.backup_dir is not None:
        monitor = threading.Thread(
            target=_monitor_backups,
            args=(
                stop_event,
                backup_lock,
                local_checkpoints,
                args.backup_dir,
                args.backup_interval,
            ),
            daemon=True,
        )
        monitor.start()
    try:
        subprocess.run(
            command,
            env=_training_environment(args.wandb),
            check=True,
        )
    finally:
        stop_event.set()
        if monitor is not None:
            monitor.join()
        if args.backup_dir is not None:
            with backup_lock:
                sync_checkpoints(local_checkpoints, args.backup_dir)

    final_checkpoints = discover_checkpoints(local_checkpoints)
    if not final_checkpoints or final_checkpoints[-1][0] != args.steps:
        raise RuntimeError(
            f"training did not produce the expected step {args.steps} checkpoint"
        )
    print(
        f"TRAINING_COMPLETE step={args.steps} batch={selected_batch} "
        f"checkpoints={len(final_checkpoints)}",
        flush=True,
    )


def _copy_processors(source_dirs: list[Path], output_dir: Path) -> None:
    patterns = (
        "policy_preprocessor*.json",
        "policy_postprocessor*.json",
        "policy_preprocessor*.safetensors",
        "policy_postprocessor*.safetensors",
    )
    copied: set[str] = set()
    for source_dir in source_dirs:
        for pattern in patterns:
            for source in source_dir.glob(pattern):
                if source.name not in copied:
                    shutil.copy2(source, output_dir / source.name)
                    copied.add(source.name)
    required = {
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    missing = sorted(required - copied)
    if missing:
        raise FileNotFoundError(f"missing policy processor files after merge: {missing}")


def merge_adapter(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} exists; pass --overwrite to replace it"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to merge the 4B pi0.5 model safely")
    config = PreTrainedConfig.from_pretrained(args.base_model, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise TypeError(f"Expected PI05Config, got {type(config).__name__}")
    config.device = "cuda"
    config.dtype = "bfloat16"
    config.gradient_checkpointing = False
    config.compile_model = False
    base_policy = PI05Policy.from_pretrained(
        args.base_model,
        config=config,
        local_files_only=True,
        strict=True,
    )
    peft_policy = PeftModel.from_pretrained(base_policy, args.adapter_dir)
    merged_policy = peft_policy.merge_and_unload(safe_merge=True)
    merged_policy.config.use_peft = False
    # Keep the 4B base on the 24GB L4 while merging.  This avoids holding both a
    # full Python model and its checkpoint tensors in Colab's smaller system RAM.
    merged_policy.save_pretrained(args.output_dir)
    _copy_processors([args.adapter_dir, args.base_model], args.output_dir)
    train_config = args.adapter_dir / "train_config.json"
    if train_config.is_file():
        shutil.copy2(train_config, args.output_dir / "pi05_lora_train_config.json")
    if args.training_manifest is not None:
        shutil.copy2(
            args.training_manifest,
            args.output_dir / "pi05_lora_data_manifest.json",
        )

    weights = args.output_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    manifest = {
        "base_model": str(args.base_model),
        "adapter_checkpoint": str(args.adapter_dir),
        "lora_target_modules": ACTION_EXPERT_LORA_TARGETS,
        "merged_weights_bytes": weights.stat().st_size,
        "torch_version": torch.__version__,
    }
    (args.output_dir / "pi05_lora_merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"merged model: {args.output_dir}")
    print(f"weights GiB: {weights.stat().st_size / 2**30:.2f}")


def validation_loss(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from torch.utils.data import DataLoader, Subset
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    manifest = _load_manifest(args.manifest)
    validation_episodes = [int(value) for value in manifest["validation_episodes"]]
    metadata = LeRobotDatasetMetadata(
        manifest["dataset_repo"], revision=manifest["dataset_revision"]
    )
    config = PreTrainedConfig.from_pretrained(args.model_dir, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise TypeError(f"Expected PI05Config, got {type(config).__name__}")
    config.device = "cuda"
    config.dtype = "bfloat16"
    config.gradient_checkpointing = False
    config.compile_model = False
    policy = PI05Policy.from_pretrained(
        args.model_dir,
        config=config,
        local_files_only=True,
        strict=True,
    ).eval()
    if args.adapter_dir is not None:
        policy = PeftModel.from_pretrained(
            policy,
            args.adapter_dir,
            is_trainable=False,
        ).eval()
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    dataset = LeRobotDataset(
        manifest["dataset_repo"],
        episodes=validation_episodes,
        delta_timestamps=delta_timestamps,
        revision=manifest["dataset_revision"],
        video_backend="pyav",
    )
    sample_count = min(args.max_samples, len(dataset))
    if sample_count < 1:
        raise ValueError("validation dataset is empty")
    if sample_count == 1:
        sample_indices = [0]
    else:
        sample_indices = [
            round(index * (len(dataset) - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # Use the same dataset statistics for base and LoRA validation.  Otherwise
    # a change in saved normalizer statistics could masquerade as a model-loss
    # change and make the comparison misleading.
    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path=str(args.model_dir),
        dataset_stats=metadata.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {
                "stats": metadata.stats,
                "features": {**config.input_features, **config.output_features},
                "norm_map": config.normalization_mapping,
            },
        },
    )

    torch.manual_seed(args.seed)
    losses: list[float] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            processed = preprocessor(batch)
            loss, _ = policy.forward(processed)
            value = float(loss.item())
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite validation loss at batch {batch_index}")
            losses.append(value)
            print(
                f"VALIDATION_PROGRESS batch={batch_index}/{len(loader)} "
                f"loss={value:.6f}",
                flush=True,
            )
    result = {
        "model_dir": str(args.model_dir),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir is not None else None,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "validation_episodes": validation_episodes,
        "validation_samples": sample_count,
        "seed": args.seed,
        "mean_flow_matching_loss": sum(losses) / len(losses),
        "batch_losses": losses,
        "interpretation": (
            "Use only for comparing checkpoints on the same held-out set and seed; "
            "lower loss does not guarantee a higher PARC score."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"mean validation loss: {result['mean_flow_matching_loss']:.6f}")
    print(f"validation result: {args.output}")


def select_checkpoint(args: argparse.Namespace) -> None:
    """Compare periodic adapters on one held-out set and select the lowest loss."""
    if args.candidate_every < 1 or args.max_samples < 1:
        raise ValueError("candidate_every and max_samples must be positive")
    checkpoints = discover_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no complete checkpoints in {args.checkpoints_dir}")
    final_step = checkpoints[-1][0]
    candidates = [
        (step, path)
        for step, path in checkpoints
        if step % args.candidate_every == 0 or step == final_step
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = _sha256_file(args.manifest)

    def cached_validation(path: Path, adapter_dir: Path | None) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            expected_adapter = str(adapter_dir) if adapter_dir is not None else None
            if (
                result.get("model_dir") != str(args.base_model)
                or result.get("adapter_dir") != expected_adapter
                or result.get("manifest_sha256") != manifest_sha256
                or int(result.get("validation_samples", -1)) != args.max_samples
                or int(result.get("seed", -1)) != args.seed
            ):
                return None
            loss = float(result["mean_flow_matching_loss"])
            return result if math.isfinite(loss) and loss > 0.0 else None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    base_output = args.output_dir / "base.json"
    base_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "validate",
        "--manifest",
        str(args.manifest),
        "--model-dir",
        str(args.base_model),
        "--output",
        str(base_output),
        "--max-samples",
        str(args.max_samples),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]
    base_result = cached_validation(base_output, None)
    if base_result is None:
        subprocess.run(base_command, check=True)
        base_result = json.loads(base_output.read_text(encoding="utf-8"))
    else:
        print(f"CHECKPOINT_VALIDATION_CACHE base={base_output}", flush=True)
    base_loss = float(base_result["mean_flow_matching_loss"])

    results: list[dict[str, Any]] = []
    for candidate_index, (step, checkpoint_dir) in enumerate(candidates, 1):
        adapter_dir = checkpoint_dir / "pretrained_model"
        output = args.output_dir / f"step_{step:06d}.json"
        print(
            f"CHECKPOINT_VALIDATION step={step} "
            f"candidate={candidate_index}/{len(candidates)}",
            flush=True,
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "validate",
            "--manifest",
            str(args.manifest),
            "--model-dir",
            str(args.base_model),
            "--adapter-dir",
            str(adapter_dir),
            "--output",
            str(output),
            "--max-samples",
            str(args.max_samples),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
        ]
        result = cached_validation(output, adapter_dir)
        if result is None:
            subprocess.run(command, check=True)
            result = json.loads(output.read_text(encoding="utf-8"))
        else:
            print(
                f"CHECKPOINT_VALIDATION_CACHE step={step} output={output}",
                flush=True,
            )
        loss = float(result["mean_flow_matching_loss"])
        if not math.isfinite(loss):
            raise RuntimeError(f"non-finite checkpoint loss at step {step}")
        results.append(
            {
                "step": step,
                "checkpoint_dir": str(checkpoint_dir),
                "adapter_dir": str(adapter_dir),
                "mean_flow_matching_loss": loss,
            }
        )

    selected = min(results, key=lambda item: item["mean_flow_matching_loss"])
    if not (0.0 < selected["mean_flow_matching_loss"] < base_loss * args.max_loss_ratio):
        raise RuntimeError(
            "no safe LoRA checkpoint was found: "
            f"base={base_loss:.6f}, best={selected['mean_flow_matching_loss']:.6f}"
        )
    summary = {
        "selection_basis": "lowest held-out flow-matching loss; no PARC evaluation result used",
        "base_mean_flow_matching_loss": base_loss,
        "candidate_every": args.candidate_every,
        "validation_samples": args.max_samples,
        "seed": args.seed,
        "candidates": results,
        "selected": selected,
        "interpretation": (
            "This selects a training checkpoint without using PARC evaluation data. "
            "Simulator success and trajectory metrics must still be checked after merge."
        ),
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"SELECTED_CHECKPOINT step={selected['step']} "
        f"loss={selected['mean_flow_matching_loss']:.6f} base={base_loss:.6f}",
        flush=True,
    )
    print(f"selection result: {args.output}")


def _common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create balanced episode split manifest.")
    _common_dataset_args(prepare_parser)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=20260812)
    prepare_parser.add_argument("--validation-per-task", type=int, default=2)
    prepare_parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=0,
        help="0 uses the smallest available task count.",
    )
    prepare_parser.set_defaults(func=prepare_manifest)

    train_parser = subparsers.add_parser("train", help="Run official LeRobot PEFT training.")
    train_parser.add_argument("--manifest", type=Path, required=True)
    train_parser.add_argument("--base-model", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--job-name", default="pi05_action_expert_lora")
    train_parser.add_argument("--steps", type=int, default=3000)
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 chooses A100=4 or L4=2, then reduces only after a confirmed OOM.",
    )
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--lora-rank", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    train_parser.add_argument("--decay-learning-rate", type=float, default=1.0e-5)
    train_parser.add_argument("--warmup-steps", type=int, default=100)
    train_parser.add_argument("--save-freq", type=int, default=250)
    train_parser.add_argument("--preflight-steps", type=int, default=5)
    train_parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Persistent checkpoint directory, normally under mounted Google Drive.",
    )
    train_parser.add_argument("--backup-interval", type=float, default=15.0)
    train_parser.add_argument("--wandb", action="store_true")
    train_parser.add_argument("--wandb-project", default="parc2026-pi05")
    train_parser.set_defaults(func=train)

    merge_parser = subparsers.add_parser("merge", help="Merge a LoRA adapter into ordinary weights.")
    merge_parser.add_argument("--base-model", type=Path, required=True)
    merge_parser.add_argument("--adapter-dir", type=Path, required=True)
    merge_parser.add_argument("--output-dir", type=Path, required=True)
    merge_parser.add_argument("--training-manifest", type=Path)
    merge_parser.add_argument("--overwrite", action="store_true")
    merge_parser.set_defaults(func=merge_adapter)

    validation_parser = subparsers.add_parser(
        "validate", help="Measure held-out flow-matching loss."
    )
    validation_parser.add_argument("--manifest", type=Path, required=True)
    validation_parser.add_argument("--model-dir", type=Path, required=True)
    validation_parser.add_argument("--adapter-dir", type=Path)
    validation_parser.add_argument("--output", type=Path, required=True)
    validation_parser.add_argument("--max-samples", type=int, default=32)
    validation_parser.add_argument("--batch-size", type=int, default=1)
    validation_parser.add_argument("--num-workers", type=int, default=2)
    validation_parser.add_argument("--seed", type=int, default=20260812)
    validation_parser.set_defaults(func=validation_loss)

    select_parser = subparsers.add_parser(
        "select", help="Select a periodic LoRA checkpoint using held-out loss."
    )
    select_parser.add_argument("--manifest", type=Path, required=True)
    select_parser.add_argument("--base-model", type=Path, required=True)
    select_parser.add_argument("--checkpoints-dir", type=Path, required=True)
    select_parser.add_argument("--output-dir", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--candidate-every", type=int, default=500)
    select_parser.add_argument("--max-samples", type=int, default=64)
    select_parser.add_argument("--batch-size", type=int, default=1)
    select_parser.add_argument("--num-workers", type=int, default=2)
    select_parser.add_argument("--seed", type=int, default=20260812)
    select_parser.add_argument("--max-loss-ratio", type=float, default=2.0)
    select_parser.set_defaults(func=select_checkpoint)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
