#!/usr/bin/env python3
"""Rollout-select saved pi0.5 q/k/v/o LoRA checkpoints without retraining."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pi05_finalize_qkvo_colab as finalizer


LOWER_IS_BETTER = (
    "avg_steps_to_success",
    "cartesian_path_length",
    "orientation_path_length",
    "rms_cartesian_jerk",
)
HIGHER_IS_BETTER = ("sparc",)


def stream(command: list[str], cwd: Path) -> None:
    print("command:", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"command failed with exit={returncode}: {command}")


def complete_adapter(checkpoint_root: Path, step: int) -> Path:
    checkpoint = checkpoint_root / f"{step:06d}"
    adapter = checkpoint / "pretrained_model"
    required = (
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
        checkpoint / "training_state/training_step.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint {step} is incomplete under {checkpoint_root}: {missing}"
        )
    state = json.loads(required[-1].read_text(encoding="utf-8"))
    recorded_step = int(state.get("step", state.get("training_step", -1)))
    if recorded_step != step:
        raise RuntimeError(
            f"checkpoint metadata mismatch: directory={step}, metadata={recorded_step}"
        )
    return adapter


def merge_checkpoint(
    *,
    repo_root: Path,
    policy_python: Path,
    base_model: Path,
    adapter: Path,
    output_dir: Path,
    training_manifest: Path | None,
) -> None:
    command = [
        str(policy_python),
        "-u",
        "examples/pi05_action_expert_lora.py",
        "merge",
        "--base-model",
        str(base_model),
        "--adapter-dir",
        str(adapter),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]
    if training_manifest is not None and training_manifest.is_file():
        command.extend(["--training-manifest", str(training_manifest)])
    stream(command, repo_root)
    finalizer.ensure_model(output_dir)


def mean_metrics(result: dict[str, Any]) -> dict[str, float]:
    track = result["tracks"][0]
    tasks = track["tasks"]
    metric_names = {
        name for task in tasks for name in task.get("metrics", {})
    }
    means: dict[str, float] = {
        "success_rate": sum(float(task["success_rate"]) for task in tasks)
        / len(tasks)
    }
    for name in metric_names:
        values = [
            float(task["metrics"][name])
            for task in tasks
            if name in task.get("metrics", {})
        ]
        if values:
            means[name] = sum(values) / len(values)
    return means


def comparison(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    epsilon = 1e-9
    success_delta = candidate["success_rate"] - baseline["success_rate"]
    collision_delta = candidate.get("collision_rate", math.inf) - baseline.get(
        "collision_rate", math.inf
    )
    wins: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    for name in LOWER_IS_BETTER:
        left = candidate.get(name, math.inf)
        right = baseline.get(name, math.inf)
        if left < right - epsilon:
            wins.append(name)
        elif left > right + epsilon:
            losses.append(name)
        else:
            ties.append(name)
    for name in HIGHER_IS_BETTER:
        left = candidate.get(name, -math.inf)
        right = baseline.get(name, -math.inf)
        if left > right + epsilon:
            wins.append(name)
        elif left < right - epsilon:
            losses.append(name)
        else:
            ties.append(name)
    return {
        "success_delta": success_delta,
        "collision_delta": collision_delta,
        "eligible": success_delta >= -epsilon and collision_delta <= epsilon,
        "secondary_wins": wins,
        "secondary_losses": losses,
        "secondary_ties": ties,
        "secondary_net_wins": len(wins) - len(losses),
    }


def safe_remove_model(path: Path, sweep_root: Path) -> None:
    resolved = path.resolve()
    parent = sweep_root.resolve()
    if resolved.parent != parent or not resolved.name.startswith("merged_"):
        raise RuntimeError(f"refusing to remove unexpected model path: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_table(records: list[dict[str, Any]], baseline_step: int) -> None:
    print("\ncheckpoint rollout sweep", flush=True)
    print(
        "step | success | collision | steps | EE path | rotation | "
        "RMS jerk | SPARC | vs baseline",
        flush=True,
    )
    print("---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---", flush=True)
    for record in records:
        metrics = record["mean_metrics"]
        comp = record.get("comparison_to_baseline")
        if record["step"] == baseline_step:
            verdict = "baseline"
        elif comp is None:
            verdict = "not compared"
        else:
            verdict = (
                f"eligible={comp['eligible']}, "
                f"secondary={comp['secondary_net_wins']:+d}"
            )
        print(
            f"{record['step']} | {metrics['success_rate']:.1%} | "
            f"{metrics.get('collision_rate', float('nan')):.1%} | "
            f"{metrics.get('avg_steps_to_success', float('nan')):.1f} | "
            f"{metrics.get('cartesian_path_length', float('nan')):.3f} | "
            f"{metrics.get('orientation_path_length', float('nan')):.3f} | "
            f"{metrics.get('rms_cartesian_jerk', float('nan')):.3f} | "
            f"{metrics.get('sparc', float('nan')):.3f} | {verdict}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("/content/pi05_runtime"))
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-drive-root", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[1500, 2000, 2500, 3000])
    parser.add_argument("--baseline-step", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-seed", type=int, default=20260814)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output_drive_root = args.output_drive_root.resolve()
    base_model = repo_root / "submission_template/model_weights/pi05_libero_finetuned_v044"
    finalizer.ensure_model(base_model)
    if (base_model / "pi05_lora_merge_manifest.json").exists():
        raise RuntimeError(
            "canonical base model is already LoRA-merged; use a fresh runtime/setup"
        )
    if args.baseline_step not in args.steps:
        raise ValueError("baseline step must be included in --steps")
    if args.episodes < 1:
        raise ValueError("episodes must be positive")

    sweep_root = Path("/content/pi05_qkvo_checkpoint_sweep")
    sweep_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    run_config = {
        "steps": args.steps,
        "baseline_step": args.baseline_step,
        "episodes_per_task": args.episodes,
        "max_steps": args.max_steps,
        "environment_seed": args.seed,
        "policy_seed": args.policy_seed,
        "replan_steps": args.replan_steps,
        "inference_steps": args.inference_steps,
        "temporal_ensemble": False,
    }

    for candidate_index, step in enumerate(args.steps, 1):
        print(
            f"\nCANDIDATE {candidate_index}/{len(args.steps)} step={step}",
            flush=True,
        )
        drive_candidate_dir = output_drive_root / "checkpoint_sweep" / f"{step:06d}"
        record_path = drive_candidate_dir / "candidate_record.json"
        if record_path.is_file() and not args.force:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("run_config") == run_config:
                print("RESUME completed candidate:", record_path, flush=True)
                records.append(record)
                continue
            raise RuntimeError(
                f"existing result uses different settings: {record_path}; pass --force"
            )

        adapter = complete_adapter(checkpoint_root, step)
        merged_model = sweep_root / f"merged_{step:06d}"
        merge_checkpoint(
            repo_root=repo_root,
            policy_python=args.policy_python,
            base_model=base_model,
            adapter=adapter,
            output_dir=merged_model,
            training_manifest=args.training_manifest,
        )
        variant = (
            f"qkvo_step_{step:06d}_replan_{args.replan_steps}_"
            f"infer_{args.inference_steps}_ensemble_0"
        )
        results_dir = repo_root / "results" / f"pi05_public_eval_{variant}"
        try:
            result_path, server_log, evaluation_log, result = finalizer.evaluate(
                repo_root=repo_root,
                policy_python=args.policy_python,
                eval_python=args.eval_python,
                runtime_root=args.runtime_root,
                model_dir=merged_model,
                results_dir=results_dir,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                replan_steps=args.replan_steps,
                inference_steps=args.inference_steps,
                temporal_ensemble=False,
                record_video=args.record_video,
                deterministic_policy_seed=args.policy_seed,
            )
            finalizer.print_metrics(result)
            artifacts = [result_path, server_log, evaluation_log]
            artifacts.extend(sorted((results_dir / "videos").glob("*.mp4")))
            for artifact in artifacts:
                finalizer.copy_file_with_progress(
                    artifact, drive_candidate_dir / artifact.name
                )
            record = {
                "step": step,
                "adapter_dir": str(adapter),
                "run_config": run_config,
                "mean_metrics": mean_metrics(result),
                "result": result,
                "git_head": finalizer.git_head(repo_root),
            }
            write_json(record_path, record)
            records.append(record)
        finally:
            safe_remove_model(merged_model, sweep_root)

        write_json(
            output_drive_root / "checkpoint_sweep/checkpoint_sweep_partial.json",
            {"run_config": run_config, "candidates": records},
        )

    records.sort(key=lambda item: args.steps.index(int(item["step"])))
    baseline = next(
        record for record in records if int(record["step"]) == args.baseline_step
    )
    baseline_metrics = baseline["mean_metrics"]
    for record in records:
        if int(record["step"]) != args.baseline_step:
            record["comparison_to_baseline"] = comparison(
                record["mean_metrics"], baseline_metrics
            )

    eligible = [
        record
        for record in records
        if int(record["step"]) == args.baseline_step
        or record["comparison_to_baseline"]["eligible"]
    ]

    def rank_key(record: dict[str, Any]) -> tuple[float, float, int, int]:
        metrics = record["mean_metrics"]
        if int(record["step"]) == args.baseline_step:
            net_wins = 0
        else:
            net_wins = int(record["comparison_to_baseline"]["secondary_net_wins"])
        return (
            -float(metrics["success_rate"]),
            float(metrics.get("collision_rate", math.inf)),
            -net_wins,
            0 if int(record["step"]) == args.baseline_step else 1,
        )

    recommended = min(eligible, key=rank_key)
    summary = {
        "selection_policy": (
            "success rate descending; collision rate ascending; then number of "
            "secondary metric wins over step-2000 baseline. Hidden competition "
            "normalization is not approximated. Re-evaluate the top candidates "
            "with at least 5 episodes/task before building a submission."
        ),
        "run_config": run_config,
        "recommended_step_for_confirmation": int(recommended["step"]),
        "candidates": records,
    }
    summary_path = output_drive_root / "checkpoint_sweep/checkpoint_sweep_summary.json"
    write_json(summary_path, summary)
    print_table(records, args.baseline_step)
    print("RECOMMENDED_FOR_5_EPISODE_CONFIRMATION:", recommended["step"], flush=True)
    print("SWEEP_SUMMARY:", summary_path, flush=True)
    print("No submission ZIP was changed or created by this sweep.", flush=True)


if __name__ == "__main__":
    main()
