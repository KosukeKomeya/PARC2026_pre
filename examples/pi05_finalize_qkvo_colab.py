#!/usr/bin/env python3
"""Evaluate a merged pi0.5 q/k/v/o model and preserve a verified PARC ZIP.

This script is the reproducible final step used by the q/k/v/o Colab notebook.
It intentionally keeps training and evaluation outside the submitted ZIP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from typing import Any
import urllib.request
import zipfile


REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def stream(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("command:", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
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


def replace_single(source: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"expected one policy setting matching {pattern!r}, got {count}")
    return updated


def configure_policy(
    policy_path: Path,
    *,
    replan_steps: int,
    inference_steps: int,
    temporal_ensemble: bool,
) -> None:
    source = policy_path.read_text(encoding="utf-8")
    source = replace_single(
        source,
        r"^    REPLAN_STEPS = \d+$",
        f"    REPLAN_STEPS = {replan_steps}",
    )
    source = replace_single(
        source,
        r"^    DEFAULT_INFERENCE_STEPS = \d+$",
        f"    DEFAULT_INFERENCE_STEPS = {inference_steps}",
    )
    source = replace_single(
        source,
        r"^    DEFAULT_TEMPORAL_ENSEMBLE = (?:True|False)$",
        f"    DEFAULT_TEMPORAL_ENSEMBLE = {temporal_ensemble}",
    )
    policy_path.write_text(source, encoding="utf-8")
    print(
        "FINAL POLICY:",
        f"replan={replan_steps}, inference={inference_steps}, "
        f"ensemble={temporal_ensemble}",
        flush=True,
    )


def ensure_model(model_source: Path) -> None:
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"merged model incomplete ({model_source}): {missing}")


def build_submission(
    repo_root: Path,
    policy_python: Path,
    model_source: Path,
) -> Path:
    stream(
        [
            str(policy_python),
            "-u",
            "examples/pi05_parc_colab_setup.py",
            "--reuse-runtime",
            "--skip-download",
            "--build-submission",
            "--model-source",
            str(model_source),
        ],
        cwd=repo_root,
    )
    submission = repo_root / "pi05_submission.zip"
    if not submission.is_file():
        raise FileNotFoundError(submission)
    return submission


def verify_submission(
    submission: Path,
    *,
    replan_steps: int,
    inference_steps: int,
    temporal_ensemble: bool,
) -> None:
    model_root = "model_weights/pi05_libero_finetuned_v044/"
    required = {
        "policy_server.py",
        "requirements.txt",
        *(model_root + name for name in REQUIRED_MODEL_FILES),
    }
    with zipfile.ZipFile(submission) as archive:
        names = set(archive.namelist())
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"submission ZIP missing: {missing}")
        if "submission_template/policy_server.py" in names:
            raise RuntimeError("submission ZIP has an extra submission_template parent")
        policy = archive.read("policy_server.py").decode("utf-8")
        requirements = archive.read("requirements.txt").decode("utf-8").lower()
    expected = (
        f"REPLAN_STEPS = {replan_steps}",
        f"DEFAULT_INFERENCE_STEPS = {inference_steps}",
        f"DEFAULT_TEMPORAL_ENSEMBLE = {temporal_ensemble}",
    )
    for setting in expected:
        if setting not in policy:
            raise RuntimeError(f"submission policy does not contain {setting!r}")
    if "evdev" in requirements:
        raise RuntimeError("evdev must not be installed in the scoring image")
    print("FINAL_SUBMISSION_STRUCTURE_VERIFIED", flush=True)


def copy_file_with_progress(
    source: Path,
    destination: Path,
    *,
    chunk_mib: int = 16,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    total = source.stat().st_size
    copied = 0
    next_report = 512 * 2**20
    with source.open("rb") as reader, temporary.open("wb") as writer:
        while block := reader.read(chunk_mib * 2**20):
            writer.write(block)
            copied += len(block)
            if copied >= next_report or copied == total:
                print(
                    f"DRIVE_COPY {source.name}: "
                    f"{copied / 2**30:.2f}/{total / 2**30:.2f} GiB",
                    flush=True,
                )
                next_report += 512 * 2**20
    temporary.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 2**20):
            digest.update(block)
    return digest.hexdigest()


def public_tasks(repo_root: Path) -> list[str]:
    tasks_path = repo_root / "compe/t1/T1_TASKS.csv"
    with tasks_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != 4 or len(set(task_ids)) != 4:
        raise RuntimeError(f"expected four unique public tasks, got {task_ids}")
    return task_ids


def evaluate(
    *,
    repo_root: Path,
    policy_python: Path,
    eval_python: Path,
    runtime_root: Path,
    model_dir: Path | None,
    results_dir: Path,
    episodes: int,
    max_steps: int,
    seed: int,
    replan_steps: int,
    inference_steps: int,
    temporal_ensemble: bool,
    record_video: bool,
    deterministic_policy_seed: int | None = None,
    save_trajectories: bool = False,
    rtc_enabled: bool = False,
    rtc_execution_horizon: int = 10,
    rtc_max_guidance_weight: float = 5.0,
    rtc_schedule: str = "EXP",
    rtc_inference_delay: int = 0,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    if temporal_ensemble and rtc_enabled:
        raise ValueError("temporal ensembling and RTC are mutually exclusive")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        server_port = probe.getsockname()[1]
    server_url = f"http://127.0.0.1:{server_port}"
    results_dir.mkdir(parents=True, exist_ok=True)
    video_dir = results_dir / "videos"
    if record_video:
        shutil.rmtree(video_dir, ignore_errors=True)
    server_log_path = results_dir / "policy_server.log"
    evaluation_log_path = results_dir / "evaluation.log"
    result_path = results_dir / f"server_{server_port}.json"

    server_env = os.environ.copy()
    active_model_dir = model_dir or (
        repo_root / "submission_template/model_weights/pi05_libero_finetuned_v044"
    )
    server_env.update(
        {
            "PI05_MODEL_DIR": str(active_model_dir),
            "PI05_TOKENIZER_DIR": str(
                repo_root / "submission_template/model_weights/paligemma-3b-pt-224"
            ),
            "PI05_DEVICE": "cuda",
            "PI05_REPLAN_STEPS": str(replan_steps),
            "PI05_INFERENCE_STEPS": str(inference_steps),
            "PI05_TEMPORAL_ENSEMBLE": "1" if temporal_ensemble else "0",
            "PI05_RTC_ENABLED": "1" if rtc_enabled else "0",
            "PI05_RTC_EXECUTION_HORIZON": str(rtc_execution_horizon),
            "PI05_RTC_MAX_GUIDANCE_WEIGHT": str(rtc_max_guidance_weight),
            "PI05_RTC_SCHEDULE": rtc_schedule,
            "PI05_RTC_INFERENCE_DELAY": str(rtc_inference_delay),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if deterministic_policy_seed is not None:
        server_env["PI05_DETERMINISTIC_EPISODES"] = "1"
        server_env["PI05_POLICY_SEED"] = str(deterministic_policy_seed)
    server_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(runtime_root / "lerobot_v044/src"),
            str(runtime_root / "transformers_lerobot_openpi/src"),
            server_env.get("PYTHONPATH", ""),
        ]
    )

    eval_env = os.environ.copy()
    eval_env.update(
        {
            "MUJOCO_GL": "egl",
            "MPLBACKEND": "Agg",
            "LIBERO_ROOT": str(repo_root / "LIBERO-plus"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    eval_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(repo_root / "LIBERO-plus"),
            str(repo_root),
            str(repo_root / "compe"),
        ]
    )

    eval_command = [
        str(eval_python),
        "-m",
        "pipeline",
        "--server-url",
        server_url,
        "--track",
        "track1",
        "--n-episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--timeout",
        "10",
        "--output-dir",
        str(results_dir),
    ]
    if record_video:
        eval_command.append("--record-video")
    if save_trajectories:
        eval_command.append("--save-trajectories")
    eval_command.extend(["--videos-per-task", "1", "--video-fps", "20"])
    eval_command.extend(["--tasks", *public_tasks(repo_root)])

    evaluation_process: subprocess.Popen[str] | None = None
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server_process = subprocess.Popen(
            [str(policy_python), "policy_server.py", "--port", str(server_port)],
            cwd=repo_root / "submission_template",
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if server_process.poll() is not None:
                    raise RuntimeError(f"policy server exited early; see {server_log_path}")
                try:
                    with urllib.request.urlopen(f"{server_url}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError(f"policy server did not start; see {server_log_path}")

            print("pi0.5 policy server ready. Starting four-task evaluation.", flush=True)
            evaluation_process = subprocess.Popen(
                eval_command,
                cwd=repo_root,
                env=eval_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            recent: list[str] = []
            assert evaluation_process.stdout is not None
            with evaluation_log_path.open("w", encoding="utf-8") as evaluation_log:
                for line in evaluation_process.stdout:
                    print(line, end="", flush=True)
                    evaluation_log.write(line)
                    evaluation_log.flush()
                    recent.append(line.rstrip())
                    recent = recent[-120:]
            returncode = evaluation_process.wait()
            if returncode:
                raise RuntimeError(
                    f"evaluation failed with exit={returncode}\n" + "\n".join(recent)
                )
        finally:
            if evaluation_process is not None and evaluation_process.poll() is None:
                evaluation_process.terminate()
                try:
                    evaluation_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    evaluation_process.kill()
                    evaluation_process.wait(timeout=5)
            server_process.terminate()
            try:
                server_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=5)

    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    track = result["tracks"][0]
    if track.get("overall_metrics", {}).get("error"):
        raise RuntimeError(f"track evaluation failed; see {result_path}")
    if len(track["tasks"]) != 4:
        raise RuntimeError(f"expected four task results, got {len(track['tasks'])}")
    return result_path, server_log_path, evaluation_log_path, result


def print_metrics(result: dict[str, Any]) -> None:
    track = result["tracks"][0]
    print("task | success | steps | EE path | rotation | RMS jerk | SPARC | collision | sec")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for task in track["tasks"]:
        metrics = task["metrics"]
        print(
            f"{task['task_name']} | {task['success_rate']:.1%} | "
            f"{metrics.get('avg_steps_to_success', float('nan')):.1f} | "
            f"{metrics.get('cartesian_path_length', float('nan')):.3f} | "
            f"{metrics.get('orientation_path_length', float('nan')):.3f} | "
            f"{metrics.get('rms_cartesian_jerk', float('nan')):.3f} | "
            f"{metrics.get('sparc', float('nan')):.3f} | "
            f"{metrics.get('collision_rate', float('nan')):.1%} | "
            f"{metrics.get('avg_episode_time_sec', float('nan')):.1f}"
        )
    print(f"overall success: {track['overall_score']:.1%}")


def read_json_if_present(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("/content/pi05_runtime"))
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    model_source = args.model_source.resolve()
    ensure_model(model_source)
    configure_policy(
        repo_root / "submission_template/policy_server.py",
        replan_steps=args.replan_steps,
        inference_steps=args.inference_steps,
        temporal_ensemble=args.temporal_ensemble,
    )
    submission = build_submission(repo_root, args.policy_python, model_source)
    verify_submission(
        submission,
        replan_steps=args.replan_steps,
        inference_steps=args.inference_steps,
        temporal_ensemble=args.temporal_ensemble,
    )

    variant = (
        f"qkvo_replan_{args.replan_steps}_infer_{args.inference_steps}_"
        f"ensemble_{int(args.temporal_ensemble)}"
    )
    results_dir = repo_root / "results" / f"pi05_public_eval_{variant}"
    result_path, server_log, evaluation_log, result = evaluate(
        repo_root=repo_root,
        policy_python=args.policy_python,
        eval_python=args.eval_python,
        runtime_root=args.runtime_root,
        model_dir=None,
        results_dir=results_dir,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        replan_steps=args.replan_steps,
        inference_steps=args.inference_steps,
        temporal_ensemble=args.temporal_ensemble,
        record_video=args.record_video,
        save_trajectories=args.save_trajectories,
    )
    print_metrics(result)

    drive_eval_dir = args.drive_root / "evaluation" / variant
    artifacts = [result_path, server_log, evaluation_log]
    artifacts.extend(sorted((results_dir / "videos").glob("*.mp4")))
    artifacts.extend(sorted((results_dir / "trajectories").glob("*.npz")))
    for artifact in artifacts:
        copy_file_with_progress(artifact, drive_eval_dir / artifact.name)

    # Promote the ZIP to FINAL only after the same activated model completes eval.
    final_name = (
        f"pi05_submission_FINAL_qkvo_r{args.replan_steps}_"
        f"i{args.inference_steps}_{'on' if args.temporal_ensemble else 'off'}.zip"
    )
    drive_submission = args.drive_root / "final" / final_name
    copy_file_with_progress(submission, drive_submission)
    local_hash = sha256(submission)
    drive_hash = sha256(drive_submission)
    if submission.stat().st_size != drive_submission.stat().st_size:
        raise RuntimeError("Drive ZIP size does not match local ZIP")
    if local_hash != drive_hash:
        raise RuntimeError("Drive ZIP SHA256 does not match local ZIP")
    print("DRIVE_SAVE_VERIFIED", drive_submission, drive_hash, flush=True)

    manifest = {
        "schema_version": 1,
        "git_head": git_head(repo_root),
        "model_source": str(model_source),
        "model_merge_manifest": read_json_if_present(
            model_source / "pi05_lora_merge_manifest.json"
        ),
        "checkpoint_selection": read_json_if_present(args.selection),
        "data_split_manifest": read_json_if_present(args.training_manifest),
        "training": {
            "steps": args.train_steps,
            "batch_size": args.batch_size,
            "lora_rank": args.lora_rank,
            "lora_target_profile": "qkvo",
        },
        "evaluation": {
            "track": "track1",
            "episodes_per_task": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "inference_steps": args.inference_steps,
            "temporal_ensemble": args.temporal_ensemble,
            "result": result,
        },
        "submission": {
            "local_path": str(submission),
            "drive_path": str(drive_submission),
            "size_bytes": submission.stat().st_size,
            "sha256": local_hash,
        },
    }
    manifest_path = args.drive_root / "final/final_reproducibility_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("REPRODUCIBILITY_MANIFEST:", manifest_path, flush=True)


if __name__ == "__main__":
    main()
