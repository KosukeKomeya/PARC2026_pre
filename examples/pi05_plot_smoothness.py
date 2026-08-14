#!/usr/bin/env python3
"""Plot pi0.5 trajectory smoothness and replan-boundary diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    return np.asarray(data[key]).item()


def _jerk(positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if len(positions) < 4:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    velocity = np.diff(positions, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    values = np.linalg.norm(np.diff(acceleration, axis=0) / dt, axis=1)
    # Each third difference ends at this executed action/position index.
    return np.arange(3, len(positions), dtype=np.int64), values


def _orientation_change(quaternions: np.ndarray) -> np.ndarray:
    if len(quaternions) < 2:
        return np.empty(0, dtype=np.float64)
    dots = np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _mean_by_phase(
    samples: list[tuple[np.ndarray, np.ndarray]], replan_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.full(replan_steps, np.nan)
    standard_errors = np.full(replan_steps, np.nan)
    counts = np.zeros(replan_steps, dtype=np.int64)
    for phase in range(replan_steps):
        values = np.concatenate(
            [value[index % replan_steps == phase] for index, value in samples]
        )
        values = values[np.isfinite(values)]
        counts[phase] = len(values)
        if len(values):
            means[phase] = float(np.mean(values))
            standard_errors[phase] = float(np.std(values) / math.sqrt(len(values)))
    return means, standard_errors, counts


def _episode_plot(
    path: Path,
    output_dir: Path,
    replan_steps: int,
    dt: float,
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        task = str(_scalar(data, "task_name"))
        episode_id = int(_scalar(data, "episode_id"))
        success = bool(_scalar(data, "success"))
        collided = bool(_scalar(data, "collided"))
        positions = np.asarray(data["ee_positions"], dtype=np.float64)
        orientations = np.asarray(data["ee_orientations"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)

    jerk_index, jerk = _jerk(positions, dt)
    action_jump = (
        np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)
        if len(actions) >= 2
        else np.empty(0)
    )
    action_jump_index = np.arange(1, len(actions), dtype=np.int64)
    rotation_change = _orientation_change(orientations)

    boundary_mask = np.isin(jerk_index % replan_steps, [0, 1])
    boundary_jerk = float(np.mean(jerk[boundary_mask])) if np.any(boundary_mask) else math.nan
    interior_jerk = float(np.mean(jerk[~boundary_mask])) if np.any(~boundary_mask) else math.nan
    boundary_ratio = (
        boundary_jerk / interior_jerk
        if np.isfinite(boundary_jerk) and interior_jerk > 0
        else math.nan
    )

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    step = np.arange(len(positions))
    for dimension, label in enumerate(("x", "y", "z")):
        axes[0].plot(step, positions[:, dimension], label=label, linewidth=1.2)
    axes[0].set_ylabel("EEF position (m)")
    axes[0].legend(loc="upper right", ncols=3)

    if len(actions):
        axes[1].plot(
            np.arange(len(actions)),
            np.linalg.norm(actions[:, :3], axis=1),
            label="translation action",
        )
        axes[1].plot(
            np.arange(len(actions)),
            np.linalg.norm(actions[:, 3:6], axis=1),
            label="rotation action",
        )
        axes[1].legend(loc="upper right")
    axes[1].set_ylabel("Action norm")

    axes[2].plot(action_jump_index, action_jump, color="tab:orange")
    axes[2].set_ylabel("Action jump\n6-D norm")

    axes[3].plot(jerk_index, jerk, color="tab:red", linewidth=1.0)
    axes[3].set_ylabel("EEF jerk (m/s³)")
    axes[3].set_xlabel("Executed step")

    for axis in axes:
        for boundary in range(replan_steps, len(positions), replan_steps):
            axis.axvline(boundary, color="black", alpha=0.12, linewidth=0.8)
        axis.grid(alpha=0.2)

    status = f"success={success}, collision={collided}"
    fig.suptitle(
        f"{task} / episode {episode_id:03d} / {status}\n"
        f"boundary jerk ratio={boundary_ratio:.3f} (phase 0–1 vs interior)"
    )
    fig.tight_layout()
    output = output_dir / f"{path.stem}__smoothness.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)

    return {
        "path": path,
        "task": task,
        "episode_id": episode_id,
        "success": success,
        "collided": collided,
        "steps": len(positions),
        "rms_jerk": float(math.sqrt(np.mean(jerk**2))) if len(jerk) else math.nan,
        "mean_action_jump": float(np.mean(action_jump)) if len(action_jump) else math.nan,
        "orientation_path": float(np.sum(rotation_change)),
        "boundary_jerk": boundary_jerk,
        "interior_jerk": interior_jerk,
        "boundary_ratio": boundary_ratio,
        "jerk_samples": (jerk_index, jerk),
        "jump_samples": (action_jump_index, action_jump),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.05)
    args = parser.parse_args()
    if args.replan_steps < 1 or args.dt <= 0:
        parser.error("--replan-steps and --dt must be positive")

    paths = sorted(args.trajectory_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No trajectory NPZ files in {args.trajectory_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _episode_plot(path, args.output_dir, args.replan_steps, args.dt)
        for path in paths
    ]
    jerk_samples = [row.pop("jerk_samples") for row in rows]
    jump_samples = [row.pop("jump_samples") for row in rows]
    jerk_mean, jerk_se, jerk_count = _mean_by_phase(jerk_samples, args.replan_steps)
    jump_mean, jump_se, jump_count = _mean_by_phase(jump_samples, args.replan_steps)

    phases = np.arange(args.replan_steps)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(phases, jerk_mean, color="tab:red", alpha=0.75)
    axes[0].errorbar(phases, jerk_mean, yerr=jerk_se, fmt="none", color="black", capsize=3)
    axes[0].set_ylabel("Mean EEF jerk (m/s³)")
    axes[1].bar(phases, jump_mean, color="tab:orange", alpha=0.75)
    axes[1].errorbar(phases, jump_mean, yerr=jump_se, fmt="none", color="black", capsize=3)
    axes[1].set_ylabel("Mean 6-D action jump")
    axes[1].set_xlabel("Step position inside each replan chunk")
    for axis in axes:
        axis.axvspan(-0.5, 1.5, color="tab:blue", alpha=0.08, label="boundary window")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="upper right")
    fig.suptitle(f"Replan-boundary smoothness profile ({len(rows)} episodes)")
    fig.tight_layout()
    summary_plot = args.output_dir / "smoothness_by_replan_phase.png"
    fig.savefig(summary_plot, dpi=170)
    plt.close(fig)

    with (args.output_dir / "episode_smoothness.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "replan_phase_smoothness.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["phase", "mean_jerk", "jerk_se", "jerk_count", "mean_action_jump", "jump_se", "jump_count"]
        )
        for phase in phases:
            writer.writerow(
                [phase, jerk_mean[phase], jerk_se[phase], jerk_count[phase], jump_mean[phase], jump_se[phase], jump_count[phase]]
            )

    ratios = np.asarray([row["boundary_ratio"] for row in rows], dtype=np.float64)
    print("SMOOTHNESS_PLOTS", args.output_dir)
    print("episodes:", len(rows))
    print("mean boundary/interior jerk ratio:", float(np.nanmean(ratios)))
    print("ratio > 1 means jerk is concentrated near replan boundaries")
    print("summary:", summary_plot)


if __name__ == "__main__":
    main()
