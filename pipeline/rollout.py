import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import EvalConfig, PerturbationConfig
from .environment import EnvironmentManager, TaskInfo
from .total_score import load_scoring_config

logger = logging.getLogger(__name__)


class PolicyInterface(Protocol):

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ...

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        ...


@dataclass
class EpisodeResult:
    task_name: str
    episode_id: int
    success: bool
    total_steps: int
    elapsed_time_sec: float

    joint_positions: list[np.ndarray] = field(default_factory=list)
    ee_positions: list[np.ndarray] = field(default_factory=list)
    ee_orientations: list[np.ndarray] = field(default_factory=list)
    gripper_qpos: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


    collided: bool = False
    video_path: str | None = None

    @property
    def trajectory(self) -> list[np.ndarray]:
        return self.joint_positions


@dataclass
class TaskResult:
    task_info: TaskInfo
    episodes: list[EpisodeResult]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    @property
    def avg_steps(self) -> float:
        successful = [e for e in self.episodes if e.success]
        if not successful:
            return 0.0
        return sum(e.total_steps for e in successful) / len(successful)

    @property
    def avg_time(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.elapsed_time_sec for e in self.episodes) / len(self.episodes)


class RolloutExecutor:

    def __init__(
        self,
        env_manager: EnvironmentManager,
        eval_config: EvalConfig,
        scoring_config: dict | None = None,
    ):
        self.env_manager = env_manager
        self.config = eval_config


        self.scoring_config = scoring_config or load_scoring_config()

    def evaluate_task(
        self,
        policy: PolicyInterface,
        task_info: TaskInfo,
        perturbation: PerturbationConfig,
        task_index: int = 1,
        task_total: int = 1,
    ) -> TaskResult:
        logger.info(
            "タスク評価開始: %s (%d エピソード)",
            task_info.name, self.config.n_eval_episodes,
        )

        env = self.env_manager.create_env(task_info)


        init_states = self.env_manager.get_perturbed_init_states(
            task_info, perturbation, self.config.n_eval_episodes
        )


        collision_enabled = bool(self.scoring_config.get("collision", {}).get("enabled", True))
        obj_of_interest = (
            self.env_manager.get_obj_of_interest(task_info) if collision_enabled else set()
        )

        episodes: list[EpisodeResult] = []
        failure_videos_saved = 0

        try:
            for ep_id in range(self.config.n_eval_episodes):
                logger.info(
                    "EVAL_PROGRESS task=%d/%d episode=%d/%d status=start",
                    task_index,
                    task_total,
                    ep_id + 1,
                    self.config.n_eval_episodes,
                )
                result = self._run_episode(
                    env=env,
                    policy=policy,
                    task_info=task_info,
                    init_state=init_states[ep_id],
                    episode_id=ep_id,
                    perturbation=perturbation,
                    obj_of_interest=obj_of_interest,
                    task_index=task_index,
                    task_total=task_total,
                    record_video=(
                        self.config.record_video
                        and failure_videos_saved < self.config.videos_per_task
                    ),
                )
                episodes.append(result)
                if result.video_path is not None:
                    failure_videos_saved += 1

                logger.info(
                    "EVAL_PROGRESS task=%d/%d episode=%d/%d status=done "
                    "success=%s collision=%s steps=%d elapsed=%.1fs",
                    task_index,
                    task_total,
                    ep_id + 1,
                    self.config.n_eval_episodes,
                    result.success,
                    result.collided,
                    result.total_steps,
                    result.elapsed_time_sec,
                )

                if result.success:
                    logger.debug(
                        "  Episode %d: 成功 (%d steps)", ep_id, result.total_steps
                    )
                else:
                    logger.debug(
                        "  Episode %d: 失敗 (%d steps)", ep_id, result.total_steps
                    )
        finally:
            env.close()

        task_result = TaskResult(task_info=task_info, episodes=episodes)
        logger.info(
            "タスク評価完了: %s — 成功率 %.1f%% (平均 %.1f steps)",
            task_info.name, task_result.success_rate * 100, task_result.avg_steps,
        )
        return task_result

    def _run_episode(
        self,
        env: Any,
        policy: PolicyInterface,
        task_info: TaskInfo,
        init_state: np.ndarray,
        episode_id: int,
        perturbation: PerturbationConfig,
        obj_of_interest: set[str],
        task_index: int = 1,
        task_total: int = 1,
        record_video: bool = False,
    ) -> EpisodeResult:
        start_time = time.time()
        joint_positions: list[np.ndarray] = []
        ee_positions: list[np.ndarray] = []
        ee_orientations: list[np.ndarray] = []
        gripper_qpos_log: list[np.ndarray] = []
        actions_log: list[np.ndarray] = []
        rewards_log: list[float] = []
        video_frames: list[np.ndarray] = []

        cc = self.scoring_config.get("collision", {})
        collision_enabled = bool(cc.get("enabled", True))
        collision_threshold = float(cc.get("threshold_m", 0.001))


        env.reset()
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()


        action_dim = env.robots[0].action_dim
        dummy_action = np.zeros(action_dim)
        for _ in range(10):
            obs, _, _, _ = env.step(dummy_action)


        object_init_pos: dict[str, np.ndarray] = {}
        if collision_enabled:
            object_init_pos = {
                k[:-4]: np.asarray(obs[k]).copy()
                for k in obs
                if k.endswith("_pos")
                and not k.startswith("robot0")
                and not k.endswith("_to_robot0_eef_pos")
                and k[:-4] not in obj_of_interest
            }
        object_max_disp: dict[str, float] = {}


        episode_seed = self.config.seed + episode_id
        policy.reset(instruction=task_info.language, seed=episode_seed)
        done = False
        total_steps = 0

        for step in range(self.config.max_steps_per_episode):

            if record_video:
                frame = self._make_video_frame(obs)
                if frame is not None:
                    video_frames.append(frame)

            obs_for_policy = self.env_manager.apply_observation_noise(
                obs, perturbation
            )


            action = policy.get_action(obs_for_policy)


            action = self.env_manager.apply_action_noise(action, perturbation)


            obs, reward, done, info = env.step(action)


            joint_positions.append(obs.get("robot0_joint_pos", np.zeros(7)).copy())
            ee_positions.append(obs.get("robot0_eef_pos", np.zeros(3)).copy())
            ee_orientations.append(obs.get("robot0_eef_quat", np.array([1, 0, 0, 0], dtype=np.float64)).copy())
            gripper_qpos_log.append(obs.get("robot0_gripper_qpos", np.zeros(2)).copy())
            actions_log.append(action.copy())
            rewards_log.append(float(reward))


            for name, p0 in object_init_pos.items():
                cur = obs.get(name + "_pos")
                if cur is not None:
                    d = float(np.sum(np.abs(np.asarray(cur) - p0)))
                    if d > object_max_disp.get(name, 0.0):
                        object_max_disp[name] = d

            total_steps = step + 1

            if total_steps % 50 == 0:
                logger.info(
                    "EVAL_PROGRESS task=%d/%d episode=%d/%d step=%d/%d "
                    "elapsed=%.1fs",
                    task_index,
                    task_total,
                    episode_id + 1,
                    self.config.n_eval_episodes,
                    total_steps,
                    self.config.max_steps_per_episode,
                    time.time() - start_time,
                )

            if done:
                break

        elapsed = time.time() - start_time


        collided = any(d > collision_threshold for d in object_max_disp.values())
        success = bool(done) and not collided

        video_path: Path | None = None
        if record_video and not success and video_frames:
            video_path = self._write_video(
                task_info.name,
                episode_id,
                video_frames,
            )
            logger.info("失敗エピソード動画を保存: %s", video_path)

        return EpisodeResult(
            task_name=task_info.name,
            episode_id=episode_id,
            success=success,
            total_steps=total_steps,
            elapsed_time_sec=elapsed,
            joint_positions=joint_positions,
            ee_positions=ee_positions,
            ee_orientations=ee_orientations,
            gripper_qpos=gripper_qpos_log,
            actions=actions_log,
            rewards=rewards_log,
            collided=collided,
            video_path=str(video_path) if video_path is not None else None,
        )

    @staticmethod
    def _make_video_frame(obs: dict[str, np.ndarray]) -> np.ndarray | None:
        """Create a human-viewable front/wrist frame from a LIBERO observation."""
        frames: list[np.ndarray] = []
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            image = obs.get(key)
            if image is None:
                continue
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] != 3:
                continue
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image, 0.0, 1.0) * 255.0
            image = image.astype(np.uint8, copy=False)
            # Match the orientation used by the pi0.5 LIBERO processor.
            frames.append(np.ascontiguousarray(image[::-1, ::-1]))

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return np.concatenate(frames, axis=1)

    def _write_video(
        self,
        task_name: str,
        episode_id: int,
        frames: list[np.ndarray],
    ) -> Path:
        import imageio.v2 as imageio

        video_dir = self.config.video_dir or self.config.output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name).strip("_")
        path = video_dir / f"{safe_name}__episode_{episode_id + 1:03d}.mp4"
        imageio.mimwrite(
            path,
            frames,
            fps=self.config.video_fps,
            codec="libx264",
            quality=7,
        )
        return path

    def evaluate_tasks(
        self,
        policy: PolicyInterface,
        task_infos: list[TaskInfo],
        perturbation: PerturbationConfig,
    ) -> list[TaskResult]:
        results = []
        for i, task_info in enumerate(task_infos):
            logger.info(
                "=== タスク %d/%d: %s ===",
                i + 1, len(task_infos), task_info.name,
            )
            result = self.evaluate_task(
                policy,
                task_info,
                perturbation,
                task_index=i + 1,
                task_total=len(task_infos),
            )
            results.append(result)
        return results
