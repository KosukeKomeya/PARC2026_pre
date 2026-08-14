"""ポリシーサーバー（提出用テンプレート）

このファイルを編集して、自分のモデルを組み込んでください。
編集が必要なのは MyPolicy クラスの中身だけです。
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可です。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

import argparse
from abc import ABC, abstractmethod

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。以下のキーが含まれる:
                - "agentview_image": (128, 128, 3) uint8
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8
                - "robot0_joint_pos": (7,) float
                - "robot0_eef_pos": (3,) float
                - "robot0_eef_quat": (4,) float
                - "robot0_gripper_qpos": (2,) float

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


class MyPolicy(BasePolicy):
    """LeRobot v0.4.4 / PyTorch π0.5-LIBERO を使う PARC Track 1 policy.

    採点時は外部通信を使わず、checkpoint と PaliGemma tokenizer を
    model_weights/ からローカルロードする。

    必要なら LeRobot / transformers の source を vendor/ に同梱できる。
    """

    # The local paired evaluation kept collision-free success at 90% while
    # reducing wall-clock time by about 22.6% versus five actions/chunk.
    REPLAN_STEPS = 10
    DEFAULT_INFERENCE_STEPS = 10
    DEFAULT_TEMPORAL_ENSEMBLE = False
    DEFAULT_ENSEMBLE_STEPS = 3
    DEFAULT_ENSEMBLE_OLD_WEIGHTS = (0.25, 0.15, 0.05)
    DEFAULT_RTC_ENABLED = False
    DEFAULT_RTC_EXECUTION_HORIZON = 10
    DEFAULT_RTC_MAX_GUIDANCE_WEIGHT = 5.0
    DEFAULT_RTC_SCHEDULE = "EXP"
    DEFAULT_RTC_INFERENCE_DELAY = 0

    def __init__(self):
        import collections
        import math
        import os
        import sys
        import time
        from pathlib import Path

        root = Path(__file__).resolve().parent

        # vendored source があれば、採点環境側のパッケージより優先する。
        vendor_transformers = root / "vendor" / "transformers" / "src"
        vendor_lerobot = root / "vendor" / "lerobot" / "src"
        if vendor_transformers.is_dir():
            sys.path.insert(0, str(vendor_transformers))
        if vendor_lerobot.is_dir():
            sys.path.insert(0, str(vendor_lerobot))

        # 採点環境は外部通信不可なので、推論サーバー自身もオフライン固定にする。
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        deterministic_value = os.environ.get(
            "PI05_DETERMINISTIC_EPISODES", "0"
        ).strip().lower()
        if deterministic_value not in {"0", "1", "false", "true"}:
            raise ValueError(
                "PI05_DETERMINISTIC_EPISODES must be one of 0/1/false/true"
            )
        self._deterministic_episodes = deterministic_value in {"1", "true"}
        self._policy_seed = int(os.environ.get("PI05_POLICY_SEED", "20260814"))
        self._instruction_episode_counts: dict[str, int] = {}

        model_dir = Path(
            os.environ.get(
                "PI05_MODEL_DIR",
                str(root / "model_weights" / "pi05_libero_finetuned_v044"),
            )
        )
        tokenizer_dir = Path(
            os.environ.get(
                "PI05_TOKENIZER_DIR",
                str(root / "model_weights" / "paligemma-3b-pt-224"),
            )
        )

        if not model_dir.is_dir():
            raise FileNotFoundError(
                "LeRobot π0.5 checkpoint が見つかりません。"
                f" expected={model_dir}"
            )
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                "model.safetensors が見つかりません。"
                f" expected={model_dir / 'model.safetensors'}"
            )
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(
                "PaliGemma tokenizer が見つかりません。"
                f" expected={tokenizer_dir}"
            )

        import torch

        device_name = os.environ.get("PI05_DEVICE", "cuda")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "PI05_DEVICE=cuda ですが CUDA が利用できません。"
            )
        self.device = torch.device(device_name)

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05 import PI05Config, PI05Policy
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        print(
            "[pi0.5] loading LeRobot/PyTorch policy "
            f"from {model_dir} on {self.device}..."
        )
        t0 = time.perf_counter()

        # v0.4.4 は基底 PreTrainedConfig から type=pi05 を解決する。
        # PI05Config.from_pretrained() を直接呼ぶと config.json の
        # "type" フィールドを PI05Config 自身が受け取って失敗する。
        config = PreTrainedConfig.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        if not isinstance(config, PI05Config):
            raise TypeError(
                "Expected PI05Config, got "
                f"{type(config).__name__} from {model_dir}"
            )

        config.device = str(self.device)
        config.compile_model = False
        config.gradient_checkpointing = False
        config.n_action_steps = int(
            os.environ.get("PI05_REPLAN_STEPS", self.REPLAN_STEPS)
        )
        config.num_inference_steps = int(
            os.environ.get(
                "PI05_INFERENCE_STEPS",
                self.DEFAULT_INFERENCE_STEPS,
            )
        )

        if config.n_action_steps <= 0:
            raise ValueError("PI05_REPLAN_STEPS must be >= 1")
        if config.n_action_steps > config.chunk_size:
            raise ValueError(
                "PI05_REPLAN_STEPS exceeds chunk_size: "
                f"{config.n_action_steps} > {config.chunk_size}"
            )
        if config.num_inference_steps <= 0:
            raise ValueError("PI05_INFERENCE_STEPS must be >= 1")

        ensemble_value = os.environ.get(
            "PI05_TEMPORAL_ENSEMBLE",
            "1" if self.DEFAULT_TEMPORAL_ENSEMBLE else "0",
        ).strip().lower()
        if ensemble_value not in {"0", "1", "false", "true"}:
            raise ValueError(
                "PI05_TEMPORAL_ENSEMBLE must be one of 0/1/false/true"
            )
        self.temporal_ensemble = ensemble_value in {"1", "true"}
        self.ensemble_steps = int(
            os.environ.get(
                "PI05_ENSEMBLE_STEPS",
                self.DEFAULT_ENSEMBLE_STEPS,
            )
        )
        weights_text = os.environ.get(
            "PI05_ENSEMBLE_OLD_WEIGHTS",
            ",".join(
                str(weight)
                for weight in self.DEFAULT_ENSEMBLE_OLD_WEIGHTS
            ),
        )
        try:
            self.ensemble_old_weights = tuple(
                float(value.strip())
                for value in weights_text.split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "PI05_ENSEMBLE_OLD_WEIGHTS must be comma-separated floats"
            ) from exc
        if self.ensemble_steps < 0:
            raise ValueError("PI05_ENSEMBLE_STEPS must be >= 0")
        if self.ensemble_steps > config.n_action_steps:
            raise ValueError(
                "PI05_ENSEMBLE_STEPS exceeds PI05_REPLAN_STEPS: "
                f"{self.ensemble_steps} > {config.n_action_steps}"
            )
        if len(self.ensemble_old_weights) < self.ensemble_steps:
            raise ValueError(
                "PI05_ENSEMBLE_OLD_WEIGHTS needs at least "
                f"{self.ensemble_steps} values"
            )
        if any(
            not 0.0 <= weight < 1.0
            for weight in self.ensemble_old_weights[: self.ensemble_steps]
        ):
            raise ValueError(
                "PI05_ENSEMBLE_OLD_WEIGHTS values must be in [0, 1)"
            )

        rtc_value = os.environ.get(
            "PI05_RTC_ENABLED",
            "1" if self.DEFAULT_RTC_ENABLED else "0",
        ).strip().lower()
        if rtc_value not in {"0", "1", "false", "true"}:
            raise ValueError(
                "PI05_RTC_ENABLED must be one of 0/1/false/true"
            )
        self.rtc_enabled = rtc_value in {"1", "true"}
        self.rtc_execution_horizon = int(
            os.environ.get(
                "PI05_RTC_EXECUTION_HORIZON",
                self.DEFAULT_RTC_EXECUTION_HORIZON,
            )
        )
        self.rtc_max_guidance_weight = float(
            os.environ.get(
                "PI05_RTC_MAX_GUIDANCE_WEIGHT",
                self.DEFAULT_RTC_MAX_GUIDANCE_WEIGHT,
            )
        )
        self.rtc_schedule = os.environ.get(
            "PI05_RTC_SCHEDULE",
            self.DEFAULT_RTC_SCHEDULE,
        ).strip().upper()
        self.rtc_inference_delay = int(
            os.environ.get(
                "PI05_RTC_INFERENCE_DELAY",
                self.DEFAULT_RTC_INFERENCE_DELAY,
            )
        )
        if self.rtc_enabled and self.temporal_ensemble:
            raise ValueError(
                "PI05_RTC_ENABLED and PI05_TEMPORAL_ENSEMBLE "
                "cannot both be enabled"
            )
        if not 1 <= self.rtc_execution_horizon <= config.chunk_size:
            raise ValueError(
                "PI05_RTC_EXECUTION_HORIZON must be in [1, chunk_size]"
            )
        if self.rtc_inference_delay < 0:
            raise ValueError("PI05_RTC_INFERENCE_DELAY must be >= 0")
        if self.rtc_inference_delay > self.rtc_execution_horizon:
            raise ValueError(
                "PI05_RTC_INFERENCE_DELAY exceeds execution horizon"
            )
        if (
            not math.isfinite(self.rtc_max_guidance_weight)
            or self.rtc_max_guidance_weight <= 0
        ):
            raise ValueError(
                "PI05_RTC_MAX_GUIDANCE_WEIGHT must be finite and > 0"
            )
        rtc_schedules = {"ZEROS", "ONES", "LINEAR", "EXP"}
        if self.rtc_schedule not in rtc_schedules:
            raise ValueError(
                "PI05_RTC_SCHEDULE must be one of "
                f"{sorted(rtc_schedules)}"
            )

        if self.rtc_enabled:
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            config.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=self.rtc_execution_horizon,
                max_guidance_weight=self.rtc_max_guidance_weight,
                prefix_attention_schedule=RTCAttentionSchedule[
                    self.rtc_schedule
                ],
            )
        else:
            config.rtc_config = None

        self._ensemble_action_queue = collections.deque()
        self._previous_raw_chunk = None
        self._rtc_action_queue = collections.deque()
        self._rtc_previous_raw_chunk = None

        self.policy = PI05Policy.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            strict=True,
        )
        self.policy.to(self.device)
        self.policy.eval()

        # checkpoint に保存された正規化統計をそのまま使う。
        # tokenizer だけ提出物内のローカルディレクトリへ差し替える。
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(model_dir),
            config_filename="policy_preprocessor.json",
            local_files_only=True,
            overrides={
                "device_processor": {"device": str(self.device)},
                "tokenizer_processor": {
                    "tokenizer_name": str(tokenizer_dir)
                },
            },
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(model_dir),
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            overrides={
                "device_processor": {"device": "cpu"},
            },
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        self.instruction = ""
        print(
            "[pi0.5] model loaded "
            f"in {time.perf_counter() - t0:.2f}s; "
            f"replan_steps={config.n_action_steps}, "
            f"inference_steps={config.num_inference_steps}, "
            f"temporal_ensemble={self.temporal_ensemble}, "
            f"ensemble_steps={self.ensemble_steps}, "
            f"rtc={self.rtc_enabled}, "
            f"rtc_horizon={self.rtc_execution_horizon}, "
            f"rtc_guidance={self.rtc_max_guidance_weight}, "
            f"rtc_schedule={self.rtc_schedule}, "
            f"rtc_delay={self.rtc_inference_delay}, "
            f"dtype={config.dtype}"
        )

        # 最初の /act だけが遅くならないよう、サーバー起動前に一度推論する。
        if os.environ.get("PI05_SKIP_WARMUP", "0") != "1":
            print("[pi0.5] warming up inference...")
            t0 = time.perf_counter()
            dummy_obs = {
                "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros(
                    (128, 128, 3), dtype=np.uint8
                ),
                "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                "robot0_eef_quat": np.array(
                    [0.0, 0.0, 0.0, 1.0], dtype=np.float32
                ),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            }
            self.instruction = "do something"
            if self.rtc_enabled:
                self._refill_rtc_queue(dummy_obs)
                self._rtc_action_queue.clear()
                self._refill_rtc_queue(dummy_obs)
                action = self._rtc_action_queue.popleft()
            else:
                action = self._select_action(dummy_obs)
            if action.shape != (7,):
                raise RuntimeError(
                    f"Unexpected warmup action shape: {action.shape}"
                )
            self.policy.reset()
            self._rtc_action_queue.clear()
            self._rtc_previous_raw_chunk = None
            self.instruction = ""
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            print(
                "[pi0.5] warmup finished "
                f"in {time.perf_counter() - t0:.2f}s"
            )

    @staticmethod
    def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
        """[x, y, z, w] quaternion を3D axis-angle vectorへ変換する。"""
        import math

        q = np.asarray(quat, dtype=np.float64)
        if q.shape != (4,):
            raise ValueError(
                f"Expected quaternion shape (4,), got {q.shape}"
            )

        w = float(np.clip(q[3], -1.0, 1.0))
        den = math.sqrt(max(0.0, 1.0 - w * w))
        if den <= 1e-10:
            return np.zeros(3, dtype=np.float32)

        angle = 2.0 * math.acos(w)
        axis = q[:3] / den
        return (axis * angle).astype(np.float32)

    @staticmethod
    def _prepare_image(image: np.ndarray):
        """LIBEROと同じ向きにし、LeRobot用 CHW float32 [0,1] tensorへ変換。"""
        import torch

        image = np.asarray(image)
        if image.shape != (128, 128, 3):
            raise ValueError(
                "Unexpected PARC image shape: "
                f"{image.shape}; expected (128, 128, 3)"
            )

        # LeRobot v0.4.4 LiberoProcessorStep と同じ180度回転。
        image = np.ascontiguousarray(image[::-1, ::-1])

        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)
            tensor = torch.from_numpy(
                image.astype(np.float32, copy=False)
            )
        else:
            tensor = torch.from_numpy(
                image.astype(np.float32, copy=False)
            ) / 255.0

        return tensor.permute(2, 0, 1).contiguous()

    def _make_lerobot_observation(
        self, obs: dict[str, np.ndarray]
    ) -> dict:
        """PARC observation をLeRobot π0.5 processor入力へ変換する。"""
        import torch

        state = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                self._quat2axisangle(obs["robot0_eef_quat"]),
                np.asarray(
                    obs["robot0_gripper_qpos"], dtype=np.float32
                ),
            )
        ).astype(np.float32)

        if state.shape != (8,):
            raise RuntimeError(
                f"Unexpected state shape: {state.shape}; expected (8,)"
            )

        return {
            "observation.images.image": self._prepare_image(
                obs["agentview_image"]
            ),
            "observation.images.image2": self._prepare_image(
                obs["robot0_eye_in_hand_image"]
            ),
            "observation.state": torch.from_numpy(state),
            "task": self.instruction,
        }

    def _select_action(
        self, obs: dict[str, np.ndarray]
    ) -> np.ndarray:
        import torch

        batch = self._make_lerobot_observation(obs)
        batch = self.preprocessor(batch)

        with torch.inference_mode():
            action = self.policy.select_action(batch)

        action = self.postprocessor(action)
        action = action.squeeze(0).detach().to(
            "cpu", dtype=torch.float32
        )
        action = action.numpy()

        if action.shape != (7,):
            raise RuntimeError(
                f"Unexpected π0.5 action shape: {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"π0.5 returned non-finite action: {action}"
            )

        return action.astype(np.float32, copy=False)

    def _predict_raw_action_chunk(
        self, obs: dict[str, np.ndarray]
    ):
        """Predict one complete normalized chunk for sparse ensembling."""
        import torch

        batch = self._make_lerobot_observation(obs)
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)

        if actions.ndim != 3 or actions.shape[0] != 1:
            raise RuntimeError(
                "pi0.5 action chunk must have shape (1, steps, dims), got "
                f"{tuple(actions.shape)}"
            )
        required_steps = self.policy.config.n_action_steps + self.ensemble_steps
        if actions.shape[1] < required_steps:
            raise RuntimeError(
                "pi0.5 action chunk is too short for boundary ensembling: "
                f"{actions.shape[1]} < {required_steps}"
            )
        return actions.detach()

    def _decode_raw_action(self, raw_action) -> np.ndarray:
        """Convert one normalized model action to the submitted 7-D action."""
        import torch

        action = self.postprocessor(raw_action)
        action = action.squeeze(0).detach().to(
            "cpu", dtype=torch.float32
        ).numpy()
        if action.shape != (7,):
            raise RuntimeError(
                f"Unexpected pi0.5 decoded action shape: {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"pi0.5 returned non-finite decoded action: {action}"
            )
        return action

    def _refill_ensemble_queue(
        self, obs: dict[str, np.ndarray]
    ) -> None:
        """Blend aligned predictions only at a sparse replan boundary.

        Position and rotation use predictions from the previous and current
        chunks. The gripper always uses the newest prediction so an open/close
        transition is never diluted by averaging.
        """
        new_chunk = self._predict_raw_action_chunk(obs)
        replan_steps = self.policy.config.n_action_steps

        for index in range(replan_steps):
            action = self._decode_raw_action(new_chunk[:, index, :])
            if (
                self._previous_raw_chunk is not None
                and index < self.ensemble_steps
            ):
                old_index = replan_steps + index
                old_action = self._decode_raw_action(
                    self._previous_raw_chunk[:, old_index, :]
                )
                old_weight = self.ensemble_old_weights[index]
                action[:6] = (
                    old_weight * old_action[:6]
                    + (1.0 - old_weight) * action[:6]
                )
            self._ensemble_action_queue.append(
                action.astype(np.float32, copy=False)
            )

        self._previous_raw_chunk = new_chunk

    def _predict_rtc_raw_action_chunk(
        self, obs: dict[str, np.ndarray]
    ):
        """Generate one normalized chunk with native pi0.5 RTC guidance."""
        batch = self._make_lerobot_observation(obs)
        batch = self.preprocessor(batch)
        previous_left_over = None
        if self._rtc_previous_raw_chunk is not None:
            previous_left_over = self._rtc_previous_raw_chunk[
                :, self.policy.config.n_action_steps :, :
            ]

        # Do not wrap this call in torch.inference_mode(): RTC temporarily
        # enables autograd to compute its vector-Jacobian correction.
        actions = self.policy.predict_action_chunk(
            batch,
            inference_delay=self.rtc_inference_delay,
            prev_chunk_left_over=previous_left_over,
            execution_horizon=self.rtc_execution_horizon,
        )
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise RuntimeError(
                "pi0.5 RTC chunk must have shape (1, steps, dims), got "
                f"{tuple(actions.shape)}"
            )
        if actions.shape[1] < self.policy.config.n_action_steps:
            raise RuntimeError(
                "pi0.5 RTC chunk is shorter than n_action_steps"
            )
        return actions.detach()

    def _refill_rtc_queue(self, obs: dict[str, np.ndarray]) -> None:
        new_chunk = self._predict_rtc_raw_action_chunk(obs)
        for index in range(self.policy.config.n_action_steps):
            self._rtc_action_queue.append(
                self._decode_raw_action(new_chunk[:, index, :])
            )
        self._rtc_previous_raw_chunk = new_chunk

    def get_action(
        self, obs: dict[str, np.ndarray]
    ) -> np.ndarray:
        if self.rtc_enabled:
            if not self._rtc_action_queue:
                self._refill_rtc_queue(obs)
            return self._rtc_action_queue.popleft()

        # PI05Policy.select_action() が n_action_steps 分の queue を内部管理する。
        if not self.temporal_ensemble:
            return self._select_action(obs)

        if not self._ensemble_action_queue:
            self._refill_ensemble_queue(obs)
        return self._ensemble_action_queue.popleft()

    def reset(self, instruction: str = "") -> None:
        self.instruction = str(instruction)
        self._ensemble_action_queue.clear()
        self._previous_raw_chunk = None
        self._rtc_action_queue.clear()
        self._rtc_previous_raw_chunk = None
        if self._deterministic_episodes:
            import hashlib
            import torch

            episode_index = self._instruction_episode_counts.get(
                self.instruction, 0
            )
            self._instruction_episode_counts[self.instruction] = (
                episode_index + 1
            )
            payload = (
                f"{self._policy_seed}\0{self.instruction}\0{episode_index}"
            ).encode("utf-8")
            episode_seed = int.from_bytes(
                hashlib.sha256(payload).digest()[:8], "little"
            ) % (2**31)
            torch.manual_seed(episode_seed)
        if hasattr(self, "policy"):
            self.policy.reset()


# ============================================================
# 以下は変更不可
# ============================================================


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
