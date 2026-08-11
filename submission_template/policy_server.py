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

    REPLAN_STEPS = 5
    DEFAULT_INFERENCE_STEPS = 10

    def __init__(self):
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
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

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
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            str(model_dir),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "tokenizer_processor": {
                    "tokenizer_name": str(tokenizer_dir)
                },
            },
            postprocessor_overrides={
                "device_processor": {"device": "cpu"},
            },
        )

        self.instruction = ""
        print(
            "[pi0.5] model loaded "
            f"in {time.perf_counter() - t0:.2f}s; "
            f"replan_steps={config.n_action_steps}, "
            f"inference_steps={config.num_inference_steps}, "
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
            action = self._select_action(dummy_obs)
            if action.shape != (7,):
                raise RuntimeError(
                    f"Unexpected warmup action shape: {action.shape}"
                )
            self.policy.reset()
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

    def get_action(
        self, obs: dict[str, np.ndarray]
    ) -> np.ndarray:
        # PI05Policy.select_action() が n_action_steps 分の queue を内部管理する。
        return self._select_action(obs)

    def reset(self, instruction: str = "") -> None:
        self.instruction = str(instruction)
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
