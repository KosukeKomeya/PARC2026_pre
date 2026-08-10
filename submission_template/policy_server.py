"""ポリシーサーバー（提出用テンプレート）

PARC 2026 Track 1 / pi0.5-LIBERO baseline adapter.

この版では、サーバー部分は変更せず MyPolicy の中だけで:
- OpenPI の pi05_libero checkpoint をロード
- PARC observation を OpenPI LIBERO 形式へ変換
- 画像を公式 LIBERO 評価と同じ向き・サイズに前処理
- quaternion を axis-angle に変換して 8D state を作成
- action chunk をキャッシュして 5 step ごとに再推論

を行う。
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
            instruction: タスクの言語指示
        """
        ...


# ============================================================
# ここを編集する
# ============================================================


class MyPolicy(BasePolicy):
    """OpenPI pi0.5-LIBERO を PARC Track 1 に接続する baseline policy.

    想定する提出ディレクトリ:

        submission/
        ├── policy_server.py
        ├── requirements.txt
        ├── model_weights/
        │   └── pi05_libero/
        │       ├── params/
        │       └── assets/
        └── vendor/
            └── openpi/
                └── src/
                    └── openpi/
                        └── ...

    checkpoint と OpenPI source は提出物内に同梱する前提。
    """

    # OpenPI 公式 LIBERO evaluator の既定値。
    RESIZE_SIZE = 224
    REPLAN_STEPS = 5

    def __init__(self):
        import collections
        import os
        import sys
        import time
        from pathlib import Path

        root = Path(__file__).resolve().parent

        # ------------------------------------------------------------
        # 1. OpenPI を import できるようにする
        # ------------------------------------------------------------
        # 採点イメージ側に互換 OpenPI が入っていればそれを優先。
        # 入っていない場合は、提出物に同梱した vendor/openpi/src を使う。
        import importlib.util

        if importlib.util.find_spec("openpi") is None:
            openpi_src = root / "vendor" / "openpi" / "src"
            if not openpi_src.is_dir():
                raise FileNotFoundError(
                    "OpenPI が import できず、vendored source も見つかりません。"
                    f" expected={openpi_src}"
                )
            sys.path.insert(0, str(openpi_src))

        # ------------------------------------------------------------
        # 2. ローカル checkpoint を指定
        # ------------------------------------------------------------
        # ローカル検証時は環境変数 PI05_CHECKPOINT_DIR で変更可能。
        checkpoint_dir = Path(
            os.environ.get(
                "PI05_CHECKPOINT_DIR",
                str(root / "model_weights" / "pi05_libero"),
            )
        )
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                "pi0.5-LIBERO checkpoint が見つかりません。"
                f" expected={checkpoint_dir}"
            )

        # JAX を import する前に設定。
        # 採点機の GPU メモリを過剰に予約しないよう、必要なら外部から上書き可能にする。
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        # import は sys.path 設定後に行う。
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        print("[pi0.5] loading pi05_libero policy...")
        t0 = time.perf_counter()

        train_config = _config.get_config("pi05_libero")
        self.policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
        )

        print(f"[pi0.5] model loaded in {time.perf_counter() - t0:.2f}s")

        # action chunk の残りを保存するキュー。
        self.action_queue = collections.deque()
        self.instruction = ""

        # ------------------------------------------------------------
        # 3. JAX の初回コンパイルを server 起動中に済ませる
        # ------------------------------------------------------------
        # PARC では /act 1回に 10 秒制限がある。
        # JAX の初回 infer は JIT compile を伴って遅くなり得るため、
        # HTTP server が起動する前の __init__ で一度 dummy inference を行う。
        print("[pi0.5] warming up / compiling inference...")
        t0 = time.perf_counter()

        dummy = {
            "observation/image": np.zeros(
                (self.RESIZE_SIZE, self.RESIZE_SIZE, 3), dtype=np.uint8
            ),
            "observation/wrist_image": np.zeros(
                (self.RESIZE_SIZE, self.RESIZE_SIZE, 3), dtype=np.uint8
            ),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "do something",
        }
        warmup_result = self.policy.infer(dummy)

        warmup_actions = np.asarray(warmup_result["actions"])
        if warmup_actions.ndim != 2 or warmup_actions.shape[1] < 7:
            raise RuntimeError(
                "Unexpected pi0.5 output shape during warmup: "
                f"{warmup_actions.shape}"
            )

        print(
            "[pi0.5] warmup finished "
            f"in {time.perf_counter() - t0:.2f}s, "
            f"chunk_shape={warmup_actions.shape}"
        )

    @staticmethod
    def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
        """LIBERO/robosuite と同じ quaternion -> axis-angle 変換。

        quat は [x, y, z, w] を想定する。
        戻り値は 3D の axis-angle vector。
        """
        import math

        q = np.asarray(quat, dtype=np.float64).copy()

        # 数値誤差で w が [-1, 1] をわずかに外れる場合を防ぐ。
        q[3] = np.clip(q[3], -1.0, 1.0)

        den = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
        if math.isclose(float(den), 0.0, abs_tol=1e-8):
            return np.zeros(3, dtype=np.float32)

        axis_angle = (q[:3] * 2.0 * math.acos(float(q[3]))) / den
        return axis_angle.astype(np.float32)

    @staticmethod
    def _resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
        """OpenPI client の resize_with_pad と同等の前処理を1枚に適用する。"""
        from PIL import Image

        image = np.asarray(image)

        if np.issubdtype(image.dtype, np.floating):
            image = (255.0 * image).clip(0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8, copy=False)

        if image.shape[:2] == (size, size):
            return np.ascontiguousarray(image)

        pil = Image.fromarray(image)
        cur_width, cur_height = pil.size

        ratio = max(cur_width / size, cur_height / size)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)

        resized = pil.resize(
            (resized_width, resized_height),
            resample=Image.BILINEAR,
        )

        canvas = Image.new(resized.mode, (size, size), 0)
        pad_height = max(0, int((size - resized_height) / 2))
        pad_width = max(0, int((size - resized_width) / 2))
        canvas.paste(resized, (pad_width, pad_height))

        return np.asarray(canvas, dtype=np.uint8)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """PARC画像 -> OpenPI LIBERO評価時の画像形式。

        OpenPI公式 evaluator に合わせ、
        1) 180度回転
        2) 224x224 に resize_with_pad
        を行う。
        """
        # [::-1, ::-1] = 高さ方向・幅方向を両方反転 = 180°回転
        image = np.ascontiguousarray(image[::-1, ::-1])
        return self._resize_with_pad(image, self.RESIZE_SIZE)

    def _make_openpi_observation(
        self, obs: dict[str, np.ndarray]
    ) -> dict:
        """PARC observation を OpenPI LiberoInputs が期待する形式へ変換。"""

        base_image = self._prepare_image(obs["agentview_image"])
        wrist_image = self._prepare_image(
            obs["robot0_eye_in_hand_image"]
        )

        # π0.5-LIBERO の proprioceptive state:
        # [EEF position(3), EEF orientation axis-angle(3), gripper qpos(2)]
        # = 8 dimensions
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
                f"Unexpected state shape: {state.shape}, expected (8,)"
            )

        return {
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "prompt": self.instruction,
        }

    def _infer_action_chunk(
        self, obs: dict[str, np.ndarray]
    ) -> np.ndarray:
        """現在の観測から π0.5 の action chunk を1回推論する。"""
        element = self._make_openpi_observation(obs)
        result = self.policy.infer(element)

        actions = np.asarray(result["actions"], dtype=np.float32)

        if actions.ndim != 2:
            raise RuntimeError(
                f"pi0.5 actions must be 2D, got shape={actions.shape}"
            )

        if actions.shape[1] < 7:
            raise RuntimeError(
                "pi0.5 action dimension is too small: "
                f"shape={actions.shape}"
            )

        if len(actions) < self.REPLAN_STEPS:
            raise RuntimeError(
                f"Need at least {self.REPLAN_STEPS} actions, "
                f"but model returned {len(actions)}"
            )

        # LIBERO action は最初の7次元だけを利用する。
        return actions[:, :7]

    def get_action(
        self, obs: dict[str, np.ndarray]
    ) -> np.ndarray:
        # action_queue が空になったら新しい chunk を推論する。
        if not self.action_queue:
            action_chunk = self._infer_action_chunk(obs)

            # OpenPI公式 LIBERO evaluator と同じく、
            # chunk 全部ではなく先頭5 stepだけ実行して再推論する。
            for action in action_chunk[: self.REPLAN_STEPS]:
                self.action_queue.append(
                    np.asarray(action, dtype=np.float32)
                )

        action = self.action_queue.popleft()

        if action.shape != (7,):
            raise RuntimeError(
                f"Unexpected action shape: {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"pi0.5 returned non-finite action: {action}"
            )

        # baseline 再現を優先し、ここでは clipping / smoothing はしない。
        return action.astype(np.float32, copy=False)

    def reset(self, instruction: str = "") -> None:
        # 前エピソードの action chunk が残らないように必ず消す。
        self.action_queue.clear()
        self.instruction = str(instruction)


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
