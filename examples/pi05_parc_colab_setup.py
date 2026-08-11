#!/usr/bin/env python3
"""Prepare a PARC-like LeRobot/PyTorch pi0.5 runtime in Google Colab.

This helper intentionally keeps the existing SmolVLA notebook environment separate.
It creates a Python 3.10 venv, installs PyTorch 2.11, checks out LeRobot v0.4.4
and the LeRobot-specific Transformers branch, downloads the pi0.5 LIBERO
checkpoint plus the PaliGemma tokenizer, and optionally runs a smoke inference.

Typical Colab use from the repository root:

    !python examples/pi05_parc_colab_setup.py
    !python examples/pi05_parc_colab_setup.py --smoke

The PaliGemma tokenizer is gated on Hugging Face. Accept its terms and log in to
Hugging Face in Colab before running this script if the download is rejected.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = REPO_ROOT / "submission_template"
MODEL_DIR = SUBMISSION_DIR / "model_weights" / "pi05_libero_finetuned_v044"
TOKENIZER_DIR = SUBMISSION_DIR / "model_weights" / "paligemma-3b-pt-224"

VENV_DIR = Path(os.environ.get("PI05_VENV_DIR", "/content/pi05_py310"))
RUNTIME_DIR = Path(os.environ.get("PI05_RUNTIME_DIR", "/content/pi05_runtime"))
LEROBOT_DIR = RUNTIME_DIR / "lerobot_v044"
TRANSFORMERS_DIR = RUNTIME_DIR / "transformers_lerobot_openpi"

LEROBOT_REF = "v0.4.4"
TRANSFORMERS_REF = "fix/lerobot_openpi"
PI05_REPO = "lerobot/pi05_libero_finetuned_v044"
PI05_REVISION = "dbf8a3f794a9c4297b44f40b752712f50073d945"
PALIGEMMA_REPO = "google/paligemma-3b-pt-224"


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("$", " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
    )


def ensure_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv

    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv installation succeeded but executable was not found")
    return uv


def clone_or_checkout(url: str, ref: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if (destination / ".git").is_dir():
        run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", ref])
        run(["git", "-C", str(destination), "checkout", "--force", "FETCH_HEAD"])
        return

    shutil.rmtree(destination, ignore_errors=True)
    run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            ref,
            url,
            str(destination),
        ]
    )


def setup_runtime() -> Path:
    uv = ensure_uv()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    run([uv, "python", "install", "3.10"])

    if not (VENV_DIR / "bin" / "python").is_file():
        run([uv, "venv", "--python", "3.10", "--seed", str(VENV_DIR)])

    python = VENV_DIR / "bin" / "python"

    # Match the PARC production PyTorch/CUDA stack as closely as Colab allows.
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "torch==2.11.0",
            "torchvision==0.26.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ]
    )

    clone_or_checkout(
        "https://github.com/huggingface/lerobot.git",
        LEROBOT_REF,
        LEROBOT_DIR,
    )
    clone_or_checkout(
        "https://github.com/huggingface/transformers.git",
        TRANSFORMERS_REF,
        TRANSFORMERS_DIR,
    )

    # Install the patched Transformers implementation first.
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "-e",
            str(TRANSFORMERS_DIR),
        ]
    )

    # Runtime-only dependencies. We intentionally omit torchcodec and avoid
    # letting LeRobot's v0.4.4 metadata downgrade torch below PARC's 2.11.
    dependencies = [
        "numpy==1.26.4",
        "huggingface-hub>=0.34.2,<0.36.0",
        "accelerate>=1.10.0,<2.0.0",
        "safetensors>=0.4.3,<1.0.0",
        "einops>=0.8.0,<0.9.0",
        "scipy>=1.10.1,<1.15",
        "draccus==0.10.0",
        "gymnasium>=1.1.1,<2.0.0",
        "packaging>=24.2,<26.0",
        "termcolor>=2.4.0,<4.0.0",
        "sentencepiece>=0.2.0",
        "Pillow>=11.0.0,<13.0.0",
        "opencv-python-headless>=4.9.0,<4.13.0",
        "datasets>=4.0.0,<5.0.0",
        "diffusers>=0.27.2,<0.36.0",
        "jsonlines>=4.0.0,<5.0.0",
        "deepdiff>=7.0.1,<9.0.0",
        "imageio[ffmpeg]>=2.34.0,<3.0.0",
        "wandb>=0.24.0,<0.25.0",
        "rerun-sdk>=0.24.0,<0.27.0",
        "pynput>=1.7.7,<1.9.0",
        "pyserial>=3.5,<4.0",
        "av>=15.0.0,<16.0.0",
        "setuptools>=71.0.0,<81.0.0",
        "cmake>=3.29.0.1,<4.2.0",
        "fastapi==0.140.7",
        "uvicorn==0.51.0",
        "msgpack==1.2.1",
    ]
    run([uv, "pip", "install", "--python", str(python), *dependencies])

    # Install LeRobot itself without dependency resolution so torch 2.11 stays intact.
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "-e",
            str(LEROBOT_DIR),
        ]
    )

    verify_code = r'''
import importlib.metadata
import torch
import transformers

print("python runtime OK")
print("torch       =", torch.__version__)
print("torch CUDA  =", torch.version.cuda)
print("CUDA        =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         =", torch.cuda.get_device_name(0))
    print("VRAM GiB    =", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
print("transformers=", transformers.__version__)
print("lerobot     =", importlib.metadata.version("lerobot"))

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
print("PI05 imports = OK")
'''
    run([str(python), "-c", verify_code])
    return python


def download_assets(python: Path) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    code = f'''
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = Path({str(MODEL_DIR)!r})
tokenizer_dir = Path({str(TOKENIZER_DIR)!r})

print("Downloading LeRobot pi0.5 LIBERO checkpoint...")
snapshot_download(
    repo_id={PI05_REPO!r},
    revision={PI05_REVISION!r},
    local_dir=model_dir,
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor*.safetensors",
        "policy_postprocessor*.safetensors",
    ],
)

print("Downloading PaliGemma tokenizer files only...")
try:
    snapshot_download(
        repo_id={PALIGEMMA_REPO!r},
        local_dir=tokenizer_dir,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )
except Exception as exc:
    raise RuntimeError(
        "PaliGemma tokenizer download failed. Accept the model terms on "
        "Hugging Face and log in in Colab, then rerun this script."
    ) from exc

required = [
    model_dir / "config.json",
    model_dir / "model.safetensors",
    model_dir / "policy_preprocessor.json",
    model_dir / "policy_postprocessor.json",
    tokenizer_dir / "tokenizer_config.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing downloaded files: " + ", ".join(missing))

print("checkpoint =", model_dir)
print("tokenizer  =", tokenizer_dir)
'''
    run([str(python), "-c", code])


def smoke_test(python: Path) -> None:
    policy_server = SUBMISSION_DIR / "policy_server.py"
    code = f'''
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ["PI05_MODEL_DIR"] = {str(MODEL_DIR)!r}
os.environ["PI05_TOKENIZER_DIR"] = {str(TOKENIZER_DIR)!r}
os.environ["PI05_DEVICE"] = "cuda"
os.environ["PI05_REPLAN_STEPS"] = "5"
os.environ["PI05_INFERENCE_STEPS"] = "10"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

spec = importlib.util.spec_from_file_location("parc_policy_server", {str(policy_server)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

props = torch.cuda.get_device_properties(0)
print("GPU:", props.name)
print("VRAM GiB:", round(props.total_memory / 2**30, 1))

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
policy = module.MyPolicy()
torch.cuda.synchronize()
print("init+warmed-up sec:", round(time.perf_counter() - t0, 3))

policy.reset("pick up the black bowl and place it on the plate")
obs = {{
    "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
    "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
    "robot0_joint_pos": np.zeros(7, dtype=np.float32),
    "robot0_eef_pos": np.zeros(3, dtype=np.float32),
    "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
}}

torch.cuda.synchronize()
t0 = time.perf_counter()
action = policy.get_action(obs)
torch.cuda.synchronize()
heavy_sec = time.perf_counter() - t0

# Second action should come from the policy's internal action queue.
torch.cuda.synchronize()
t1 = time.perf_counter()
action_cached = policy.get_action(obs)
torch.cuda.synchronize()
cached_sec = time.perf_counter() - t1

print("action shape:", action.shape, action.dtype)
print("heavy /act-equivalent sec:", round(heavy_sec, 3))
print("cached action sec:", round(cached_sec, 6))
print("peak CUDA GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))

if action.shape != (7,) or action.dtype != np.float32:
    raise RuntimeError("Unexpected action format")
if not np.isfinite(action).all():
    raise RuntimeError("Non-finite action")
if heavy_sec >= 10.0:
    raise RuntimeError(f"Inference exceeds PARC 10-second limit: {{heavy_sec:.3f}} sec")

print("PI05_SMOKE_TEST=PASS")
'''

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(LEROBOT_DIR / "src"),
            str(TRANSFORMERS_DIR / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    run([str(python), "-c", code], env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="After setup/download, instantiate MyPolicy and time one heavy inference.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse already downloaded checkpoint/tokenizer files.",
    )
    args = parser.parse_args()

    python = setup_runtime()

    if not args.skip_download:
        download_assets(python)

    print("\nPreparation completed.")
    print("Python:", python)
    print("Model:", MODEL_DIR)
    print("Tokenizer:", TOKENIZER_DIR)

    if args.smoke:
        smoke_test(python)
    else:
        print(
            "Next: python examples/pi05_parc_colab_setup.py --smoke "
            "(use --skip-download to avoid re-checking the downloads)"
        )


if __name__ == "__main__":
    main()
