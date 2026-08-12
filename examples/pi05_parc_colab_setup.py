#!/usr/bin/env python3
"""Prepare a PARC-like LeRobot/PyTorch pi0.5 runtime in Google Colab.

This helper intentionally keeps the existing SmolVLA notebook environment separate.
It creates a Python 3.10 venv, installs PyTorch 2.11, checks out LeRobot v0.4.4
and the LeRobot-specific Transformers branch, downloads the pi0.5 LIBERO
checkpoint plus the PaliGemma tokenizer, and optionally runs a smoke inference.

Typical Colab use from the repository root:

    !python examples/pi05_parc_colab_setup.py
    !python examples/pi05_parc_colab_setup.py --smoke
    !python examples/pi05_parc_colab_setup.py --skip-download --build-submission

The final command copies pinned LeRobot/Transformers sources into vendor/ and
creates pi05_submission.zip. Generated weights, vendor sources, and the zip are
gitignored; GitHub only stores the reproducible build recipe.

The PaliGemma tokenizer is gated on Hugging Face. Accept its terms and log in to
Hugging Face in Colab before running this script if the download is rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = REPO_ROOT / "submission_template"
MODEL_DIR = SUBMISSION_DIR / "model_weights" / "pi05_libero_finetuned_v044"
TOKENIZER_DIR = SUBMISSION_DIR / "model_weights" / "paligemma-3b-pt-224"

VENV_DIR = Path(os.environ.get("PI05_VENV_DIR", "/content/pi05_py310"))
RUNTIME_DIR = Path(os.environ.get("PI05_RUNTIME_DIR", "/content/pi05_runtime"))
LEROBOT_DIR = RUNTIME_DIR / "lerobot_v044"
TRANSFORMERS_DIR = RUNTIME_DIR / "transformers_lerobot_openpi"

# Moving branches are deliberately avoided so that a submission rebuilt later
# uses exactly the source revision that was smoke-tested here.
LEROBOT_REF = "8fff0fde7c79f23a93d845d1a50e985de01f8b8a"  # v0.4.4
TRANSFORMERS_REF = "dcddb970176382c0fcf4521b0c0e6fc15894dfe0"
PI05_REPO = "lerobot/pi05_libero_finetuned_v044"
PI05_REVISION = "dbf8a3f794a9c4297b44f40b752712f50073d945"
PALIGEMMA_REPO = "google/paligemma-3b-pt-224"
DEFAULT_SUBMISSION_ZIP = REPO_ROOT / "pi05_submission.zip"


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
    else:
        shutil.rmtree(destination, ignore_errors=True)
        run(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--no-checkout",
                url,
                str(destination),
            ]
        )
        run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", ref])

    run(["git", "-C", str(destination), "checkout", "--detach", "--force", ref])
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != ref:
        raise RuntimeError(
            f"Unexpected checkout for {destination}: expected={ref}, actual={actual}"
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

    # Runtime-only dependencies. We intentionally omit torchcodec and rerun-sdk,
    # and avoid letting LeRobot v0.4.4 metadata downgrade PARC's torch 2.11.
    # rerun-sdk 0.24-0.26 requires NumPy >=2, while PARC uses NumPy 1.26.4;
    # it is not needed for PI0.5 policy inference.
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
        "opencv-python-headless==4.11.0.86",
        "datasets>=4.0.0,<5.0.0",
        "diffusers>=0.27.2,<0.36.0",
        "jsonlines>=4.0.0,<5.0.0",
        "deepdiff>=7.0.1,<9.0.0",
        "imageio[ffmpeg]>=2.34.0,<3.0.0",
        "wandb>=0.24.0,<0.25.0",
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
import numpy
import torch
import transformers

print("python runtime OK")
print("numpy       =", numpy.__version__)
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
    model_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors",
    model_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
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


def prepare_vendored_sources() -> None:
    """Copy the two pinned source trees required by offline PARC inference."""
    vendor_dir = SUBMISSION_DIR / "vendor"
    lerobot_vendor = vendor_dir / "lerobot"
    transformers_vendor = vendor_dir / "transformers"

    for target in (lerobot_vendor, transformers_vendor):
        shutil.rmtree(target, ignore_errors=True)

    shutil.copytree(
        LEROBOT_DIR / "src",
        lerobot_vendor / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    shutil.copytree(
        TRANSFORMERS_DIR / "src",
        transformers_vendor / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    shutil.copy2(LEROBOT_DIR / "LICENSE", lerobot_vendor / "LICENSE")
    shutil.copy2(TRANSFORMERS_DIR / "LICENSE", transformers_vendor / "LICENSE")

    manifest = {
        "policy": "LeRobot PyTorch pi0.5 LIBERO",
        "lerobot_commit": LEROBOT_REF,
        "transformers_commit": TRANSFORMERS_REF,
        "model_repo": PI05_REPO,
        "model_revision": PI05_REVISION,
        "tokenizer_repo": PALIGEMMA_REPO,
    }
    (SUBMISSION_DIR / "pi05_build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _submission_files() -> list[Path]:
    excluded_names = {
        ".DS_Store",
        ".git",
        ".gitignore",
        ".ipynb_checkpoints",
        "__pycache__",
        ".pytest_cache",
        # Legacy OpenPI/JAX experiment; the active server is policy_server.py.
        "policy_server_pi05.py",
    }
    files: list[Path] = []
    for path in SUBMISSION_DIR.rglob("*"):
        relative = path.relative_to(SUBMISSION_DIR)
        if any(part in excluded_names for part in relative.parts):
            continue
        if path.is_file() and path.suffix != ".pyc":
            files.append(path)
    return sorted(files)


def build_submission(python: Path, output_path: Path) -> Path:
    """Create an offline, root-level PARC submission zip with ZIP64 enabled."""
    required_assets = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "model.safetensors",
        MODEL_DIR / "policy_preprocessor.json",
        MODEL_DIR / "policy_postprocessor.json",
        MODEL_DIR / "policy_preprocessor_step_2_normalizer_processor.safetensors",
        MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        TOKENIZER_DIR / "tokenizer_config.json",
    ]
    missing = [str(path) for path in required_assets if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot build submission; missing assets: " + ", ".join(missing)
        )

    prepare_vendored_sources()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    files = _submission_files()
    print(f"Building {output_path} from {len(files)} files...")
    with zipfile.ZipFile(output_path, mode="w", allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(SUBMISSION_DIR).as_posix()
            # Model weights are already dense and should not be recompressed.
            if path.suffix in {".safetensors", ".model"} or path.stat().st_size > 32 * 1024 * 1024:
                archive.write(path, relative, compress_type=zipfile.ZIP_STORED)
            else:
                archive.write(
                    path,
                    relative,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                )

    size_gib = output_path.stat().st_size / 2**30
    print(f"Submission zip: {output_path} ({size_gib:.2f} GiB)")
    run(
        [
            str(python),
            str(REPO_ROOT / "validate_submission.py"),
            str(output_path),
            "--static",
        ]
    )
    return output_path


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
    parser.add_argument(
        "--build-submission",
        action="store_true",
        help="Vendor pinned sources and create a validated offline submission zip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SUBMISSION_ZIP,
        help=f"Output zip path (default: {DEFAULT_SUBMISSION_ZIP}).",
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

    if args.build_submission:
        build_submission(python, args.output)


if __name__ == "__main__":
    main()
