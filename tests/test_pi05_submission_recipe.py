import ast
import importlib.util
import json
import os
import sys
import types
from argparse import Namespace
from fractions import Fraction
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_setup_module():
    path = ROOT / "examples" / "pi05_parc_colab_setup.py"
    spec = importlib.util.spec_from_file_location("pi05_parc_colab_setup", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_lora_module():
    path = ROOT / "examples" / "pi05_action_expert_lora.py"
    spec = importlib.util.spec_from_file_location("pi05_action_expert_lora", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_policy_server_changes_are_scoped_to_mypolicy_contract():
    source = (ROOT / "submission_template" / "policy_server.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    policy = classes["MyPolicy"]
    methods = {
        node.name
        for node in policy.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert {"__init__", "get_action", "reset"} <= methods
    for endpoint in ("/health", "/reset", "/act"):
        assert endpoint in source


def test_pi05_replan_10_and_sparse_boundary_ensemble_are_configurable():
    source = (ROOT / "submission_template" / "policy_server.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    policy = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MyPolicy"
    )
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in policy.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }

    assert constants["REPLAN_STEPS"] == 10
    assert constants["DEFAULT_INFERENCE_STEPS"] == 10
    assert constants["DEFAULT_TEMPORAL_ENSEMBLE"] is False
    assert constants["DEFAULT_ENSEMBLE_STEPS"] == 3
    assert constants["DEFAULT_ENSEMBLE_OLD_WEIGHTS"] == (0.25, 0.15, 0.05)
    assert "PI05_TEMPORAL_ENSEMBLE" in source
    assert "PI05_ENSEMBLE_STEPS" in source
    assert "PI05_ENSEMBLE_OLD_WEIGHTS" in source
    assert "self.policy.predict_action_chunk(batch)" in source
    assert "old_index = replan_steps + index" in source
    assert "action[:6]" in source


def test_pi05_sources_and_model_are_commit_pinned():
    setup = _load_setup_module()
    assert len(setup.LEROBOT_REF) == 40
    assert len(setup.TRANSFORMERS_REF) == 40
    assert len(setup.PI05_REVISION) == 40
    int(setup.LEROBOT_REF, 16)
    int(setup.TRANSFORMERS_REF, 16)
    int(setup.PI05_REVISION, 16)


def test_submission_recipe_requires_normalization_statistics():
    source = (
        ROOT / "examples" / "pi05_parc_colab_setup.py"
    ).read_text(encoding="utf-8")
    assert "policy_preprocessor_step_2_normalizer_processor.safetensors" in source
    assert "policy_postprocessor_step_0_unnormalizer_processor.safetensors" in source


def test_submission_recipe_excludes_legacy_server():
    setup = _load_setup_module()
    relative_files = {
        path.relative_to(setup.SUBMISSION_DIR).as_posix()
        for path in setup._submission_files()
    }
    assert "policy_server.py" in relative_files
    assert "requirements.txt" in relative_files
    assert "policy_server_pi05.py" not in relative_files


def test_submission_recipe_excludes_unused_linux_input_dependencies():
    requirements = (
        ROOT / "submission_template" / "requirements.txt"
    ).read_text(encoding="utf-8").lower()
    setup_source = (
        ROOT / "examples" / "pi05_parc_colab_setup.py"
    ).read_text(encoding="utf-8").lower()
    assert "pynput" not in requirements
    assert "evdev" not in requirements
    assert "pynput" not in setup_source


def test_pi05_colab_notebook_is_valid_and_builds_submission():
    notebook_path = ROOT / "examples" / "pi05_parc_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    all_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "--smoke" in all_source
    assert "--reuse-runtime" in all_source
    assert "--build-submission" in all_source
    assert "pi05_submission.zip" in all_source
    assert "pi05_action_expert_lora.py" in all_source
    assert "--model-source" in all_source
    assert "RUN_LORA_TRAINING = True" in all_source


def test_pi05_qkvo_experiment_notebook_is_isolated_and_recoverable():
    notebook_path = ROOT / "examples" / "pi05_qkvo_experiment_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    all_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert '"--lora-target-profile", "qkvo"' in all_source
    assert "pi05_action_expert_lora_qkvo" in all_source
    assert "pi05_action_expert_lora_full40" in all_source
    assert "pi05_lora_merge_manifest.json" in all_source
    assert "subprocess.Popen" in all_source
    assert "PYTHONUNBUFFERED" in all_source
    assert "QKVO_BEATS_QV" in all_source
    assert "QV_REFERENCE_LOSS = 0.027940072183810116" in all_source
    assert "drive/MyDrive/PARC2026/pi05_action_expert_lora_qkvo" in all_source


def test_pi05_lora_targets_only_action_side_modules():
    workflow = _load_lora_module()
    targets = workflow.ACTION_EXPERT_LORA_TARGETS
    assert "gemma_expert" in targets
    assert "action_in_proj" in targets
    assert "action_out_proj" in targets
    assert "paligemma" not in targets
    assert len(workflow.DEFAULT_DATASET_REVISION) == 40
    int(workflow.DEFAULT_DATASET_REVISION, 16)
    assert workflow.DEFAULT_DATASET_REPO == "lerobot/libero"
    qkvo_targets = workflow.ACTION_EXPERT_LORA_PROFILES["qkvo"]
    for projection in ("q", "k", "v", "o"):
        assert projection in qkvo_targets
    assert "paligemma" not in qkvo_targets


def test_pi05_lora_split_is_balanced_deterministic_and_episode_disjoint():
    workflow = _load_lora_module()
    groups = {
        "task b": [10, 11, 12, 13, 14],
        "task a": [0, 1, 2, 3, 4, 5],
    }
    first = workflow.balanced_episode_split(
        groups,
        seed=7,
        validation_per_task=1,
        max_per_task=None,
    )
    second = workflow.balanced_episode_split(
        groups,
        seed=7,
        validation_per_task=1,
        max_per_task=None,
    )
    assert first == second
    train, validation, by_task = first
    assert set(train).isdisjoint(validation)
    assert {len(parts["train"]) for parts in by_task.values()} == {4}
    assert {len(parts["validation"]) for parts in by_task.values()} == {1}


def test_pi05_lora_recovers_episode_tasks_from_frame_indices():
    workflow = _load_lora_module()
    groups = workflow._groups_from_task_indices(
        episode_indices=[0, 0, 1, 1, 2, 2],
        task_indices=[0, 0, 1, 1, 0, 0],
        task_names_by_index={0: "task zero", 1: "task one"},
    )
    assert groups == {"task zero": [0, 2], "task one": [1]}

    with pytest.raises(ValueError, match="multiple task indices"):
        workflow._groups_from_task_indices(
            episode_indices=[0, 0],
            task_indices=[0, 1],
            task_names_by_index={0: "task zero", 1: "task one"},
        )


def test_pi05_lora_training_command_freezes_vlm_and_enables_peft(tmp_path):
    workflow = _load_lora_module()
    args = Namespace(
        steps=600,
        batch_size=1,
        lora_rank=8,
        save_freq=200,
        learning_rate=1.0e-4,
        decay_learning_rate=1.0e-5,
        warmup_steps=50,
        num_workers=2,
        base_model=tmp_path / "base",
        output_dir=tmp_path / "output",
        job_name="test",
        wandb=True,
        wandb_project="test-project",
    )
    manifest = {
        "dataset_repo": workflow.DEFAULT_DATASET_REPO,
        "dataset_revision": workflow.DEFAULT_DATASET_REVISION,
        "train_episodes": [1, 2, 3],
    }
    command = workflow.build_train_command(args, manifest)
    joined = "\n".join(command)
    assert "--policy.train_expert_only=true" in command
    assert "--policy.freeze_vision_encoder=true" in command
    assert "--policy.gradient_checkpointing=true" in command
    assert "--policy.push_to_hub=false" in command
    assert "--peft.method_type=LORA" in command
    assert "--peft.r=8" in command
    assert "gemma_expert" in joined
    assert "paligemma" not in joined
    transform_arg = next(
        part for part in command if part.startswith("--dataset.image_transforms.tfs=")
    )
    assert "brightness" in transform_arg
    assert "affine" not in transform_arg


def test_pi05_qkvo_profile_changes_only_action_expert_attention(tmp_path):
    workflow = _load_lora_module()
    args = Namespace(
        steps=3000,
        batch_size=2,
        lora_rank=16,
        lora_target_profile="qkvo",
        save_freq=250,
        learning_rate=1.0e-4,
        decay_learning_rate=1.0e-5,
        warmup_steps=100,
        num_workers=2,
        base_model=tmp_path / "base",
        output_dir=tmp_path / "output",
        job_name="qkvo-test",
        wandb=False,
        wandb_project="test-project",
    )
    manifest = {
        "dataset_repo": workflow.DEFAULT_DATASET_REPO,
        "dataset_revision": workflow.DEFAULT_DATASET_REVISION,
        "train_episodes": [1, 2, 3],
    }
    command = workflow.build_train_command(args, manifest)
    target = next(part for part in command if part.startswith("--peft.target_modules="))
    assert "(q|k|v|o)_proj" in target
    assert "gemma_expert" in target
    assert "paligemma" not in target
    assert "--policy.push_to_hub=false" in command


def test_pi05_lora_training_environment_loads_video_compatibility():
    workflow = _load_lora_module()
    environment = workflow._training_environment(False)
    paths = environment["PYTHONPATH"].split(os.pathsep)
    compatibility_dir = ROOT / "examples" / "pi05_runtime_compat"
    assert str(compatibility_dir) in paths
    assert str(ROOT) in paths
    sitecustomize = compatibility_dir / "sitecustomize.py"
    assert sitecustomize.is_file()
    source = sitecustomize.read_text(encoding="utf-8")
    assert "install_video_reader_compat" in source


def test_pi05_video_compatibility_is_scoped_to_removed_videoreader():
    source = (
        ROOT / "examples" / "pi05_torchvision_video_compat.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "hasattr(torchvision.io, \"VideoReader\")" in source
    assert "av.open" in source
    assert "torchvision.io.VideoReader = PyAVVideoReaderCompat" in source
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "PyAVVideoReaderCompat"
        for node in ast.walk(tree)
    )


def test_pi05_video_compatibility_seeks_and_returns_channel_first(monkeypatch):
    stream = types.SimpleNamespace(time_base=Fraction(1, 20), thread_type=None)

    class FakeFrame:
        pts = 2
        time_base = Fraction(1, 20)

        def to_ndarray(self, *, format):
            assert format == "rgb24"
            return "rgb-array"

    class FakeContainer:
        def __init__(self):
            self.streams = types.SimpleNamespace(video=[stream])
            self.seek_call = None

        def seek(self, offset, **kwargs):
            self.seek_call = (offset, kwargs)

        def decode(self, selected_stream):
            assert selected_stream is stream
            return iter([FakeFrame()])

        def close(self):
            pass

    class FakeTensor:
        def __init__(self, value):
            self.value = value
            self.permutation = None

        def permute(self, *dimensions):
            self.permutation = dimensions
            return self

    container = FakeContainer()
    fake_io = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=lambda _: container))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(from_numpy=lambda value: FakeTensor(value)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torchvision",
        types.SimpleNamespace(io=fake_io),
    )

    path = ROOT / "examples" / "pi05_torchvision_video_compat.py"
    spec = importlib.util.spec_from_file_location("pi05_video_compat_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.install_video_reader_compat() is True
    reader = fake_io.VideoReader("episode.mp4", "video")
    assert stream.thread_type == "AUTO"
    assert reader.seek(0.1, keyframes_only=True) is reader
    assert container.seek_call == (
        2,
        {"stream": stream, "backward": True, "any_frame": False},
    )
    frames = list(reader)
    assert frames[0]["pts"] == pytest.approx(0.1)
    assert frames[0]["data"].value == "rgb-array"
    assert frames[0]["data"].permutation == (2, 0, 1)
    assert module.install_video_reader_compat() is False


def test_pi05_lora_gpu_batch_defaults_are_conservative():
    workflow = _load_lora_module()
    assert workflow.recommended_batch_size(80.0) == 4
    assert workflow.recommended_batch_size(40.0) == 4
    assert workflow.recommended_batch_size(24.0) == 2
    with pytest.raises(RuntimeError, match="L4-class"):
        workflow.recommended_batch_size(16.0)


def _fake_checkpoint(workflow, checkpoints_dir: Path, step: int) -> Path:
    checkpoint = checkpoints_dir / f"{step:06d}"
    for relative in workflow.CHECKPOINT_REQUIRED_FILES:
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "training_state/training_step.json":
            path.write_text(json.dumps({"step": step}), encoding="utf-8")
        elif path.suffix == ".json":
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_bytes(b"checkpoint")
    return checkpoint


def test_pi05_lora_checkpoints_are_validated_backed_up_and_restored(tmp_path):
    workflow = _load_lora_module()
    local = tmp_path / "local"
    backup = tmp_path / "drive"
    restored = tmp_path / "restored"
    complete = _fake_checkpoint(workflow, local, 250)
    incomplete = _fake_checkpoint(workflow, local, 500)
    (incomplete / "pretrained_model/adapter_model.safetensors").unlink()

    assert workflow.checkpoint_step(complete) == 250
    assert workflow.checkpoint_step(incomplete) is None
    assert [step for step, _ in workflow.discover_checkpoints(local)] == [250]
    assert [path.name for path in workflow.sync_checkpoints(local, backup)] == [
        "000250"
    ]
    assert [path.name for path in workflow.restore_checkpoints(backup, restored)] == [
        "000250"
    ]
    assert workflow.checkpoint_step(restored / "000250") == 250
    assert not list(backup.glob(".*.partial"))


def test_pi05_lora_resume_uses_lerobot_training_state(tmp_path):
    workflow = _load_lora_module()
    config = tmp_path / "000250/pretrained_model/train_config.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "batch_size": 4,
                "output_dir": "/content/old",
                "num_workers": 8,
                "steps": 3000,
                "save_freq": 250,
                "policy": {
                    "scheduler_decay_steps": 3000,
                    "scheduler_warmup_steps": 100,
                },
            }
        ),
        encoding="utf-8",
    )
    workflow._patch_resume_config(
        config,
        batch_size=2,
        output_dir=tmp_path / "new",
        num_workers=2,
        steps=1000,
        save_freq=200,
    )
    resumed = json.loads(config.read_text(encoding="utf-8"))
    assert resumed["batch_size"] == 2
    assert resumed["steps"] == 1000
    assert resumed["save_freq"] == 200
    assert resumed["policy"]["scheduler_decay_steps"] == 1000
    command = workflow.build_resume_command(config)
    assert "--resume=true" in command
    assert f"--config_path={config}" in command


def test_pi05_colab_has_recoverable_full_libero_training_defaults():
    notebook_source = (
        ROOT / "examples" / "pi05_parc_colab.ipynb"
    ).read_text(encoding="utf-8")
    assert 'drive.mount(\\"/content/drive\\")' in notebook_source
    assert 'TRAIN_STEPS = 3000' in notebook_source
    assert 'SAVE_FREQ = 250' in notebook_source
    assert 'LORA_RANK = 16' in notebook_source
    assert 'TRAIN_BATCH_SIZE = 0' in notebook_source
    assert 'DATASET_REPO = \\"lerobot/libero\\"' in notebook_source
    assert '\\"--backup-dir\\"' in notebook_source
    assert '\\"select\\"' in notebook_source
    assert '\\"--candidate-every\\", \\"500\\"' in notebook_source
    assert 'DRIVE_COPY' in notebook_source


def test_pi05_colab_evaluates_all_four_public_tasks():
    notebook_path = ROOT / "examples" / "pi05_parc_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    all_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "T1_TASKS.csv" in all_source
    assert "len(PUBLIC_TASK_IDS) != 4" in all_source
    assert "EVAL_EPISODES_PER_TASK" in all_source
    assert '"--tasks"' in all_source
    assert '"--timeout"' in all_source
    assert '"10"' in all_source
    assert "MUJOCO_GL" in all_source
    assert "public_eval_result_path" in all_source
    assert "collision_rate" in all_source
    assert '"--record-video"' in all_source
    assert "VIDEOS_PER_TASK = 1" in all_source
    assert "subprocess.Popen" in all_source
    assert 'print(line, end="", flush=True)' in all_source
    assert "IPython.display import Video" in all_source
    assert 'os.environ["WANDB_MODE"] = "disabled"' in all_source
    assert 'os.environ["WANDB_DISABLED"] = "true"' in all_source
    assert "REPLAN_STEPS = 10" in all_source
    assert "TEMPORAL_ENSEMBLE = False" in all_source
    assert '"PI05_TEMPORAL_ENSEMBLE"' in all_source
    assert 'ENSEMBLE_OLD_WEIGHTS = "0.25,0.15,0.05"' in all_source
    assert "EVAL_VARIANT" in all_source


def test_pi05_colab_python_cells_parse_after_removing_magics():
    notebook_path = ROOT / "examples" / "pi05_parc_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        python_source = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        ast.parse(python_source, filename=f"notebook-cell-{index}")


def test_public_evaluation_setup_is_colab_safe():
    setup_source = (ROOT / "setup.sh").read_text(encoding="utf-8")
    notebook_source = (
        ROOT / "examples" / "pi05_parc_colab.ipynb"
    ).read_text(encoding="utf-8")

    assert '"setuptools<82"' in setup_source
    assert "export MPLBACKEND=Agg" in setup_source
    assert 'setup_env[\\"MPLBACKEND\\"] = \\"Agg\\"' in notebook_source
    assert '\\"MPLBACKEND\\": \\"Agg\\"' in notebook_source


def test_public_evaluation_streams_progress_and_records_video():
    notebook_source = (
        ROOT / "examples" / "pi05_parc_colab.ipynb"
    ).read_text(encoding="utf-8")
    cli_source = (ROOT / "pipeline" / "cli.py").read_text(encoding="utf-8")
    rollout_source = (
        ROOT / "pipeline" / "rollout.py"
    ).read_text(encoding="utf-8")

    assert "evaluation_process.stdout" in notebook_source
    assert 'print(line, end=\\"\\", flush=True)' in notebook_source
    assert "evaluation_process.terminate()" in notebook_source
    assert '"--record-video"' in cli_source
    assert "EVAL_PROGRESS task=%d/%d episode=%d/%d" in rollout_source
    assert "imageio.mimwrite" in rollout_source
    assert "if record_video and not success and video_frames" in rollout_source
    assert "failure_videos_saved" in rollout_source
    assert "agentview_image" in rollout_source
    assert "robot0_eye_in_hand_image" in rollout_source
