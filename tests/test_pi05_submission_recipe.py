import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_setup_module():
    path = ROOT / "examples" / "pi05_parc_colab_setup.py"
    spec = importlib.util.spec_from_file_location("pi05_parc_colab_setup", path)
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


def test_pi05_colab_notebook_is_valid_and_builds_submission():
    notebook_path = ROOT / "examples" / "pi05_parc_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    all_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "--smoke" in all_source
    assert "--build-submission" in all_source
    assert "pi05_submission.zip" in all_source


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
    assert 'os.environ["WANDB_MODE"] = "disabled"' in all_source
    assert 'os.environ["WANDB_DISABLED"] = "true"' in all_source


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
