# pi0.5 RTC final submission reproduction

This reproduces the final PARC Track 1 candidate from an already merged q/k/v/o
LoRA checkpoint at training step 2500. Run it in the same Colab runtime used by
`pi05_checkpoint_sweep_colab.ipynb`, after `MODEL_2500_DIR`, `PI05_PYTHON`, and
`EVAL_PYTHON` have been defined.

The fixed inference settings are:

- replan steps: 10
- flow inference steps: 8
- RTC: enabled
- RTC execution horizon: 10
- RTC maximum guidance weight: 5.0
- RTC prefix schedule: `EXP`
- RTC inference delay: 0
- temporal ensembling: disabled

The five-episode public evaluation preserved 95% success and 5% collision while
reducing mean RMS jerk from 5.787 to 4.677. At the replan boundary, phase-0 jerk
fell from 9.204 to 3.687 and mean action jump fell from 0.195 to 0.071.

```python
from pathlib import Path
import subprocess

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/PARC2026/"
    "pi05_action_expert_lora_qkvo_batch4"
)

command = [
    str(PI05_PYTHON),
    "-u",
    "examples/pi05_finalize_qkvo_colab.py",
    "--repo-root", str(REPO_DIR),
    "--policy-python", str(PI05_PYTHON),
    "--eval-python", str(EVAL_PYTHON),
    "--runtime-root", "/content/pi05_runtime",
    "--model-source", str(MODEL_2500_DIR),
    "--submission-output",
    "/content/pi05_submission_rtc_step2500_h10_g5.zip",
    "--drive-root", str(DRIVE_ROOT),
    "--checkpoint-step", "2500",
    "--episodes", "5",
    "--max-steps", "600",
    "--seed", "42",
    "--replan-steps", "10",
    "--inference-steps", "8",
    "--rtc",
    "--rtc-execution-horizon", "10",
    "--rtc-max-guidance-weight", "5.0",
    "--rtc-schedule", "EXP",
    "--rtc-inference-delay", "0",
    "--record-video",
    "--save-trajectories",
    "--train-steps", "3000",
    "--batch-size", "4",
    "--lora-rank", "16",
]
subprocess.run(command, cwd=REPO_DIR, check=True)
```

The command builds and statically validates the root-level ZIP, checks the
embedded step-2500 merge manifest and RTC defaults, evaluates the exact activated
model, verifies the copied ZIP by SHA-256, and writes
`final/final_reproducibility_manifest_rtc.json` on Drive. A `.partial` file is
not a completed submission.
