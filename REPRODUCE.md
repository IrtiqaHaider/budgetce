# Reproduction and evidence

Author: **Irtiqa Haider**. Repository: `https://github.com/IrtiqaHaider/budgetce`.

## Included measurements

`data/initial-study.zip` retains all 128 original raw blocks (20 calibration, 90 evaluation, 18 training), the numerical gate, frozen decisions, configuration, recorded implementation, and tables required for arithmetic cross-checks. Retained members are byte-identical to the submitted archive. Redundant figures, LaTeX/preview reports, setup logs, and duplicated documentation were removed. The subset has its own hash manifest and parent-archive SHA-256; it is not represented as the original ZIP.

`data/matched-confirmation.zip` is the unmodified nine-trial follow-up archive. It includes 900 measured and 180 warmup optimizer updates, two additional numerical preflights, per-trial loss trajectories, the exact runner, and source identity. See `data/provenance.json` for original and packaged archive hashes.

The confirmation was selected after the original results were examined. Do not pool its 100-update windows with the original 10-update windows. The selected CUDA plan is fixed, not credited to the cost model.

Run `python scripts/reproduce.py --out reanalysis` to reconstruct every published CSV. It checks **37 initial-study and 27 confirmation invariants**. Successful integrity checks do not prove that the physical GPU ran exactly as reported; they establish consistency of the submitted records. The initial GPU gate has 21 recorded checks; the confirmation adds two shape-matched gradient checks.

## CPU tests

In a clean CPU environment, install PyTorch from the official CPU wheel index, then the supporting requirements:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
```

The CPU test environment is separate from the recorded GPU environment. CI validates CPU code and recorded arithmetic; it does not compile or benchmark the CUDA extension. Package-generation checks and their environment are recorded in `results/package_validation.json`.

## Fresh GPU reproduction

Use a single NVIDIA GPU with compute capability at least 7.5 and a compatible CUDA compiler. Only the T4 configuration has recorded performance evidence. Do not install the CPU wheel above in a GPU runtime.

```bash
python -m pip install -r requirements.txt
# Preserve the notebook's CUDA-enabled torch. The original configuration uses FP16 compute.
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MAX_JOBS=1
OUT="$PWD/runs/confirmation_001"
mkdir -p "$OUT/validation"
python -m pytest -q tests/test_artifact.py tests/test_dispatch.py tests/test_model.py tests/test_ops.py tests/test_planner.py > "$OUT/validation/cpu-tests.txt"
python -m budgetce.cli init      --out "$OUT" --config configs/confirm.json
python -m budgetce.cli validate  --out "$OUT"
python -m budgetce.cli operators --out "$OUT"
python -m budgetce.cli train     --out "$OUT"
python -m budgetce.cli analyze   --out "$OUT"
python -m budgetce.cli export    --out "$OUT"
python scripts/confirm9.py --prior "$OUT"
```

Run commands only after the preceding command succeeds. Shell users can start with `set -e` to stop on failure. The notebook uses checked subprocess calls. The final command is the exact nine-trial runner captured in the uploaded archive. It verifies source identity and the original GPU gate, then runs two new shape-matched numerical checks. Native, PyTorch:1024 and CUDA:1024 are randomized within each repeat.

The original engine hash is `858c1389d6b334900d7ed06a232e0b4d4a5b94993bb64fa66583638246b4f8e0`. The measured runner hash is `645ea36aa3399550fd69ebbf56af52e39781c5d69f3a469b94ee947ff46547f3`. Files under `budgetce/` and `scripts/confirm9.py` are unchanged. Editing them requires a new source identity and, for the hard-locked confirmation helper, deliberately updating its guard; never bypass a correctness failure.

## Measurement definitions

- Operator time: synchronized wall time for projection + cross-entropy + backward, divided by measured calls; no optimizer inside these blocks.
- Training throughput: sum of measured token counts divided by sum of measured trial wall times, not arithmetic averaging of per-trial rates.
- Peak allocated memory: maximum PyTorch live-tensor allocation after warmup; extra operator memory subtracts the resident-input baseline. Reserved memory and before/after NVML snapshots are recorded separately.
- Repeat variation: min–max and sample coefficient of variation over three trial means. These are not confidence intervals. Warmup, validation, compilation and input preparation are outside the measured window.
- Correctness: declared first-order relative-error tolerances and finite updates. All-ignore means intentionally return zero; higher-order gradients, label smoothing and class weights are outside the operator contract.

## Release and report

`SHA256SUMS.json` covers the packaged release. Reproduce into `reanalysis/` to avoid replacing the distributed tables and figures. `verify_release.py` reports deliberate edits as modifications. The compiled PDF contains the full report and author metadata; no LaTeX build files are included. Report equations and tables can be checked against the retained code and evidence.

GPU UUIDs and generic runtime paths remain in the evidence to support environment comparisons; no model weights, user datasets, credentials, or API keys are included.
