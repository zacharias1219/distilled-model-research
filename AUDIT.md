# Notebook audit: gaps, fixes, run order

Quick reference so you can complete the app as fast as possible.

---

## Run order (critical)

1. **Runtime setup** — In Colab: Runtime → Change runtime type → **GPU** (e.g. T4).
2. **TPU (optional)** — Only if you chose TPU runtime; else skip.
3. **Install dependencies** — pip torch, transformers, datasets, etc.
4. **Install EasyDistill** — Clone + `pip install -e easydistill`.
5. **Clone repo (if needed)** — Only if you opened from Drive/upload and get FileNotFoundError for `distill_app.py`.
6. **HF token (optional)** — Colab: set `HF_TOKEN` from Secrets; local: `.env` is loaded by `distill_app`.
7. **Project root and imports** — Sets `ROOT`, `os.chdir(ROOT)`, imports `distill_app`. **Must run before any Prepare / Run / Test / Comparison cells.**
8. **System 1:** Config → Prepare Data → Run Distillation → Test System 1.
9. **System 2:** Config → Prepare CoT Data → Run CoT Distillation → Test System 2.
10. **Comparison** — Teacher vs System 1 vs System 2 (needs GPU for teacher 8-bit; checkpoints use ROOT paths).

---

## What was lacking (from saved outputs)

| Issue | Cause | Fix applied |
|-------|--------|-------------|
| Test System 1: "Checkpoint not found" after successful distillation | Test cell used `./distilled-qwen2.5-0.5b`; cwd may not be project root (e.g. Colab). | Test cell now uses `ROOT / "distilled-qwen2.5-0.5b"` (fallback `Path.cwd()` if `ROOT` undefined). |
| Test System 2: same | Same relative path vs cwd. | Same: `ROOT / "distilled-qwen2.5-1.5b-cot"`. |
| Comparison: "Model not found" for both students | `compare_models(..., system1_path="./...", system2_path="./...")` relative to cwd. | Comparison cell now builds paths with `ROOT` and passes them to `compare_models`. |
| OmniThought: `trust_remote_code is not supported anymore` | Newer `datasets` deprecation/removal of script execution. | `prepare_system2_dataset` tries load **without** `trust_remote_code` first (Parquet); fallback with `trust_remote_code` only if error mentions custom code. |
| Duplicate "Clone repo" section | Two markdown cells with same heading. | Removed the duplicate markdown cell. |

---

## What you still need to do

- **Run with GPU:** Current saved outputs show `nvidia-smi` not found and bitsandbytes "None of the available devices...". Use Colab GPU (or local CUDA) so teacher loads and training runs.
- **Run cells in order:** Especially: Project root and imports → then Prepare → Run → Test for each system. Re-run "Project root and imports" after cloning the repo.
- **System 2:** If OmniThought still errors (e.g. dataset uses script and your `datasets` version forbids it), pin an older `datasets` or use a Parquet-only fork if available.
- **PRD ACs:** To close PRD acceptance criteria: run full System 1 + System 2 pipelines, then comparison, and confirm checkpoint dirs exist and outputs are non-empty and on-topic.

---

## Optional improvements (speed / robustness)

- Have `distill_system1` / `distill_system2` **return the resolved absolute path** and use that in Test/Comparison (e.g. `path_sys1` from Run cell) so you don’t depend on `ROOT` being in scope.
- Add a one-line reminder at the top of the notebook: *"Run 'Project root and imports' first, then run cells in order."*
- If you run comparison without GPU: teacher load fails (bitsandbytes); you could skip teacher and only compare System 1 vs System 2 when no CUDA is available.
