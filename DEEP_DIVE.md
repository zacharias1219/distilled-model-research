# Qwen Distillation Lab — Explained at 4 Depths

---

## LEVEL 1: EXECUTIVE (Explain to a 5-year-old)

**Problem:** Big AI brains are slow and need lots of power. We want smaller brains that still answer well but run faster and cheaper.

**Users:** Students and researchers who want to try “shrinking” a big Qwen model into smaller ones, and engineers who need faster, cheaper models for prototypes.

**Magic:** It turns one big, slow teacher model into two smaller, faster student models: one that follows instructions quickly, and one that shows its reasoning step-by-step.

**Example:** “It’s like a famous chef (the teacher) teaching two assistants (the students): one to cook simple dishes fast, and one to explain the recipe step-by-step so others can follow.”

---

## LEVEL 2: JUNIOR DEVELOPER (Onboarding)

**Architecture overview**

- **Frontend:** None. A Jupyter notebook (`notebook.ipynb`) is the “UI”: you run cells to set config, prepare data, run distillation, and compare models.
- **Backend:** None. Logic lives in Python: `distill_app.py` (helpers) and the notebook (orchestration). EasyDistill is run as a CLI subprocess.
- **Data:** No database. Data lives on disk: Hugging Face datasets (DistilQwen_100k, OmniThought), JSON files under `data/` (e.g. `train_labeled.json`, `omnithought_cot.json`), and model checkpoints in folders like `./distilled-qwen2.5-0.5b`.
- **ML models:** Teacher = `Qwen/Qwen2.5-7B-Instruct` (8-bit). Students = e.g. `Qwen/Qwen2.5-0.5B-Instruct` (System 1), `Qwen/Qwen2.5-1.5B-Instruct` (System 2). They don’t “predict” in the classic sense; distillation trains the students to mimic the teacher’s behavior.

**File structure**

- **Start here:** `notebook.ipynb` (run top to bottom: Setup → System 1 → System 2 → Comparison).
- **Key files:**
  - `distill_app.py` → All core logic: data prep, config writing, EasyDistill runner, `distill_system1`, `distill_system2`, `compare_models`, inference helpers.
  - `notebook.ipynb` → Config variables and cell-by-cell workflow; no separate “components” or “models” folders.
- **Data:** `data/` (created at runtime): `train_instructions.json`, `train_labeled.json`, `omnithought_cot.json`. `configs/` holds generated EasyDistill JSON configs. Checkpoints sit in project root (e.g. `./distilled-qwen2.5-0.5b`).

**Data flow**

1. **Config** → You set variables in notebook cells (model names, dataset slice, epochs).
2. **Prepare data** → `prepare_system1_dataset` / `prepare_system2_dataset` load HF datasets, map to `{instruction, input, output}`, optionally relabel with teacher, and save JSON to `data/`.
3. **Distillation** → `distill_system1(config)` / `distill_system2(config)` validate inputs, write EasyDistill config to `configs/`, run `easydistill --config <path>`, which trains the student and writes a checkpoint directory.
4. **Comparison** → `compare_models(prompts, ...)` loads teacher and student(s) (or uses pre-loaded), runs each prompt, prints Prompt / Teacher / System 1 / System 2 side-by-side.

**Common tasks**

- **Change dataset size** → Edit `DATASET_SLICE_SYS1` or `DATASET_SLICE_SYS2` in the notebook config cells.
- **Point to a different student** → Change `STUDENT_MODEL_SYS1` / `STUDENT_MODEL_SYS2` or the `out_dir` in the config dict passed to `distill_system1` / `distill_system2`.
- **Add comparison prompts** → Append to `COMPARE_PROMPTS` in the comparison cell and re-run `compare_models`.
- **Fix data prep / edge cases** → Edit `prepare_system1_dataset` or `prepare_system2_dataset` in `distill_app.py`.
- **Change training hyperparameters** → Edit `write_system1_config` or `write_system2_config` in `distill_app.py` (epochs, batch size, LR, etc.).

---

## LEVEL 3: SENIOR ENGINEER (Technical Deep-Dive)

**Design patterns**

- **Script + notebook split:** All reusable logic lives in `distill_app.py`; the notebook only holds config and orchestration. Keeps the notebook readable and logic testable/importable.
- **Config-as-dict:** `distill_system1(config)` and `distill_system2(config)` take a single dict (model names, paths, epochs). Easy to override from the notebook without changing function signatures.
- **Fail-fast with clear messages:** Empty data → “No data for System 1 distillation” and no EasyDistill run. Invalid student model → “Invalid student model: <name>” + example. Missing checkpoint in `compare_models` → “Model not found at <path>” and skip that column.

**Architecture decisions**

- **No backend/DB:** Chosen for single-user, research-in-Colab use. Avoids deployment, auth, and persistence complexity; all state is files and checkpoints.
- **EasyDistill via subprocess:** Training is delegated to the EasyDistill CLI instead of in-process Python API to match the library’s supported usage and to isolate GPU/memory in a separate process.
- **Teacher in 8-bit:** `load_in_8bit=True` reduces VRAM so the 7B teacher fits on Colab GPUs (e.g. T4). Trade-off: slightly slower inference when used for optional relabeling.

**Performance considerations**

- **Teacher loading:** One-time 8-bit load; optional relabeling is the main cost if enabled.
- **Distillation:** Dominated by EasyDistill (GPU training). Targets in PRD: ~1 hr for System 1 (1k samples, 1 epoch), ~2 hr for System 2 (2k samples).
- **Comparison:** Loads up to three models (teacher + Sys1 + Sys2). For many prompts, pre-load once and pass `teacher_model`/`sys1_model`/`sys2_model` into `compare_models` to avoid reloading.

**Security / safety**

- **No auth:** Notebook is assumed private (e.g. your Colab). No API keys required for core flow; HF token only if using gated models/datasets.
- **Inputs:** Only HF dataset names and local paths are used in config; no user-generated text is persisted beyond what you put in `COMPARE_PROMPTS`. No PII in the PRD scope.
- **Subprocess:** `run_easy_distill` runs `easydistill --config <path>`; config path is built in code, not from user string, limiting injection risk.

**Testing**

- No formal test suite in the repo. Validation is manual: run the notebook end-to-end and inspect outputs. Edge behavior is encoded in the helper (empty data, invalid model, missing checkpoint).
- For future tests: unit tests on `format_prompt`, `prepare_*` output shape, and `compare_models` with mocks; integration test with tiny data slice and stub EasyDistill.

**Deployment**

- **Colab:** Upload or clone repo, open `notebook.ipynb`, set runtime to GPU, run cells. “Deploy” = run the notebook; artifacts are Colab disk and optional Drive.
- **Local:** `pip install -r requirements.txt`, run the same notebook or call `distill_app` from a script. Checkpoints and `data/` are local directories.

---

## LEVEL 4: ARCHITECT (System Design Analysis)

**Scalability analysis**

- **Current:** Single-user, single-notebook, one GPU. No concurrency; no request/sec target. “Scale” here = dataset size and model size within one machine.
- **10x data (e.g. 20k CoT samples):** Same process; longer training time and more disk for JSON/checkpoints. Possible OOM → reduce batch size or gradient accumulation in config.
- **Multi-user / multi-run:** Not designed. Would require job queue, shared storage for checkpoints, and possibly separate training workers. Out of scope for v1.

**Failure modes**

1. **EasyDistill fails (OOM, crash):** No automatic retry. User sees non-zero return code and message in notebook. Mitigation: smaller student, smaller batch, or 8-bit teacher already in use.
2. **HF dataset/model unavailable:** `load_dataset` or `from_pretrained` can fail. Mitigation: cache to Drive or local after first run; keep small JSON slices for dev.
3. **Colab timeout / disconnect:** Session loss; no built-in resume. Mitigation: save checkpoints frequently (`save_steps` in config) and re-run from last checkpoint if the tool supports it.

**Cost structure**

- **Compute:** Colab GPU (free tier or Pro). No separate server cost. Training cost = Colab runtime time.
- **Storage:** Colab disk (ephemeral) or Google Drive for checkpoints and data. No dedicated DB or object-store cost.
- **Scaling:** Linear in human time (more runs / bigger slices = more Colab usage). No infra scaling in v1.

**Alternatives considered**

- **In-process training API:** Avoid subprocess and config files. Rejected: EasyDistill’s primary interface is CLI + config; wrapping it keeps compatibility and isolates GPU process.
- **Web UI (e.g. Gradio):** Rejected per PRD: “No web frontend, backend API, authentication.”
- **Database for prompts/results:** Rejected: single-user research; JSON and notebook outputs are enough.

**Technical debt**

1. **Hardcoded paths in EasyDistill config:** e.g. `instruction_path: "data/train_instructions.json"`. Fine for single-repo layout; breaks if notebook is run from another cwd. Priority: Low. Fix: resolve paths relative to config file or a project root.
2. **No checkpoint resume:** If EasyDistill added resume, we don’t expose it. Priority: Low until long runs are common.
3. **Schema coupling to OmniThought/DistilQwen:** Field names in `prepare_system2_dataset` and dataset iteration are tied to HF dataset schemas. Priority: Med. Fix: small adapter layer or configurable field mapping.

**Future refactoring**

- **v2 (if moving off “notebook-only”):** Optional thin API (e.g. FastAPI) that runs `distill_system1`/`distill_system2` as jobs and serves `compare_models` as an endpoint, still using local/Drive storage. Reason: if multiple people need to trigger runs or view comparisons without opening the notebook.
- **v2 (if scaling data):** Replace single JSON files with chunked or streaming datasets and configurable EasyDistill data paths to avoid loading everything into memory.

---

## VISUAL DIAGRAMS (text-based)

**System architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│  Google Colab (or local Jupyter)                                │
│  ┌─────────────────┐     ┌──────────────────────────────────┐ │
│  │ notebook.ipynb   │────▶│ distill_app.py                    │ │
│  │ (config + cells) │     │ - prepare_system1_dataset         │ │
│  └─────────────────┘     │ - prepare_system2_dataset         │ │
│           │               │ - distill_system1 / distill_system2│ │
│           │               │ - compare_models, load_* , infer  │ │
│           │               └──────────────┬─────────────────────┘ │
│           │                              │                       │
│           │                              ▼                       │
│           │               ┌──────────────────────────────────┐  │
│           │               │ subprocess: easydistill --config   │  │
│           │               └──────────────┬─────────────────────┘  │
│           ▼                              ▼                       │
│  ┌─────────────────┐     ┌──────────────────────────────────┐  │
│  │ data/*.json      │     │ configs/*.json                    │  │
│  │ (train_labeled, │     │ distilled-qwen2.5-0.5b (dir)       │  │
│  │  omnithought_cot)│     │ distilled-qwen2.5-1.5b-cot (dir)   │  │
│  └─────────────────┘     └──────────────────────────────────┘  │
│           │                              │                       │
│           ▼                              ▼                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Hugging Face (datasets + models): DistilQwen_100k, OmniThought,│
│  │ Qwen2.5-7B-Instruct, Qwen2.5-0.5B/1.5B-Instruct              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Data flow (System 1)**

```
DistilQwen_100k (HF) ──slice──▶ [{instruction, input, output}]
                                      │
                    optional relabel with teacher (8-bit)
                                      │
                                      ▼
                    data/train_instructions.json, data/train_labeled.json
                                      │
                                      ▼
                    write_system1_config ──▶ configs/kd_black_box_*.json
                                      │
                                      ▼
                    easydistill --config ... ──▶ ./distilled-qwen2.5-0.5b/
```

**Data flow (System 2)**

```
OmniThought (HF) ──slice + RV/CD filter──▶ [{instruction, input, output=cot}]
                                      │
                                      ▼
                    data/omnithought_cot.json
                                      │
                                      ▼
                    write_system2_config ──▶ configs/kd_cot_*.json
                                      │
                                      ▼
                    easydistill --config ... ──▶ ./distilled-qwen2.5-1.5b-cot/
```

**Comparison flow**

```
COMPARE_PROMPTS ──▶ compare_models(...)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    load_teacher    load_student     load_student
    (7B, 8-bit)     (Sys1 path)      (Sys2 path)
          │               │               │
          └───────────────┼───────────────┘
                          ▼
              For each prompt: format_prompt → generate → truncate
                          │
                          ▼
              Print: [Prompt] | Teacher | System 1 | System 2
```

---

## NAVIGATION GUIDE

**“I want to run distillation end-to-end.”**  
→ Open `notebook.ipynb`. Run: Setup → System 1 config + Prepare Data + Run Distillation → System 2 config + Prepare CoT Data + Run CoT Distillation → Comparison.

**“I want to change training data or size.”**  
→ Notebook: edit `DATASET_SLICE_SYS1`, `DATASET_SLICE_SYS2`, or `RV_MIN`/`CD_MIN`. Re-run the corresponding “Prepare” and “Run” cells.

**“I want to fix ‘No data’ or ‘Invalid student model’.”**  
→ `distill_app.py`: `distill_system1` / `distill_system2` (validation and messages). Data emptiness comes from `prepare_system1_dataset` / `prepare_system2_dataset` (they print and return without saving when empty).

**“I want to add or change comparison prompts.”**  
→ Notebook: edit `COMPARE_PROMPTS` in the comparison cell. For different model paths, pass `teacher_path`, `system1_path`, `system2_path` into `compare_models`.

**“I want to change how prompts are formatted (e.g. for CoT).”**  
→ `distill_app.py`: `format_prompt(..., mode="system1"|"system2")`. Used by `label_with_teacher`, `infer_student`, and the teacher branch inside `compare_models`.

**“Core business logic lives in…”**  
→ `distill_app.py`: data prep (`prepare_system1_dataset`, `prepare_system2_dataset`), config generation (`write_system1_config`, `write_system2_config`), entry points (`distill_system1`, `distill_system2`), and comparison + inference (`compare_models`, `infer_student`, `load_teacher`, `load_student`). The notebook only wires config and calls these.

**“EasyDistill fails or OOMs.”**  
→ Check `configs/*.json` (written by `write_system1_config` / `write_system2_config`). Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`, or use a smaller student model. Teacher is already 8-bit in `load_teacher`.
