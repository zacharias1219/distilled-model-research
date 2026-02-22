# Architecture Documentation

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Runtime (Google Colab or local Jupyter)                            │
│                                                                     │
│  ┌─────────────────┐         ┌─────────────────────────────────┐   │
│  │ notebook.ipynb  │────────▶│ distill_app.py                  │   │
│  │ (config + flow) │         │ • prepare_system1/2_dataset      │   │
│  └─────────────────┘         │ • distill_system1 / distill_system2 │   │
│           │                  │ • compare_models, load_*, infer   │   │
│           │                  └──────────────┬────────────────────┘   │
│           │                                 │                        │
│           │                                 ▼                        │
│           │                  ┌─────────────────────────────────┐   │
│           │                  │ subprocess: easydistill --config │   │
│           │                  └──────────────┬────────────────────┘   │
│           ▼                                 ▼                        │
│  ┌─────────────────┐         ┌─────────────────────────────────┐   │
│  │ data/*.json      │         │ configs/*.json                   │   │
│  │ (train_labeled,  │         │ checkpoint dirs                  │   │
│  │  omnithought_cot)│         │ (distilled-qwen2.5-0.5b, etc.)   │   │
│  └────────┬────────┘         └─────────────────────────────────┘   │
│            │                                 │                        │
└────────────┼─────────────────────────────────┼────────────────────────┘
             │                                 │
             ▼                                 ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  Hugging Face (datasets + models)                                 │
  │  DistilQwen_100k, OmniThought, Qwen2.5-7B/0.5B/1.5B-Instruct      │
  └──────────────────────────────────────────────────────────────────┘
```

There is **no browser frontend**, **no HTTP backend**, and **no database**. The “UI” is the notebook; the “backend” is `distill_app.py` plus the EasyDistill CLI.

---

## Components

### Notebook (orchestration)

- **Technology:** Jupyter (`.ipynb`), run in Colab or local.
- **Purpose:** Single entry point: set config variables, run data prep, trigger distillation, run comparison. No business logic; only imports and calls into `distill_app`.
- **Key file:** `notebook.ipynb` — sections: Setup, System 1 (config → prepare → run), System 2 (config → prepare → run), Teacher vs Student Comparison, optional single-prompt inference.

### distill_app (logic)

- **Technology:** Python 3.x, `torch`, `transformers`, `datasets`, `bitsandbytes`, stdlib `json`/`pathlib`/`subprocess`.
- **Purpose:** Data loading and mapping, optional teacher labeling, EasyDistill config generation, subprocess invocation, model loading, inference, and comparison.
- **Key file:** `distill_app.py`
  - **Data:** `prepare_system1_dataset`, `prepare_system2_dataset`, `save_json`/`read_json`.
  - **Training:** `write_system1_config`, `write_system2_config`, `run_easy_distill`, `distill_system1`, `distill_system2`.
  - **Inference:** `load_teacher`, `load_student`, `format_prompt`, `label_with_teacher`, `infer_student`, `compare_models`.

### Model layer (training)

- **Technology:** EasyDistill (CLI), Hugging Face Transformers.
- **Purpose:** Run KD training (black-box for System 1, SFT-style for System 2) and produce checkpoint directories.
- **Models:** Teacher `Qwen/Qwen2.5-7B-Instruct` (8-bit in app); students e.g. `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`. Inference uses the same Transformers stack.

### Data layer

- **Technology:** File system only. No SQL/NoSQL.
- **Purpose:** Persist training data (JSON) and model checkpoints (directories).
- **Paths:**
  - `data/train_instructions.json`, `data/train_labeled.json` — System 1.
  - `data/omnithought_cot.json` — System 2.
  - `configs/*.json` — Generated EasyDistill configs.
  - `./distilled-qwen2.5-0.5b`, `./distilled-qwen2.5-1.5b-cot` — Checkpoint dirs (configurable).

---

## Data Flow

1. **Config** — User sets variables in notebook cells (model names, dataset slice, epochs, paths).
2. **Prepare data** — Notebook calls `prepare_system1_dataset` and/or `prepare_system2_dataset`; they load HF datasets, map to `{instruction, input, output}`, optionally relabel with teacher, and write JSON to `data/`.
3. **Distillation** — Notebook builds a config dict and calls `distill_system1(config)` or `distill_system2(config)`. These validate (student model, non-empty data), write EasyDistill config to `configs/`, then run `easydistill --config <path>`. EasyDistill trains and writes the checkpoint directory.
4. **Comparison** — User sets `COMPARE_PROMPTS` and calls `compare_models(prompts, ...)`. The function loads teacher and students (or uses pre-loaded instances), runs each prompt through the appropriate model, truncates, and prints Prompt | Teacher | System 1 | System 2. Missing checkpoint → "Model not found at <path>" and that column is skipped.

---

## Technology Decisions

| Decision | Rationale |
|----------|-----------|
| Jupyter notebook as only “UI” | Single researcher, no need for web UI; notebook is reproducible and self-documenting. |
| No backend server | Scope is Colab/local experimentation; avoids deployment, auth, and scaling. |
| No database | All state is files and checkpoints; sufficient for one user and reproducible runs. |
| EasyDistill via subprocess | Library is CLI-driven; subprocess keeps compatibility and isolates GPU process. |
| Teacher in 8-bit | Fits 7B on Colab GPUs (e.g. T4); trade-off is slightly slower inference when relabeling. |
| Config as dict from notebook | Single object for all knobs; easy to override without changing function signatures. |
| Hugging Face datasets | Standard, no custom ETL; DistilQwen_100k and OmniThought match the PRD. |

---

## Scalability Considerations

- **Current:** Single user, single notebook, one GPU. No concurrency or request/sec targets.
- **10x data (e.g. 20k samples):** Same process; longer training and more disk. If OOM, reduce batch size or increase gradient accumulation in the generated config.
- **Multi-user / multi-run:** Not in scope. Would require job queue, shared storage, and possibly separate workers; not implemented in v1.
