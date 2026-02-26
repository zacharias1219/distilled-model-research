# Qwen Distillation Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/distilled-model-research/blob/main/notebook.ipynb)

A Colab-based research lab to distill **Qwen2.5-7B-Instruct** into smaller, faster student models: one for instruction-following (System 1) and one for step-by-step reasoning (System 2). No backend or database—just a Jupyter notebook and a Python helper module. Get started in under 5 minutes and run full distillation + comparison in about 2 hours on a single GPU.

## What is this?

You get a **teacher** (7B Qwen) and train two **students** (e.g. 0.5B and 1.5B): one that answers quickly and one that shows chain-of-thought. The project uses Hugging Face datasets (DistilQwen_100k, OmniThought) and EasyDistill for training. Everything runs in Google Colab or locally with one notebook and `distill_app.py`.

## Features

- **System 1 distillation** — Black-box KD from 7B teacher to a small instruction-following student (e.g. 0.5B).
- **System 2 distillation** — CoT student trained on OmniThought with optional RV/CD quality filtering.
- **Teacher vs student comparison** — Side-by-side outputs for custom prompts; handles missing checkpoints cleanly.
- **Colab-first** — No server or DB; clone, open notebook, run cells. Optional local run with same code.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/distilled-model-research.git
cd distilled-model-research

# Install (local only; Colab uses notebook pip cell)
pip install -r requirements.txt

# Run: open the notebook
# Colab: Upload notebook.ipynb or use "Open in Colab" badge, set runtime to GPU, run all cells.
# Local:  jupyter notebook notebook.ipynb   # or VS Code / Cursor
# Visit:  Run cells in order (Setup → System 1 → System 2 → Comparison)
```

**Under 5 minutes to start:** Open `notebook.ipynb` in Colab → Runtime → Change runtime type → **GPU** (or **TPU** for faster training) → Run the setup cells. Then run System 1 config + Prepare Data + Run Distillation when ready.

**TPU:** If you select a TPU runtime, run the **TPU setup** cell after the dependency install. Teacher labeling and System 1 training then run on TPU (no vllm; we use Hugging Face generate + in-notebook TPU training). System 2 (CoT) still uses the GPU-oriented EasyDistill path.

## Hugging Face token (optional but recommended)

The repo does **not** include `.env` or any secrets (they are in `.gitignore`). After you clone or open the notebook from GitHub:

- **Colab:** Add your Hugging Face token once: click the key icon in the left sidebar → Add new secret → name `HF_TOKEN`, value your token → run the notebook’s “HF token (optional)” cell so the token is used for dataset/model downloads and higher rate limits.
- **Local:** Create a `.env` file in the repo root with one line: `HF_TOKEN=hf_...`. It is loaded automatically when you `import distill_app`.

Without a token, downloads still work but you may see rate-limit warnings.

## EasyDistill setup

The notebook clones [EasyDistill](https://github.com/modelscope/easydistill) and installs it so we can run KD training. Two things matter:

1. **Config path** — We pass an **absolute** path to `easydistill --config` so the CLI loads our project’s JSON (EasyDistill otherwise resolves relative paths against its package dir).
2. **Template path** — The chat template lives in the EasyDistill repo at `configs/chat_template/chat_template_kd.jinja`. After cloning, the notebook sets `template_path` in the config to that file’s absolute path (Colab: `/content/easydistill/...`, local: `./easydistill/...`). System 1 and System 2 both use it for formatting.

System 1 uses `kd_black_box_local` (teacher inference → then train). System 2 uses `kd_black_box_train_only` (train on pre-built CoT data only).

## Tech Stack

- **Interface:** Jupyter notebook (no separate frontend).
- **Backend:** None (logic in `distill_app.py`; EasyDistill runs as CLI subprocess).
- **ML:** Qwen2.5-7B-Instruct (teacher, 8-bit), Qwen2.5-0.5B/1.5B-Instruct (students), EasyDistill (KD training), Hugging Face `datasets` + `transformers`.
- **Data:** File-based (JSON under `data/`, checkpoints in project dir); no database.

## Demo

- **Live:** Run the notebook in [Google Colab](https://colab.research.google.com) (use the badge and replace `YOUR_USERNAME` with your GitHub username, or upload `notebook.ipynb` manually).
- **Artifacts:** After running, you get checkpoint dirs (e.g. `./distilled-qwen2.5-0.5b`, `./distilled-qwen2.5-1.5b-cot`) and the comparison cell prints teacher vs student outputs.

## Documentation

- [API](API.md) — Public interface of `distill_app.py` (functions, configs, contracts).
- [Architecture](ARCHITECTURE.md) — System overview, components, data flow, technology decisions.
- [Deployment](DEPLOYMENT.md) — Colab vs local, environment, running and rollback.
- [Contributing](CONTRIBUTING.md) — How to contribute, code style, commits, PRs.
- [Deep dive](DEEP_DIVE.md) — Four levels of explanation (executive → architect).

## License

[MIT](LICENSE).
