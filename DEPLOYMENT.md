# Deployment Guide

This project has no traditional “deploy to production” step. “Deployment” means **running the notebook** in Google Colab or on a local machine with GPU. There is no web server, no container orchestration, and no CI/CD pipeline by default.

---

## Prerequisites

- **Python:** 3.9+ (3.10+ recommended).
- **GPU:** Recommended for distillation and comparison (Colab GPU or local CUDA-capable GPU).
- **For Colab:** Google account; optionally Colab Pro for longer/larger sessions.
- **For local:** `pip` or `conda`; optionally Jupyter or VS Code/Cursor with notebook support.

---

## Environment Variables

No required env vars for the core flow. Optional:

```bash
# Optional: for gated Hugging Face models/datasets
export HF_TOKEN=your_huggingface_token

# Optional: cache directory for Hugging Face (default ~/.cache/huggingface)
export HF_HOME=/path/to/cache
```

There is no `.env.example` in the repo; no database URL or API keys are used.

---

## Local “Deployment” (run on your machine)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/distilled-model-research.git
cd distilled-model-research
pip install -r requirements.txt
```

### Run the notebook

```bash
# Option A: Jupyter
jupyter notebook notebook.ipynb

# Option B: VS Code / Cursor — open notebook.ipynb and run cells

# Option C: headless (run all cells)
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook-executed.ipynb
```

Ensure `distill_app.py` is in the same directory as the notebook (or on `sys.path`). Use a GPU runtime if available so distillation and comparison run in reasonable time.

### “Build” (not applicable)

There is no `npm run build` or artifact to compile. The only “build” is installing dependencies and having the checkpoint dirs after training.

---

## Colab “Deployment”

1. **Open the notebook**
   - Use the [Open in Colab](https://colab.research.google.com) badge from the README (replace `YOUR_USERNAME` in the URL), or  
   - Upload `notebook.ipynb` to Colab (File → Upload notebook).

2. **Set GPU runtime**
   - Runtime → Change runtime type → Hardware accelerator: **GPU** (e.g. T4) → Save.

3. **Run cells in order**
   - Run the first two setup cells (pip install, imports).
   - Run System 1: config cell → Prepare Data & Label → Run Distillation.
   - Run System 2: config cell → Prepare CoT Data → Run CoT Distillation.
   - Run Comparison: set `COMPARE_PROMPTS`, run `compare_models(...)`.

4. **Optional: persist checkpoints**
   - Mount Google Drive and copy `data/` and checkpoint dirs to Drive so they survive session end.

No separate “production” URL; the “deployment” is your Colab session.

---

## Docker (optional)

The repo does not ship a Dockerfile. If you add one, a minimal pattern would be:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY distill_app.py notebook.ipynb ./
# No CMD: user runs jupyter or nbconvert manually with GPU access
```

For GPU support you’d use an image with CUDA and pass `--gpus all`. This is optional and not required for Colab or local run.

---

## Post–“Deployment” Checks

- **Colab:** After running the comparison cell, confirm that Teacher, System 1, and System 2 columns print (or “Model not found” for missing checkpoints).
- **Local:** Same; optionally run a minimal test:  
  `python -c "from distill_app import format_prompt; print(format_prompt('Hi', mode='system1')[:50])"`

There is no `/health` endpoint; “health” is that the notebook runs and the helper module imports.

---

## Rollback

- **Code:** Revert with Git: `git checkout <previous-commit>` or `git revert HEAD` and re-run the notebook.
- **Checkpoints:** Keep previous checkpoint directories and point `system1_path` / `system2_path` in `compare_models` (or in the notebook config) to the older dir to “rollback” to a prior model.
- **Colab:** Re-run from the cell that failed; no automated rollback.

---

## Monitoring

- **Colab:** Use the notebook’s output and Colab’s session/RAM indicators. No built-in error tracking or analytics.
- **Local:** Use terminal/log output from Jupyter or your IDE. For long runs, consider logging to a file or using `tqdm` where already integrated.

No Sentry, UptimeRobot, or similar is configured in the repo; add them if you introduce a server or API.
