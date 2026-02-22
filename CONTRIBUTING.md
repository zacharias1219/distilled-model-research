# Contributing Guide

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork:  
   `git clone https://github.com/YOUR_USERNAME/distilled-model-research.git`
3. **Create a branch:**  
   `git checkout -b feature/my-feature` or `fix/my-fix`
4. **Make your changes** (code, docs, or notebook).
5. **Run tests** (if present):  
   `pytest` or `python -m pytest`
6. **Commit** with a clear message:  
   `git commit -m "feat: add optional resume from checkpoint"`
7. **Push** to your fork:  
   `git push origin feature/my-feature`
8. **Open a Pull Request** against the upstream `main` (or default) branch.

---

## Development Setup

- **Python:** 3.9+ (3.10+ recommended).
- **Install:**  
  `pip install -r requirements.txt`
- **Notebook:** Open `notebook.ipynb` and run the setup cells so `import distill_app` works (same directory or `sys.path`).
- **GPU:** Optional but recommended for running distillation and comparison.

See [README.md](README.md) for quick start and [DEPLOYMENT.md](DEPLOYMENT.md) for run options.

---

## Code Style

- **Language:** Python 3.9+.
- **Formatting:** Prefer Black-style formatting; line length 88–100 is fine. No formal Black/Flake8 config in repo yet.
- **Linting:** Run `ruff check .` or `flake8 distill_app.py` if you use them locally.
- **Types:** Type hints are used in `distill_app.py`; keep them for public functions.
- **Docstrings:** Public functions have a short docstring; keep them in sync with [API.md](API.md) when you change signatures.

---

## Commit Convention

**Format:** `type(scope): description`

**Types:**

- `feat` — New feature (e.g. new helper or notebook section).
- `fix` — Bug fix.
- `docs` — Documentation only (README, API, ARCHITECTURE, etc.).
- `style` — Formatting, no logic change.
- `refactor` — Code restructure, no behavior change.
- `test` — Add or update tests.
- `chore` — Maintenance (deps, tooling, repo config).

**Examples:**

- `feat(distill): add resume_from_checkpoint in config`
- `fix(compare): handle missing tokenizer.pad_token`
- `docs(README): add Colab GPU requirement`

---

## Testing

- **Current state:** No automated test suite in the repo. Validation is manual (run notebook end-to-end, inspect outputs).
- **Adding tests:** Prefer `pytest`. Good candidates: `format_prompt`, JSON round-trip (`save_json`/`read_json`), and `compare_models` with mocked models. Integration test with a tiny data slice and stub EasyDistill is optional.
- **Coverage:** Not enforced yet; aim to keep new logic covered when you add tests.

---

## Pull Request Guidelines

- **Size:** Prefer smaller PRs (<400 lines of code when possible); split large changes into logical commits.
- **Description:** Clearly describe what changed and why (e.g. “Add RV_MIN/CD_MIN to System 2 config”).
- **Behavior:** All existing notebook flows should still run; document any new required steps in README or DEPLOYMENT.
- **Docs:** Update [API.md](API.md) if you add or change public functions or config keys; update [ARCHITECTURE.md](ARCHITECTURE.md) if you add components or data flow.
- **Notebooks:** If you change the notebook, ensure cells run in order without errors.

---

## Code Review

1. Maintainers (or automated checks, if added) run tests and lint.
2. A reviewer may comment on logic, style, or docs.
3. Address feedback and push to the same branch.
4. Once approved, the PR is merged. There is no automated deploy; “release” is the merged state of the repo.

---

## Questions

- **Bugs or ideas:** Open a GitHub issue.
- **Security:** Prefer responsible disclosure (e.g. private report) for sensitive issues.
