# API Documentation

This project has **no HTTP API**. The public interface is the **Python module `distill_app`** and the **EasyDistill CLI**. This document describes the programmatic API: functions, config dicts, and file contracts.

---

## Base usage

- **Import:** `from distill_app import distill_system1, distill_system2, compare_models, load_teacher, load_student, ...`
- **Authentication:** None. For gated Hugging Face models/datasets, set `HF_TOKEN` or log in via `huggingface-cli login`.
- **Context:** Run from the project root (or ensure `distill_app.py` is on `sys.path`). Paths in configs are relative to current working directory unless noted.

---

## Entry points (high-level)

### `distill_system1(config: Dict[str, Any]) -> Optional[str]`

Runs System 1 (instruction-following) distillation end-to-end.

**Config keys:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `teacher_model` | str | `"Qwen/Qwen2.5-7B-Instruct"` | Hugging Face model id for teacher. |
| `student_model` | str | `"Qwen/Qwen2.5-0.5B-Instruct"` | Hugging Face model id for student. |
| `labeled_path` | str | `"data/train_labeled.json"` | Path to JSON list of `{instruction, input, output}`. |
| `num_epochs` | int | 1 | Training epochs. |
| `out_dir` | str | `"./distilled-qwen2.5-0.5b"` | Checkpoint output directory. |
| `config_path` | str | `"configs/kd_black_box_qwen_0_5b.json"` | Where to write EasyDistill config. |

**Returns:** `out_dir` on success, `None` on failure (invalid student, no data, or EasyDistill non-zero exit).

**Side effects:** Writes JSON config to `config_path`; runs `easydistill --config <config_path>`; creates `out_dir` and checkpoint files.

**Errors (printed, not raised):**

- Invalid student model → `"Invalid student model: <name>"` + example.
- Missing or empty `labeled_path` → `"No data for System 1 distillation"`.

---

### `distill_system2(config: Dict[str, Any]) -> Optional[str]`

Runs System 2 (CoT) distillation end-to-end.

**Config keys:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `student_model` | str | `"Qwen/Qwen2.5-1.5B-Instruct"` | Hugging Face model id for student. |
| `cot_path` | str | `"data/omnithought_cot.json"` | Path to JSON list of `{instruction, input, output}` (output = CoT). |
| `num_epochs` | int | 1 | Training epochs. |
| `out_dir` | str | `"./distilled-qwen2.5-1.5b-cot"` | Checkpoint output directory. |
| `config_path` | str | `"configs/kd_cot_qwen_1_5b.json"` | Where to write EasyDistill config. |

**Returns:** `out_dir` on success, `None` on failure.

**Side effects:** Same pattern as System 1 (writes config, runs EasyDistill).

**Errors (printed):** Invalid student; missing or empty `cot_path` → no data message.

---

### `compare_models(prompts, ...) -> None`

Prints a side-by-side comparison: for each prompt, prints **Prompt**, **Teacher** (truncated), **System 1 student**, **System 2 student**. Missing or unloadable models are skipped with `"Model not found at <path>"`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | List[str] | required | List of prompt strings. |
| `teacher_path` | Optional[str] | `"Qwen/Qwen2.5-7B-Instruct"` | Teacher HF id or local path. |
| `system1_path` | Optional[str] | `"./distilled-qwen2.5-0.5b"` | System 1 checkpoint dir. |
| `system2_path` | Optional[str] | `"./distilled-qwen2.5-1.5b-cot"` | System 2 checkpoint dir. |
| `teacher_model` | optional | None | Pre-loaded teacher model (avoids reload). |
| `teacher_tokenizer` | optional | None | Pre-loaded teacher tokenizer. |
| `sys1_model` | optional | None | Pre-loaded System 1 model. |
| `sys1_tokenizer` | optional | None | Pre-loaded System 1 tokenizer. |
| `sys2_model` | optional | None | Pre-loaded System 2 model. |
| `sys2_tokenizer` | optional | None | Pre-loaded System 2 tokenizer. |
| `max_tokens_teacher` | int | 256 | Max new tokens for teacher. |
| `max_tokens_sys1` | int | 256 | Max new tokens for System 1. |
| `max_tokens_sys2` | int | 256 | Max new tokens for System 2. |

**Returns:** None (prints to stdout).

**Errors:** Missing local path or load failure → message and that column skipped.

---

## Data preparation

### `prepare_system1_dataset(...) -> None`

Loads `alibaba-pai/DistilQwen_100k`, slices, maps to `{instruction, input, output}`, optionally relabels with teacher, and saves JSON files.

**Parameters:** `slice_str`, `teacher_model`, `teacher_tokenizer`, `relabel_with_teacher`, `out_instructions`, `out_labeled`. See docstring in `distill_app.py`.

**Side effects:** Writes `out_instructions` and `out_labeled`. If slice is empty, prints `"No data for System 1 distillation"` and returns without writing.

---

### `prepare_system2_dataset(...) -> None`

Loads `alibaba-pai/OmniThought`, optionally filters by `rv_score`/`cd_score`, maps to `{instruction, input, output}` (output = CoT), saves JSON.

**Parameters:** `slice_str`, `rv_min`, `cd_min`, `out_cot`. See docstring.

**Side effects:** Writes `out_cot`. On schema mismatch, prints `"Unexpected OmniThought schema, please inspect omni[0]"` and returns without writing.

---

## Model loading and inference

### `load_teacher(model_name: str) -> Tuple[Model, Tokenizer]`

Loads teacher from Hugging Face with 8-bit quantization. Sets `pad_token` to `eos_token` if unset.

**Raises:** Propagates Hugging Face `from_pretrained` errors.

---

### `load_student(model_path: str) -> Tuple[Model, Tokenizer]`

Loads student from local checkpoint directory. Same tokenizer convention as teacher.

---

### `infer_student(model, tokenizer, prompt: str, mode="system1"|"system2", max_new_tokens=256) -> str`

Formats `prompt` with `format_prompt(..., mode)`, runs generation, returns only the assistant reply (after `"Assistant:"` or end of prompt).

---

## Helpers (used by entry points)

### `format_prompt(instruction, input_text="", mode="system1"|"system2") -> str`

Returns chat string: `User: ... [Input: ...] [CoT instruction for system2] Assistant:`.

### `label_with_teacher(teacher_model, teacher_tokenizer, items, batch_size=2, max_new_tokens=256, temperature=0.8, top_p=0.9, mode="system1"|"system2") -> List[Dict]`

Takes list of `{instruction, input}` (optional `output`), generates `output` with teacher, returns list of dicts with `output` filled.

### `write_system1_config(...) -> str`

Writes EasyDistill black-box KD config JSON; returns `config_path`.

### `write_system2_config(...) -> str`

Writes EasyDistill kd_black_box_train_only config JSON; returns `config_path`. Optional `template_path` (absolute path to `chat_template_kd.jinja` in cloned EasyDistill) for correct formatting.

### `run_easy_distill(config_path: str) -> int`

Runs `easydistill --config <config_path>`; returns subprocess return code.

### `save_json(path, data)` / `read_json(path)`

Standard JSON write/read with UTF-8 and `ensure_ascii=False` for write.

---

## Data contracts

### Labeled / CoT JSON format

All training JSON files are a **list of objects** with:

- `instruction` (str)
- `input` (str, may be `""`)
- `output` (str)

### EasyDistill config (generated)

- **System 1:** `job_type: "kd_black_box_local"`, `dataset.instruction_path`, `dataset.labeled_path`, `models.teacher`, `models.student`, `training.output_dir`, etc.
- **System 2:** `job_type: "kd_black_box_train_only"`, `dataset.labeled_path`, `dataset.template`, `models.student`, `training.output_dir`, etc.

Paths inside the config are as passed (relative to cwd when EasyDistill runs).

---

## External CLI: EasyDistill

**Command:** `easydistill --config <path_to_json>`

**Input:** JSON config file produced by `write_system1_config` or `write_system2_config`.

**Output:** Checkpoint directory (e.g. `./distilled-qwen2.5-0.5b`), created by EasyDistill. Exit code 0 = success; non-zero = failure (see terminal output).

**Rate limiting:** N/A (single local process). No HTTP or API keys.
