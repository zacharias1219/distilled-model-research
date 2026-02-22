# Product Requirements Document (PRD)  
**Project Name:** Qwen Distillation Lab (System 1 + System 2)  
**Owner:** You (research-focused project in Colab)

***

## 1. Product Overview

### 1.1 Vision (Simple + Technical)

**Kid-level explanation:**  
We have a **big, smart robot brain** (teacher) that is slow and heavy. We want to make **smaller robot brains** (students) that can still answer questions and solve puzzles almost as well, but **faster** and using **less power**.

**Technical vision:**  
Build a Colab-based “distillation lab” that lets a single researcher:

1. Distill a **7B Qwen teacher** into:
   - **System 1 student**: smaller model good at instruction-following.  
   - **System 2 student**: model that can show **step-by-step reasoning (CoT)**.  
2. Understand what is gained/lost by distillation through qualitative and simple quantitative evaluation.  
3. Do all of this without any complicated infrastructure (no backend, no web UI, no DB).

***

### 1.2 Target Users & Personas

**Target Users**

- ML students and researchers doing a distillation-focused project.  
- Practitioners who want to **experiment** with distilling Qwen locally.

**Persona 1 – Richard (Student Researcher)**  
- Final-year BTech student working on an AI/ML research project.  
- Goal: Understand distillation deeply, generate results for a report or paper, and maybe publish code.  
- Tools: Google Colab, Hugging Face, GitHub.  
- Constraints: Limited time and GPU budget, prefers clear, simple, reproducible scripts.

**Persona 2 – Sara (Applied ML Engineer)**  
- Works at a startup, needs to compress big LLMs to fit cost and latency budgets.  
- Goal: Quickly test if a smaller Qwen student can replace a larger teacher in a prototype.  
- Tools: Colab for experiments, later port to cloud GPUs if successful.

***

### 1.3 Success KPIs

1. **Quality Retention**  
   - System 1 and System 2 students achieve **≥90%** of teacher quality on a small internal evaluation set (instruction tasks + reasoning tasks).

2. **Efficiency Gain**  
   - Student inference is **≥3× faster** than the teacher on the same Colab GPU for typical prompts (measured as tokens/sec or latency per response).

3. **Usability**  
   - A new user can clone the repo, open the notebook, and run **both System 1 and System 2 distillation end-to-end in ≤2 hours** without editing Python code.

***

## 2. Scope & Non-Scope

### 2.1 In-Scope

- Running **System 1 distillation** (instruction-following) with Qwen.  
- Running **System 2 distillation** (CoT reasoning) with Qwen.  
- Comparing teacher vs student outputs.  
- Producing usable models stored in Colab filesystem.  
- Documenting everything in **one clear notebook + one helper Python file**.

### 2.2 Out-of-Scope (v1)

- Any web frontend, backend API, authentication.  
- Serving models to external users.  
- Advanced RLHF/RLAIF pipelines beyond what EasyDistill provides.  
- Large-scale multi-user infrastructure.

***

## 3. Feature Specifications

We focus on **three main features**.

***

### Feature 1: System 1 Distillation (Instruction-Following) | Priority: P0

#### 3.1.1 User Story

As an ML practitioner, I want to **distill a large Qwen teacher into a smaller student** for instruction-following so that I can answer user prompts faster and cheaper.

#### 3.1.2 Behavior (Acceptance Criteria)

- [ ] **AC1 – End-to-end run**  
  Given a teacher model (`Qwen/Qwen2.5-7B-Instruct`), a student model (`Qwen/Qwen2.5-0.5B-Instruct`), and a dataset slice (e.g., `train[:1000]` of DistilQwen_100k), when I run the “System 1 Distillation” cell, then:  
  - EasyDistill training completes without errors.  
  - A checkpoint directory (e.g., `distilled-qwen2.5-0.5b`) is created with model and tokenizer files.

- [ ] **AC2 – Basic quality**  
  Given a test prompt (e.g., “Explain what overfitting means.”), when I query the System 1 student, then:  
  - The answer is **non-empty** and **on-topic**.  
  - Latency per answer (128 tokens) is **<2 seconds p95** on Colab Pro GPU.

- [ ] **AC3 – Edge case: no data**  
  If the dataset slice results in **0 samples** (e.g., `train[:0]`), then the distillation cell:  
  - Prints `"No data for System 1 distillation"` and **does not** call EasyDistill.  
  - Does **not** create an empty model directory.

- [ ] **AC4 – Edge case: invalid model name**  
  If the student model name is invalid, the cell catches the exception and prints:  
  - `"Invalid student model: <name>"`  
  - A suggestion like `"Example: Qwen/Qwen2.5-0.5B-Instruct"`.

#### 3.1.3 “UI” – Notebook Layout (for System 1)

No frontend, but the notebook acts like it.

- Section title: **System 1 Distillation (Instruction-Following)**  
- Cells:  
  1. **Config Cell**:  
     - Variables:  
       - `TEACHER_MODEL_SYS1 = "Qwen/Qwen2.5-7B-Instruct"`  
       - `STUDENT_MODEL_SYS1 = "Qwen/Qwen2.5-0.5B-Instruct"`  
       - `DATASET_SLICE_SYS1 = "train[:1000]"`  
       - `NUM_EPOCHS_SYS1 = 1`  
  2. **Prepare Data & Label (Optional)**:  
     - Loads `alibaba-pai/DistilQwen_100k`, maps to `{instruction, input, output}`, optionally re-labels with teacher.  
  3. **Run Distillation**:  
     - Calls `distill_system1(config)` from `distill_app.py`.  
     - Prints summary and final path.

#### 3.1.4 Technical Implementation (System 1)

- **Data**:  
  - Source: `alibaba-pai/DistilQwen_100k` (instruction dataset). [huggingface](https://huggingface.co/datasets/alibaba-pai/DistilQwen_100k)
  - Use HF `datasets` to load and slice.  
  - Optionally generate new teacher outputs with `transformers` and save to `data/train_labeled.json`.

- **Teacher**:  
  - `Qwen/Qwen2.5-7B-Instruct` loaded in 8-bit to save VRAM.  

- **Student**:  
  - `Qwen/Qwen2.5-0.5B-Instruct`.  

- **Training**:  
  - Use EasyDistill black-box KD job (`job_type="kd_black_box_local"`).  
  - Generate KD config JSON in code, then call `!easydistill --config <path>`.  

- **Output**:  
  - Checkpoint folder: `./distilled-qwen2.5-0.5b`.  

***

### Feature 2: System 2 Distillation (Reasoning / Chain-of-Thought) | Priority: P0

#### 3.2.1 User Story

As an ML practitioner, I want to train a **CoT-capable student** so that the model can show its reasoning steps when solving math and logic problems.

#### 3.2.2 Behavior (Acceptance Criteria)

- [ ] **AC1 – End-to-end CoT run**  
  Given a CoT dataset (OmniThought) and a student model (`Qwen/Qwen2.5-1.5B-Instruct`), when I run the “System 2 Distillation” cell, then:  
  - CoT-style KD training completes.  
  - A checkpoint directory (e.g., `distilled-qwen2.5-1.5b-cot`) is created.

- [ ] **AC2 – CoT behavior**  
  Given a reasoning prompt (e.g., “A train travels 120 km in 2 hours…”), when I query the System 2 student with CoT-style prompting, then:  
  - The response contains **multi-step reasoning** (multiple sentences) before a final answer in at least **80%** of tested prompts.  

- [ ] **AC3 – Edge case: OmniThought schema mismatch**  
  If OmniThought fields are not as expected (no `instruction` or `cot` fields), the notebook prints:  
  - `"Unexpected OmniThought schema, please inspect omni[0]"`  
  - Shows `omni[0]` and **does not** start training.

#### 3.2.3 Notebook Layout (System 2)

- Section title: **System 2 Distillation (Reasoning / CoT)**  
- Cells:  
  1. **Config Cell**:  
     - `STUDENT_MODEL_SYS2 = "Qwen/Qwen2.5-1.5B-Instruct"`  
     - `DATASET_SLICE_SYS2 = "train[:2000]"`  
     - `RV_MIN = 0.6`  
     - `CD_MIN = 0.6`  
     - `NUM_EPOCHS_SYS2 = 1`  
  2. **Prepare CoT Data Cell**:  
     - Load OmniThought, filter by RV/CD if present, map to `{instruction, output=cot}` and save `omnithought_cot.json`.  
  3. **Run CoT Distillation Cell**:  
     - Calls `distill_system2(config)` from `distill_app.py`.  

#### 3.2.4 Technical Implementation (System 2)

- **Data**:  
  - Source: `alibaba-pai/OmniThought` dataset.  
  - Filter by `rv_score >= RV_MIN` and `cd_score >= CD_MIN` when available.  
  - Map to JSON with keys `{instruction, input="", output=cot}`.

- **Student**:  
  - `Qwen/Qwen2.5-1.5B-Instruct` (or `0.5B` if VRAM-limited).  

- **Training**:  
  - Use EasyDistill SFT-style KD (`job_type="kd_sft"`).  
  - Config: student model path, `labeled_path="data/omnithought_cot.json"`, hyperparams.  

- **Inference Prompting**:  
  - Prepend:  
    > “Please think step by step and explain your reasoning before giving the final answer.”  

- **Output**:  
  - Checkpoint folder: `./distilled-qwen2.5-1.5b-cot`.  

***

### Feature 3: Teacher vs Student Comparison & Analysis | Priority: P1

#### 3.3.1 User Story

As a researcher, I want to **compare teacher and student outputs side-by-side** so that I can see how behavior changes after distillation.

#### 3.3.2 Behavior (Acceptance Criteria)

- [ ] For a configured list of prompts, a comparison cell prints:  
  - Prompt  
  - Teacher answer (truncated)  
  - System 1 student answer  
  - System 2 student answer (for relevant prompts).  
- [ ] The format is clearly separated by lines and labels, with no ambiguity which model produced which answer.  
- [ ] If a model directory is missing, it prints `"Model not found at <path>"` and skips that model.

#### 3.3.3 Technical Implementation

- Implement a helper function `compare_models(prompts)` in the notebook or `distill_app.py`.  
- Use existing loaded models and tokenizers.  
- Use consistent prompt formatting for fair comparison.  

***

## 4. Technical Requirements

### 4.1 Environment & Tools

- **Runtime**: Google Colab (Python 3.x), with GPU (e.g., T4, P100, or A100).  
- **Notebook**: `notebook.ipynb` in `/content/easydistill`.  
- **Helper Module**: `distill_app.py` in the same directory.

### 4.2 Libraries (Keep Simple)

- `torch`  
- `transformers`  
- `datasets`  
- `easydistill`  
- `bitsandbytes` (for 8-bit loading)  
- `tqdm` (optional)  
- Stdlib: `json`, `pathlib`, `subprocess`, `os`

Total: Well under 15 packages.

### 4.3 Data

- **Datasets** (Hugging Face):  
  - `alibaba-pai/DistilQwen_100k` for System 1. [huggingface](https://huggingface.co/datasets/alibaba-pai/DistilQwen_100k)
  - `alibaba-pai/OmniThought` for System 2.  
- **Storage**: Colab filesystem under `/content/easydistill/data` and model folders.

### 4.4 Models

- **Teacher**:  
  - `Qwen/Qwen2.5-7B-Instruct` (Hugging Face).  
- **Students**:  
  - System 1: `Qwen/Qwen2.5-0.5B-Instruct`.  
  - System 2: `Qwen/Qwen2.5-1.5B-Instruct` (or 0.5B if needed).

***

## 5. Non-Functional Requirements

### 5.1 Performance

- **Inference**:  
  - System 1: p95 <2 seconds per 128-token answer.  
  - System 2: p95 <3 seconds per 256-token CoT answer.  

- **Training Time Targets** (rough):  
  - System 1: 1 epoch on 1k samples within ~1 hour.  
  - System 2: 1 epoch on ~2k OmniThought samples within ~2 hours.

### 5.2 Security

- No PII: You do not ingest any real user data; only HF datasets and your own prompts.  
- Notebook is private to your account.  
- No external API keys required for core flow.

### 5.3 Scale

- Designed strictly for **single-user, single-notebook** experimentation.  
- No concurrency or multi-user considerations.

### 5.4 Accessibility

- Not applicable in the web sense, but notebook sections must be clearly labeled, with comments explaining each step in simple and technical terms.

***

## 6. Risks & Dependencies

### 6.1 Risks

1. **Risk: Student performance too low (<80% of teacher)**  
   - *Mitigation:*  
     - Increase dataset size.  
     - Increase training epochs.  
     - For System 2, tighten RV/CD filters to higher-quality CoTs.  

2. **Risk: Colab GPU limitations (OOM, timeouts)**  
   - *Mitigation:*  
     - Load teacher in 8-bit.  
     - Use smaller student (0.5B instead of 1.5B).  
     - Reduce batch size, use gradient accumulation.

3. **Risk: HF model/dataset temporarily down**  
   - *Mitigation:*  
     - Cache models/datasets to Google Drive or local storage after first run.  
     - Keep small exported JSON slices for development.

### 6.2 Dependencies

- Hugging Face model and dataset availability.  
- EasyDistill library and its CLI behavior.  
- Colab GPU availability and session lifetime.

***

## 7. Launch Plan (for Research Project)

### 7.1 “Launch” Definition

The project is considered “launched” when:

- `notebook.ipynb` runs end-to-end (System 1 + System 2 + comparison) on Colab without errors.  
- `distill_app.py` contains all helper functions and is under version control.  
- At least one pair of student checkpoints (Sys1 & Sys2) are stored and testable.

### 7.2 Demo

- **Entry Point**:  
  - GitHub repo with `notebook.ipynb` and `distill_app.py`.  
  - “Open in Colab” badge in the README.  

- **How to demo**:  
  1. Open notebook in Colab.  
  2. Run setup cells.  
  3. Run System 1 distillation (small slice).  
  4. Run System 2 distillation (small slice).  
  5. Run comparison cell and walk through examples.

### 7.3 Monitoring & Debugging

- Use Colab output & tracebacks.  
- For important steps, print clear messages:
  - Loaded dataset size.  
  - Config used.  
  - Paths of saved models.  
- If errors occur, add simple `try/except` with helpful messages.

### 7.4 Rollback

- **Code**: Use Git to revert to previous commit if a change breaks the notebook.  
- **Models**: Keep older student checkpoint folders and manually switch `model_path` in inference cells.

***

### Part 1 – `distill_app.py` Skeleton

Below is a **clean, small skeleton** you can drop into `/content/easydistill/distill_app.py`.  
It’s written to match the PRD and your Colab workflow.

```python
# distill_app.py
"""
Minimal helper module for Qwen distillation experiments.

Responsibilities:
- Load teacher model
- Prepare datasets (System 1 & System 2)
- Optional teacher labeling
- Write EasyDistill config JSONs
- Run EasyDistill via CLI
- Load & query student models
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Literal, Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# ---------------------------
# Utility: JSON helpers
# ---------------------------

def save_json(path: str, data) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# Teacher loading & labeling
# ---------------------------

def load_teacher(model_name: str):
    """
    Load teacher model in 8-bit if possible to save VRAM.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(instruction: str,
                  input_text: str = "",
                  mode: Literal["system1", "system2"] = "system1") -> str:
    """
    Simple prompt wrapper for teacher and students.
    """
    if mode == "system2":
        # Encourage chain-of-thought
        extra = (
            "Please think step by step and explain your reasoning "
            "before giving the final answer."
        )
        if input_text:
            return f"User: {instruction}\nInput: {input_text}\n{extra}\nAssistant:"
        return f"User: {instruction}\n{extra}\nAssistant:"
    else:
        # System 1: simple instruction-following
        if input_text:
            return f"User: {instruction}\nInput: {input_text}\nAssistant:"
        return f"User: {instruction}\nAssistant:"


@torch.no_grad()
def label_with_teacher(
    teacher_model,
    tokenizer,
    items: List[Dict],
    batch_size: int = 2,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    mode: Literal["system1", "system2"] = "system1"
) -> List[Dict]:
    """
    Given a list of dicts with 'instruction' and optional 'input',
    use the teacher to generate 'output' texts.
    Returns new list of dicts with 'output' filled.
    """
    device = teacher_model.device
    labeled = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = [
            format_prompt(ex["instruction"], ex.get("input", ""), mode=mode)
            for ex in batch
        ]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        gen = teacher_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for orig, prompt, full_text in zip(batch, prompts, decoded):
            if "Assistant:" in full_text:
                answer = full_text.split("Assistant:")[-1].strip()
            else:
                answer = full_text[len(prompt):].strip()
            new_ex = dict(orig)
            new_ex["output"] = answer
            labeled.append(new_ex)

    return labeled


# ---------------------------
# Dataset preparation
# ---------------------------

def prepare_system1_dataset(
    slice_str: str = "train[:1000]",
    teacher_model=None,
    teacher_tokenizer=None,
    relabel_with_teacher: bool = False,
    out_instructions: str = "data/train_instructions.json",
    out_labeled: str = "data/train_labeled.json",
) -> None:
    """
    System 1 (instruction-following) dataset prep.
    Loads DistilQwen_100k, converts to {instruction, input, output}, and
    optionally re-labels with the teacher.
    """
    ds = load_dataset("alibaba-pai/DistilQwen_100k", split=slice_str)
    items = []

    for ex in ds:
        instr = ex.get("instruction", "") or ex.get("input", "")
        # Keep existing output, but we may overwrite it if relabeling
        out = ex.get("output", "")
        items.append({
            "instruction": instr,
            "input": "",
            "output": out
        })

    if len(items) == 0:
        print("No data for System 1 distillation (slice was empty).")
        return

    save_json(out_instructions, items)

    if relabel_with_teacher:
        if teacher_model is None or teacher_tokenizer is None:
            raise ValueError("Teacher model/tokenizer required for relabeling.")
        print(f"Relabeling {len(items)} items with teacher...")
        labeled = label_with_teacher(
            teacher_model,
            teacher_tokenizer,
            items,
            mode="system1"
        )
    else:
        labeled = items

    save_json(out_labeled, labeled)
    print(f"Saved System 1 instructions to {out_instructions}")
    print(f"Saved System 1 labeled data to {out_labeled}")


def prepare_system2_dataset(
    slice_str: str = "train[:2000]",
    rv_min: float = 0.6,
    cd_min: float = 0.6,
    out_cot: str = "data/omnithought_cot.json"
) -> None:
    """
    System 2 (CoT) dataset prep.
    Loads OmniThought, optionally filters by RV/CD, and maps to
    {instruction, input, output=cot}.
    """
    omni = load_dataset("alibaba-pai/OmniThought", split=slice_str)
    print("First OmniThought sample keys:", omni[0].keys())

    # Check schema
    if not (
        ("instruction" in omni[0] or "question" in omni[0] or "input" in omni[0])
        and ("cot" in omni[0] or "output" in omni[0] or "answer" in omni[0])
    ):
        print("Unexpected OmniThought schema, please inspect omni[0]:", omni[0])
        return

    has_rv = "rv_score" in omni[0]
    has_cd = "cd_score" in omni[0]

    if has_rv and has_cd:
        omni = omni.filter(
            lambda x: x["rv_score"] >= rv_min and x["cd_score"] >= cd_min
        )
        print(f"Filtered by RV/CD, remaining samples: {len(omni)}")
    else:
        print("RV/CD scores not found, using raw slice:", len(omni))

    items = []
    for ex in omni:
        instr = ex.get("instruction") or ex.get("question") or ex.get("input", "")
        cot = ex.get("cot") or ex.get("output") or ex.get("answer", "")
        items.append({
            "instruction": instr,
            "input": "",
            "output": cot
        })

    if len(items) == 0:
        print("No data for System 2 distillation after filtering.")
        return

    save_json(out_cot, items)
    print(f"Saved System 2 CoT data to {out_cot}")


# ---------------------------
# EasyDistill config & runner
# ---------------------------

def write_system1_config(
    teacher_model: str,
    student_model: str,
    labeled_path: str,
    out_dir: str,
    num_train_epochs: int = 1,
    config_path: str = "configs/kd_black_box_qwen_0_5b.json",
) -> str:
    """
    Write EasyDistill config for System 1 (black-box KD).
    """
    config = {
        "job_type": "kd_black_box_local",
        "dataset": {
            "instruction_path": "data/train_instructions.json",
            "labeled_path": labeled_path,
            "template": "easydistill/chat_template/chat_template_kd.jinja",
            "seed": 42
        },
        "inference": {
            "enable_chunked_prefill": True,
            "seed": 777,
            "gpu_memory_utilization": 0.9,
            "temperature": 0.8,
            "trust_remote_code": True,
            "enforce_eager": False,
            "max_model_len": 4096,
            "max_new_tokens": 512
        },
        "models": {
            "teacher": teacher_model,
            "student": student_model
        },
        "training": {
            "output_dir": out_dir,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "save_steps": 200,
            "logging_steps": 10,
            "max_length": 512,
            "learning_rate": 2e-5,
            "weight_decay": 0.05,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "bf16": False,
            "fp16": True
        }
    }
    save_json(config_path, config)
    print(f"Wrote System 1 KD config to {config_path}")
    return config_path


def write_system2_config(
    student_model: str,
    cot_path: str,
    out_dir: str,
    num_train_epochs: int = 1,
    config_path: str = "configs/kd_cot_qwen_1_5b.json",
) -> str:
    """
    Write EasyDistill config for System 2 (CoT SFT).
    """
    config = {
        "job_type": "kd_sft",
        "dataset": {
            "labeled_path": cot_path,
            "seed": 123
        },
        "models": {
            "student": student_model
        },
        "training": {
            "output_dir": out_dir,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "save_steps": 200,
            "logging_steps": 10,
            "max_length": 1024,
            "learning_rate": 1e-5,
            "weight_decay": 0.05,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "bf16": False,
            "fp16": True
        }
    }
    save_json(config_path, config)
    print(f"Wrote System 2 KD config to {config_path}")
    return config_path


def run_easy_distill(config_path: str) -> int:
    """
    Run EasyDistill CLI with the given config.
    Returns the subprocess return code.
    """
    cmd = ["easydistill", "--config", config_path]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("EasyDistill run completed successfully.")
    else:
        print("EasyDistill run failed with code:", result.returncode)
    return result.returncode


# ---------------------------
# Student loading & inference
# ---------------------------

def load_student(model_path: str):
    """
    Load a distilled student model from a local checkpoint path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def infer_student(
    model,
    tokenizer,
    prompt: str,
    mode: Literal["system1", "system2"] = "system1",
    max_new_tokens: int = 256
) -> str:
    """
    Simple inference helper for student models.
    """
    text = format_prompt(prompt, "", mode=mode)
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    if "Assistant:" in decoded:
        return decoded.split("Assistant:")[-1].strip()
    return decoded.strip()
```

This is a **skeleton**: you can trim or adapt for your experiments, but it already matches:

- System 1 dataset prep + KD config + training  
- System 2 CoT dataset prep + KD config + training  
- Teacher loading + optional labeling  
- Student loading + inference  

***

### Part 2 – Experiments & Ablations Checklist

Below is a **concrete list of experiments** you can run and document in your report.  
Each one helps you understand a key aspect of distillation.

#### Experiment Group A – Data Size Ablation (System 1)

**Goal:** How does the number of training examples affect student quality?

1. **Setup**  
   - Teacher: `Qwen/Qwen2.5-7B-Instruct`  
   - Student: `Qwen/Qwen2.5-0.5B-Instruct`  
   - Dataset: DistilQwen_100k  
   - Epochs: 1 (fixed)  

2. **Variants**  
   - A1: `train[:500]`  
   - A2: `train[:2000]`  
   - A3: `train[:8000]`  

3. **Evaluation**  
   - Fixed prompt set (e.g., 30 instruction prompts: definition, coding, explanation).  
   - For each variant:
     - Measure average subjective score (you rate 1–5 for quality).  
     - Measure avg latency (tokens/sec).  

4. **Report**  
   - Table: data size vs perceived quality vs latency.  
   - Short discussion: diminishing returns, over/underfitting signs.

***

#### Experiment Group B – Teacher Labeling vs Reusing Dataset Outputs (System 1)

**Goal:** Does re-labeling with the teacher improve the student?

1. **Setup**  
   - Same as Group A, fixed size: e.g., `train[:2000]`.  

2. **Variants**  
   - B1: Use existing `output` from DistilQwen_100k (no relabel).  
   - B2: Use `label_with_teacher` to re-generate all outputs from Qwen2.5-7B.  

3. **Evaluation**  
   - Same prompt set as Group A.  
   - Compare clarity, helpfulness, and consistency with teacher.  

4. **Report**  
   - Describe if teacher-labeled data produces more “teacher-like” student answers.  
   - Note runtime cost of relabeling.

***

#### Experiment Group C – System 1 vs Teacher: Failure Analysis

**Goal:** Understand *where* the distilled model fails compared to teacher.

1. **Procedure**  
   - Pick 20 mixed prompts (general questions, coding, explanation, edge cases).  
   - For each prompt:
     - Save teacher answer and System 1 student answer.  

2. **Analysis Dimensions**  
   - Hallucination (fabricates facts).  
   - Missing details / shallow explanation.  
   - Formatting problems (code blocks, lists).  

3. **Report**  
   - Qualitative examples of clear failures.  
   - Hypotheses: is it data size, capacity, or KD choice?

***

#### Experiment Group D – CoT Distillation: RV/CD Filtering (System 2)

**Goal:** How do RV/CD thresholds impact reasoning quality?

1. **Setup**  
   - Student: `Qwen/Qwen2.5-1.5B-Instruct` (or 0.5B).  
   - Dataset: OmniThought.  
   - Epochs: 1, same training config.  

2. **Variants**  
   - D1: No filtering (use `train[:2000]` as-is).  
   - D2: `rv_score >= 0.4`, `cd_score >= 0.4`.  
   - D3: `rv_score >= 0.7`, `cd_score >= 0.7`.  

3. **Evaluation**  
   - 20 reasoning prompts (math, logic, word problems).  
   - For each variant, measure:
     - % answers with explicit multi-step reasoning.  
     - Your rating of correctness (0, 0.5, 1).  

4. **Report**  
   - Plot or table of filter strictness vs reasoning quality.  
   - Discuss trade-off between data quantity and quality.

***

#### Experiment Group E – System 2 Student Size Ablation

**Goal:** How much capacity do you need for CoT?

1. **Setup**  
   - Same RV/CD filter and data size.  

2. **Variants**  
   - E1: Student = `Qwen/Qwen2.5-0.5B-Instruct`.  
   - E2: Student = `Qwen/Qwen2.5-1.5B-Instruct`.  

3. **Evaluation**  
   - Same 20 reasoning prompts.  
   - Compare:
     - CoT completeness (number of steps).  
     - Correctness.  
     - Latency.  

4. **Report**  
   - Discuss capacity vs performance trade-offs for CoT.

***

#### Experiment Group F – Prompting Ablation (System 2)

**Goal:** How sensitive is CoT behavior to prompting?

1. **Variants**  
   - F1: No CoT instruction (just “User: <question>”).  
   - F2: “Please think step by step…” (current formulation).  
   - F3: More explicit: “First list known facts, then compute, then decide.”  

2. **Evaluation**  
   - Compare step-by-step structure and correctness across variants.  

3. **Report**  
   - Show examples where prompt design dramatically changes behavior.  
   - Connect to distillation: student learned teacher’s style, but still needs good prompts.

***

#### Experiment Group G – Robustness / Generalization (Optional)

**Goal:** See if distilled models generalize beyond training domains.

1. **Procedure**  
   - Design prompts that differ from DistilQwen_100k / OmniThought style:  
     - Programming tasks not seen before.  
     - Multi-lingual prompts.  
     - Domain-specific questions (e.g., ML theory, your coursework).  

2. **Evaluation**  
   - Teacher vs System 1 vs System 2 on these prompts.  

3. **Report**  
   - Where does the student generalize well vs fail?  
   - Does CoT help or hurt in out-of-distribution cases?

***

#### How to Turn These Into a Report

For each experiment group:

1. **Explain the question** you’re asking (1–2 sentences).  
2. **Describe setup** (models, data, config).  
3. **Show results** (table or simple plot + a few qualitative examples).  
4. **Write a short conclusion** (what you learned, what you’d do next).

If you want, I can next:

- Turn one experiment group (e.g., A or D) into a **ready-to-run notebook section**, or  
- Suggest a **report outline** that ties all these experiments into a coherent narrative.