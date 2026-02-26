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
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Literal, Optional, Any

# Load .env from project root so HF_TOKEN is available for Hugging Face Hub
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent
    load_dotenv(_root / ".env")
except ImportError:
    pass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# ---------------------------
# Utility: JSON helpers
# ---------------------------

def save_json(path: str, data: Any) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# Teacher loading & labeling
# ---------------------------

def _is_tpu() -> bool:
    """True if TPU is available (e.g. Colab TPU runtime)."""
    if os.environ.get("COLAB_TPU_ADDR"):
        return True
    try:
        import torch_xla.core.xla_model as xm
        return xm.xrt_world_size() > 0
    except Exception:
        return False


def load_teacher(model_name: str, use_tpu: Optional[bool] = None):
    """
    Load teacher model. On GPU: 8-bit, device_map='auto'. On TPU: full precision on XLA device.
    use_tpu: if None, auto-detect from _is_tpu().
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if use_tpu is None:
        use_tpu = _is_tpu()
    if use_tpu:
        try:
            import torch_xla.core.xla_model as xm
        except ImportError:
            raise ImportError("TPU requested but torch_xla not installed. Run the 'TPU setup' cell first.")
        device = xm.xla_device()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(
    instruction: str,
    input_text: str = "",
    mode: Literal["system1", "system2"] = "system1",
) -> str:
    """
    Simple prompt wrapper for teacher and students.
    """
    if mode == "system2":
        extra = (
            "Please think step by step and explain your reasoning "
            "before giving the final answer."
        )
        if input_text:
            return f"User: {instruction}\nInput: {input_text}\n{extra}\nAssistant:"
        return f"User: {instruction}\n{extra}\nAssistant:"
    else:
        if input_text:
            return f"User: {instruction}\nInput: {input_text}\nAssistant:"
        return f"User: {instruction}\nAssistant:"


@torch.no_grad()
def label_with_teacher(
    teacher_model,
    teacher_tokenizer,
    items: List[Dict],
    batch_size: int = 2,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    mode: Literal["system1", "system2"] = "system1",
) -> List[Dict]:
    """
    Given a list of dicts with 'instruction' and optional 'input',
    use the teacher to generate 'output' texts.
    Returns new list of dicts with 'output' filled.
    """
    device = teacher_model.device
    labeled = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        prompts = [
            format_prompt(ex["instruction"], ex.get("input", ""), mode=mode)
            for ex in batch
        ]

        enc = teacher_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        gen = teacher_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=teacher_tokenizer.eos_token_id,
        )

        decoded = teacher_tokenizer.batch_decode(gen, skip_special_tokens=True)

        for orig, prompt, full_text in zip(batch, prompts, decoded):
            if "Assistant:" in full_text:
                answer = full_text.split("Assistant:")[-1].strip()
            else:
                answer = full_text[len(prompt) :].strip()
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
        out = ex.get("output", "")
        items.append({"instruction": instr, "input": "", "output": out})

    if len(items) == 0:
        print("No data for System 1 distillation")
        return

    save_json(out_instructions, items)

    if relabel_with_teacher:
        if teacher_model is None or teacher_tokenizer is None:
            raise ValueError("Teacher model/tokenizer required for relabeling.")
        print(f"Relabeling {len(items)} items with teacher...")
        labeled = label_with_teacher(
            teacher_model, teacher_tokenizer, items, mode="system1"
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
    out_cot: str = "data/omnithought_cot.json",
) -> None:
    """
    System 2 (CoT) dataset prep.
    Loads OmniThought, optionally filters by RV/CD, and maps to
    {instruction, input, output=cot}.
    """
    # Skip split size verification to avoid NonMatchingSplitsSizesError (e.g. partial
    # cache or dataset metadata mismatch).
    def _is_split_error(e: Exception) -> bool:
        return "NonMatchingSplitsSizesError" in type(e).__name__ or "non_matching" in str(e).lower()

    try:
        omni = load_dataset(
            "alibaba-pai/OmniThought",
            split=slice_str,
            trust_remote_code=True,
            verification_mode="no_checks",
        )
    except TypeError:
        try:
            omni = load_dataset("alibaba-pai/OmniThought", split=slice_str, trust_remote_code=True)
        except Exception as e2:
            if _is_split_error(e2):
                omni = load_dataset(
                    "alibaba-pai/OmniThought",
                    split=slice_str,
                    trust_remote_code=True,
                    download_mode="force_redownload",
                )
            else:
                raise
    except Exception as e:
        if _is_split_error(e):
            omni = load_dataset(
                "alibaba-pai/OmniThought",
                split=slice_str,
                trust_remote_code=True,
                verification_mode="no_checks",
                download_mode="force_redownload",
            )
        else:
            raise
    print("First OmniThought sample keys:", list(omni[0].keys()))

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
        items.append({"instruction": instr, "input": "", "output": cot})

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
    template_path: Optional[str] = None,
) -> str:
    """Write EasyDistill config for System 1 (black-box KD).
    template_path: path to chat_template_kd.jinja; if None, uses package-relative path.
    In cloned repo the template is at <repo_root>/configs/chat_template/chat_template_kd.jinja .
    """
    # Official EasyDistill uses configs/chat_template/chat_template_kd.jinja (repo root).
    # When we run from our project root, pass absolute template_path from cloned repo.
    template = template_path if template_path else "configs/chat_template/chat_template_kd.jinja"
    # Resolve paths so config works regardless of cwd when EasyDistill runs
    _instruction = str(Path("data/train_instructions.json").resolve())
    _labeled = str(Path(labeled_path).resolve())
    _out_dir = str(Path(out_dir).resolve())
    config = {
        "job_type": "kd_black_box_local",
        "dataset": {
            "instruction_path": _instruction,
            "labeled_path": _labeled,
            "template": template,
            "seed": 42,
        },
        "inference": {
            "enable_chunked_prefill": True,
            "seed": 777,
            "gpu_memory_utilization": 0.9,
            "temperature": 0.8,
            "trust_remote_code": True,
            "enforce_eager": False,
            "max_model_len": 4096,
            "max_new_tokens": 512,
        },
        "models": {"teacher": teacher_model, "student": student_model},
        "training": {
            "output_dir": _out_dir,
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
            "fp16": True,
        },
    }
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    save_json(config_path, config)
    print(f"Wrote System 1 KD config to {config_path}")
    return config_path


def write_system2_config(
    student_model: str,
    cot_path: str,
    out_dir: str,
    num_train_epochs: int = 1,
    config_path: str = "configs/kd_cot_qwen_1_5b.json",
    template_path: Optional[str] = None,
) -> str:
    """Write EasyDistill config for System 2 (CoT SFT). Uses kd_black_box_train_only (train on
    pre-labeled CoT data; no teacher inference). Pass template_path from cloned EasyDistill
    repo when running from this project."""
    _cot = str(Path(cot_path).resolve())
    _out_dir = str(Path(out_dir).resolve())
    template = template_path if template_path else "configs/chat_template/chat_template_kd.jinja"
    config = {
        "job_type": "kd_black_box_train_only",
        "dataset": {"labeled_path": _cot, "template": template, "seed": 123},
        "models": {"student": student_model},
        "training": {
            "output_dir": _out_dir,
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
            "fp16": True,
        },
    }
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    save_json(config_path, config)
    print(f"Wrote System 2 KD config to {config_path}")
    return config_path


def run_easy_distill(config_path: str) -> int:
    """Run EasyDistill CLI with the given config. Returns the subprocess return code.
    Config path is resolved to absolute so EasyDistill (which joins relative paths with
    its package dir) loads the correct file when we run from our project root.
    Uses the same Python as the current process so the CLI is found after pip install -e.
    """
    abs_config = str(Path(config_path).resolve())
    cmd = [sys.executable, "-m", "easydistill.cli", "--config", abs_config]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("EasyDistill run completed successfully.")
    else:
        print("EasyDistill run failed with code:", result.returncode)
    return result.returncode


def run_training_tpu(
    labeled_path: str,
    student_model: str,
    out_dir: str,
    num_epochs: int = 1,
    max_length: int = 512,
    learning_rate: float = 2e-5,
) -> int:
    """
    Run SFT training on TPU (no EasyDistill/vllm). Uses the same data format as System 1.
    Returns 0 on success.
    """
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    import torch_xla.core.xla_model as xm

    data = read_json(labeled_path)
    if not data:
        print("No labeled data for TPU training")
        return 1
    # Format as Qwen-style chat for SFT
    def _format(ex):
        inst = ex.get("instruction", "")
        out = ex.get("output", "")
        return (
            "<|im_start|>user\n" + inst + "<|im_end|>\n"
            "<|im_start|>assistant\n" + out + "<|im_end|>"
        )
    texts = [_format(ex) for ex in data]
    dataset = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(student_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    device = xm.xla_device()
    model = model.to(device)

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_seq_length=max_length,
        learning_rate=learning_rate,
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=200,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_length,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("TPU training finished. Checkpoint at:", out_dir)
    return 0


# ---------------------------
# High-level entry points
# ---------------------------

def _validate_student_model(model_name: str) -> bool:
    """Return True if student model name is valid (can load tokenizer)."""
    try:
        AutoTokenizer.from_pretrained(model_name, use_fast=True)
        return True
    except Exception:
        return False


def distill_system1(config: Dict[str, Any]) -> Optional[str]:
    """
    Run System 1 distillation end-to-end.
    config: teacher_model, student_model, labeled_path, num_epochs, out_dir, config_path (optional).
    Returns final checkpoint path on success, None otherwise.
    """
    teacher_model = config.get("teacher_model", "Qwen/Qwen2.5-7B-Instruct")
    student_model = config.get("student_model", "Qwen/Qwen2.5-0.5B-Instruct")
    labeled_path = config.get("labeled_path", "data/train_labeled.json")
    num_epochs = config.get("num_epochs", 1)
    out_dir = config.get("out_dir", "./distilled-qwen2.5-0.5b")
    config_path = config.get("config_path", "configs/kd_black_box_qwen_0_5b.json")

    if not _validate_student_model(student_model):
        print(f"Invalid student model: {student_model}")
        print("Example: Qwen/Qwen2.5-0.5B-Instruct")
        return None

    path = Path(labeled_path)
    if not path.exists():
        print("No data for System 1 distillation")
        return None
    data = read_json(labeled_path)
    if not data or len(data) == 0:
        print("No data for System 1 distillation")
        return None

    if _is_tpu():
        print("TPU detected: using TPU training path (no vllm).")
        code = run_training_tpu(
            labeled_path=labeled_path,
            student_model=student_model,
            out_dir=out_dir,
            num_epochs=num_epochs,
        )
        if code != 0:
            return None
        print("Summary: System 1 distillation (TPU) finished. Checkpoint at:", out_dir)
        return out_dir

    write_system1_config(
        teacher_model=teacher_model,
        student_model=student_model,
        labeled_path=labeled_path,
        out_dir=out_dir,
        num_train_epochs=num_epochs,
        config_path=config_path,
        template_path=config.get("template_path"),
    )
    code = run_easy_distill(config_path)
    if code != 0:
        return None
    print("Summary: System 1 distillation finished. Checkpoint at:", out_dir)
    return out_dir


def distill_system2(config: Dict[str, Any]) -> Optional[str]:
    """
    Run System 2 (CoT) distillation end-to-end.
    config: student_model, cot_path, num_epochs, out_dir, config_path (optional).
    Returns final checkpoint path on success, None otherwise.
    """
    student_model = config.get("student_model", "Qwen/Qwen2.5-1.5B-Instruct")
    cot_path = config.get("cot_path", "data/omnithought_cot.json")
    num_epochs = config.get("num_epochs", 1)
    out_dir = config.get("out_dir", "./distilled-qwen2.5-1.5b-cot")
    config_path = config.get("config_path", "configs/kd_cot_qwen_1_5b.json")

    if not _validate_student_model(student_model):
        print(f"Invalid student model: {student_model}")
        print("Example: Qwen/Qwen2.5-1.5B-Instruct")
        return None

    path = Path(cot_path)
    if not path.exists():
        print("No CoT data found. Run Prepare CoT Data cell first.")
        return None
    data = read_json(cot_path)
    if not data or len(data) == 0:
        print("No data for System 2 distillation")
        return None

    write_system2_config(
        student_model=student_model,
        cot_path=cot_path,
        out_dir=out_dir,
        num_train_epochs=num_epochs,
        config_path=config_path,
        template_path=config.get("template_path"),
    )
    code = run_easy_distill(config_path)
    if code != 0:
        return None
    print("Summary: System 2 distillation finished. Checkpoint at:", out_dir)
    return out_dir


# ---------------------------
# Student loading & inference
# ---------------------------

def load_student(model_path: str):
    """Load a distilled student model from a local checkpoint path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def infer_student(
    model,
    tokenizer,
    prompt: str,
    mode: Literal["system1", "system2"] = "system1",
    max_new_tokens: int = 256,
) -> str:
    """Simple inference helper for student models."""
    text = format_prompt(prompt, "", mode=mode)
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    if "Assistant:" in decoded:
        return decoded.split("Assistant:")[-1].strip()
    return decoded.strip()


def _truncate(s: str, max_len: int = 400) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def compare_models(
    prompts: List[str],
    teacher_path: Optional[str] = "Qwen/Qwen2.5-7B-Instruct",
    system1_path: Optional[str] = "./distilled-qwen2.5-0.5b",
    system2_path: Optional[str] = "./distilled-qwen2.5-1.5b-cot",
    teacher_model=None,
    teacher_tokenizer=None,
    sys1_model=None,
    sys1_tokenizer=None,
    sys2_model=None,
    sys2_tokenizer=None,
    max_tokens_teacher: int = 256,
    max_tokens_sys1: int = 256,
    max_tokens_sys2: int = 256,
) -> None:
    """
    Print side-by-side comparison: Prompt, Teacher answer, System 1 student, System 2 student.
    If a model path is missing (or directory doesn't exist), prints "Model not found at <path>" and skips.
    Can pass pre-loaded model/tokenizer pairs to avoid reloading; otherwise loads from paths.
    """
    teacher_loaded = teacher_model is not None and teacher_tokenizer is not None
    sys1_loaded = sys1_model is not None and sys1_tokenizer is not None
    sys2_loaded = sys2_model is not None and sys2_tokenizer is not None

    if not teacher_loaded and teacher_path:
        if not Path(teacher_path).exists() and not teacher_path.startswith("Qwen/"):
            print(f"Model not found at {teacher_path}")
            teacher_path = None
        else:
            try:
                teacher_model, teacher_tokenizer = load_teacher(teacher_path)
                teacher_loaded = True
            except Exception as e:
                print(f"Model not found at {teacher_path} ({e})")
                teacher_path = None
    if not sys1_loaded and system1_path:
        if not Path(system1_path).exists():
            print(f"Model not found at {system1_path}")
            system1_path = None
        else:
            try:
                sys1_model, sys1_tokenizer = load_student(system1_path)
                sys1_loaded = True
            except Exception as e:
                print(f"Model not found at {system1_path} ({e})")
                system1_path = None
    if not sys2_loaded and system2_path:
        if not Path(system2_path).exists():
            print(f"Model not found at {system2_path}")
            system2_path = None
        else:
            try:
                sys2_model, sys2_tokenizer = load_student(system2_path)
                sys2_loaded = True
            except Exception as e:
                print(f"Model not found at {system2_path} ({e})")
                system2_path = None

    for i, prompt in enumerate(prompts):
        print("\n" + "=" * 60)
        print(f"[Prompt {i + 1}] {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print("=" * 60)

        if teacher_loaded:
            t_text = format_prompt(prompt, "", mode="system1")
            enc = teacher_tokenizer(t_text, return_tensors="pt").to(teacher_model.device)
            with torch.no_grad():
                gen = teacher_model.generate(
                    **enc,
                    max_new_tokens=max_tokens_teacher,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=teacher_tokenizer.eos_token_id,
                )
            dec = teacher_tokenizer.decode(gen[0], skip_special_tokens=True)
            ans = dec.split("Assistant:")[-1].strip() if "Assistant:" in dec else dec.strip()
            print("Teacher (truncated):", _truncate(ans))
        else:
            print("Teacher: (skipped)")

        if sys1_loaded:
            ans = infer_student(sys1_model, sys1_tokenizer, prompt, mode="system1", max_new_tokens=max_tokens_sys1)
            print("System 1 student:", _truncate(ans))
        elif system1_path:
            print("System 1 student: (skipped)")
        else:
            print("System 1 student: (not loaded)")

        if sys2_loaded:
            ans = infer_student(sys2_model, sys2_tokenizer, prompt, mode="system2", max_new_tokens=max_tokens_sys2)
            print("System 2 student:", _truncate(ans))
        elif system2_path:
            print("System 2 student: (skipped)")
        else:
            print("System 2 student: (not loaded)")
        print()
