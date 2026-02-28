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
import glob


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
# Utility: Checkpoint discovery
# ---------------------------

_MODEL_FILES = {
    "config.json", "pytorch_model.bin", "model.safetensors",
    "adapter_config.json", "adapter_model.safetensors",
}


def _dir_has_model(d: Path) -> bool:
    """Return True if directory contains recognizable model files."""
    if not d.is_dir():
        return False
    files = {f.name for f in d.iterdir() if f.is_file()}
    # Also check for sharded safetensors (model-00001-of-*.safetensors)
    has_sharded = any(f.name.startswith("model-") and f.name.endswith(".safetensors") for f in d.iterdir())
    return bool(files & _MODEL_FILES) or has_sharded


def find_checkpoint(out_dir: str) -> Optional[str]:
    """Find a valid model checkpoint in *out_dir* or its subdirectories.

    EasyDistill may save the final model directly in *out_dir* or inside a
    numbered subdirectory like ``checkpoint-200``.  This helper looks in both
    places and returns the best (highest-step) checkpoint path, or *None* if
    nothing is found.
    """
    base = Path(out_dir)
    if not base.exists():
        return None
    # 1. Check the directory itself
    if _dir_has_model(base):
        return str(base)
    # 2. Look for checkpoint-* subdirs (sorted by step number, highest first)
    ckpts = sorted(
        base.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        reverse=True,
    )
    for ckpt in ckpts:
        if _dir_has_model(ckpt):
            return str(ckpt)
    # 3. Check any immediate subdirectory
    for child in sorted(base.iterdir(), reverse=True):
        if child.is_dir() and _dir_has_model(child):
            return str(child)
    return None


def check_resume(out_dir: str) -> Optional[str]:
    """If a partial checkpoint exists in *out_dir*, return the path for resuming."""
    ckpt = find_checkpoint(out_dir)
    if ckpt:
        print(f"Found existing checkpoint at {ckpt} — can resume from here.")
    return ckpt


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
    Falls back to CPU with float16 if no GPU/bitsandbytes available.
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
        model = None
        # Try 1: 8-bit quantized on GPU
        if torch.cuda.is_available():
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    load_in_8bit=True,
                )
            except Exception as e:
                print(f"8-bit loading failed: {e}")
        # Try 2: float16 on GPU
        if model is None and torch.cuda.is_available():
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                print("Loaded teacher in float16 on GPU.")
            except Exception as e:
                print(f"float16 GPU loading failed: {e}")
        # Try 3: CPU fallback with float32 (slow but works)
        if model is None:
            print("WARNING: Loading teacher on CPU (float32). This will be slow for large models.")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not load teacher model '{model_name}' on any device. "
                    f"Last error: {e}. Try using a GPU runtime."
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
    Loads OmniThought via streaming (to avoid downloading all 135 shards),
    optionally filters by RV/CD, and maps to {instruction, input, output=cot}.
    """
    from tqdm import tqdm

    # Parse the requested sample count from slice_str, e.g. "train[:2000]" -> 2000
    import re
    count_match = re.search(r'\[:(\d+)\]', slice_str)
    max_samples = int(count_match.group(1)) if count_match else 2000
    print(f"Preparing System 2 dataset: collecting {max_samples} samples via streaming...")

    # ---------- Try streaming first (avoids downloading all 135 shards) ----------
    omni_stream = None
    try:
        omni_stream = load_dataset(
            "alibaba-pai/OmniThought",
            split="train",
            streaming=True,
            verification_mode="no_checks",
        )
    except Exception as e:
        if "trust_remote_code" in str(e).lower() or "custom code" in str(e).lower():
            try:
                omni_stream = load_dataset(
                    "alibaba-pai/OmniThought",
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                    verification_mode="no_checks",
                )
            except Exception:
                pass

    if omni_stream is not None:
        # Streaming path: iterate and collect samples
        items = []
        first_sample = None
        for ex in tqdm(omni_stream, total=max_samples, desc="Streaming OmniThought"):
            if first_sample is None:
                first_sample = ex
                print("First OmniThought sample keys:", list(ex.keys()))

            has_reasoning_list = "reasoning" in ex and isinstance(ex["reasoning"], list)

            if has_reasoning_list:
                cd_thresh = rv_min * 10 if rv_min <= 1.0 else rv_min
                rv_thresh = cd_min * 10 if cd_min <= 1.0 else cd_min
                instr = ex.get("question") or ex.get("instruction") or ex.get("input", "")
                entries = ex.get("reasoning", [])
                if not entries:
                    continue
                best = None
                for entry in entries:
                    cd_lvl = entry.get("Cognitive_Difficulty", {}).get("level", 0)
                    rv_lvl = entry.get("Reasoning_Verbosity", {}).get("level", 0)
                    if cd_lvl >= cd_thresh and rv_lvl >= rv_thresh:
                        best = entry
                        break
                if best is None:
                    best = entries[0]
                cot = best.get("full_response") or best.get("solution") or best.get("thought", "")
                if not cot or not cot.strip():
                    continue
                items.append({"instruction": instr, "input": "", "output": cot})
            else:
                instr = ex.get("instruction") or ex.get("question") or ex.get("input", "")
                cot = ex.get("cot") or ex.get("output") or ex.get("answer", "")
                if cot:
                    items.append({"instruction": instr, "input": "", "output": cot})

            if len(items) >= max_samples:
                break

        print(f"Collected {len(items)} CoT samples via streaming.")
    else:
        # ---------- Fallback: non-streaming download ----------
        print("Streaming not available, falling back to full download...")
        def _is_split_error(e: Exception) -> bool:
            return "NonMatchingSplitsSizesError" in type(e).__name__ or "non_matching" in str(e).lower()
        try:
            omni = load_dataset(
                "alibaba-pai/OmniThought",
                split=slice_str,
                verification_mode="no_checks",
            )
        except Exception as e:
            if "trust_remote_code" in str(e).lower() or "custom code" in str(e).lower():
                try:
                    omni = load_dataset(
                        "alibaba-pai/OmniThought",
                        split=slice_str,
                        trust_remote_code=True,
                        verification_mode="no_checks",
                    )
                except TypeError:
                    omni = load_dataset("alibaba-pai/OmniThought", split=slice_str, trust_remote_code=True)
            elif _is_split_error(e):
                omni = load_dataset(
                    "alibaba-pai/OmniThought",
                    split=slice_str,
                    verification_mode="no_checks",
                    download_mode="force_redownload",
                )
            else:
                raise
        sample_keys = list(omni[0].keys())
        print("First OmniThought sample keys:", sample_keys)

        has_question = "instruction" in omni[0] or "question" in omni[0] or "input" in omni[0]
        has_reasoning_list = "reasoning" in omni[0] and isinstance(omni[0]["reasoning"], list)
        has_flat_cot = "cot" in omni[0] or "output" in omni[0] or "answer" in omni[0]

        if not has_question or (not has_reasoning_list and not has_flat_cot):
            print("Unexpected OmniThought schema, please inspect omni[0]:")
            print({k: type(v).__name__ for k, v in omni[0].items()})
            return

        items = []

        if has_reasoning_list:
            cd_thresh = rv_min * 10 if rv_min <= 1.0 else rv_min
            rv_thresh = cd_min * 10 if cd_min <= 1.0 else cd_min
            for ex in omni:
                instr = ex.get("question") or ex.get("instruction") or ex.get("input", "")
                entries = ex.get("reasoning", [])
                if not entries:
                    continue
                best = None
                for entry in entries:
                    cd_lvl = entry.get("Cognitive_Difficulty", {}).get("level", 0)
                    rv_lvl = entry.get("Reasoning_Verbosity", {}).get("level", 0)
                    if cd_lvl >= cd_thresh and rv_lvl >= rv_thresh:
                        best = entry
                        break
                if best is None:
                    best = entries[0]
                cot = best.get("full_response") or best.get("solution") or best.get("thought", "")
                if not cot or not cot.strip():
                    continue
                items.append({"instruction": instr, "input": "", "output": cot})
            print(f"Extracted {len(items)} CoT samples from reasoning entries (CD>={cd_thresh}, RV>={rv_thresh})")
        else:
            has_rv = "rv_score" in omni[0]
            has_cd = "cd_score" in omni[0]
            if has_rv and has_cd:
                omni = omni.filter(
                    lambda x: x["rv_score"] >= rv_min and x["cd_score"] >= cd_min
                )
                print(f"Filtered by RV/CD, remaining samples: {len(omni)}")
            else:
                print("RV/CD scores not found, using raw slice:", len(omni))
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
    Captures stdout/stderr for diagnostic output.
    """
    abs_config = str(Path(config_path).resolve())
    cmd = [sys.executable, "-m", "easydistill.cli", "--config", abs_config]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Print last 50 lines of stdout for diagnostics
    if result.stdout:
        lines = result.stdout.strip().splitlines()
        if len(lines) > 50:
            print(f"... ({len(lines) - 50} lines omitted) ...")
        for line in lines[-50:]:
            print(line)
    if result.stderr:
        err_lines = result.stderr.strip().splitlines()
        # Show last 20 lines of stderr (often warnings)
        for line in err_lines[-20:]:
            print("[stderr]", line)
    if result.returncode == 0:
        print("EasyDistill run completed successfully.")
    else:
        print("EasyDistill run failed with code:", result.returncode)
        if result.stderr:
            print("Full stderr:")
            print(result.stderr[-2000:])
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

    # Check for existing checkpoint (resume support)
    existing = find_checkpoint(out_dir)
    if existing:
        print(f"Existing checkpoint found at {existing}. Skipping training (delete to retrain).")
        return existing

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
        resolved = find_checkpoint(out_dir) or str(Path(out_dir).resolve())
        print("Summary: System 1 distillation (TPU) finished. Checkpoint at:", resolved)
        return resolved

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

    # Validate checkpoint after training
    resolved = find_checkpoint(out_dir)
    if resolved:
        print("Summary: System 1 distillation finished. Checkpoint at:", resolved)
        return resolved
    else:
        print(f"WARNING: EasyDistill exited successfully but no model files found in {out_dir}")
        print(f"Directory contents: {list(Path(out_dir).rglob('*'))[:20] if Path(out_dir).exists() else '(dir does not exist)'}")
        # Return the dir anyway so the user can inspect
        return str(Path(out_dir).resolve())


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

    # Check for existing checkpoint (resume support)
    existing = find_checkpoint(out_dir)
    if existing:
        print(f"Existing checkpoint found at {existing}. Skipping training (delete to retrain).")
        return existing

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

    # Validate checkpoint after training
    resolved = find_checkpoint(out_dir)
    if resolved:
        print("Summary: System 2 distillation finished. Checkpoint at:", resolved)
        return resolved
    else:
        print(f"WARNING: EasyDistill exited successfully but no model files found in {out_dir}")
        print(f"Directory contents: {list(Path(out_dir).rglob('*'))[:20] if Path(out_dir).exists() else '(dir does not exist)'}")
        return str(Path(out_dir).resolve())


# ---------------------------
# Student loading & inference
# ---------------------------

def load_student(model_path: str):
    """Load a distilled student model from a local checkpoint path.
    Falls back to CPU with float16/float32 if no GPU is available.
    Also uses find_checkpoint() to locate the actual model files.
    """
    # Try to find actual checkpoint if the path doesn't directly contain model files
    actual_path = find_checkpoint(model_path) or model_path
    if actual_path != model_path:
        print(f"Found model files at: {actual_path} (resolved from {model_path})")

    tokenizer = AutoTokenizer.from_pretrained(actual_path, use_fast=True)
    model = None
    # Try 1: auto device map (GPU if available)
    if torch.cuda.is_available():
        try:
            model = AutoModelForCausalLM.from_pretrained(actual_path, device_map="auto")
        except Exception as e:
            print(f"Auto device map failed: {e}")
    # Try 2: CPU with float16
    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                actual_path, torch_dtype=torch.float16, device_map="cpu",
                low_cpu_mem_usage=True,
            )
            print("Loaded student on CPU (float16).")
        except Exception as e:
            print(f"float16 CPU loading failed: {e}")
    # Try 3: CPU with float32
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            actual_path, torch_dtype=torch.float32, device_map="cpu",
            low_cpu_mem_usage=True,
        )
        print("Loaded student on CPU (float32).")
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
        is_hf_model = "/" in teacher_path and not Path(teacher_path).exists()
        is_local = Path(teacher_path).exists()
        if not is_hf_model and not is_local:
            print(f"Teacher model not found at {teacher_path} (skipping teacher)")
            teacher_path = None
        else:
            try:
                teacher_model, teacher_tokenizer = load_teacher(teacher_path)
                teacher_loaded = True
            except Exception as e:
                print(f"Could not load teacher '{teacher_path}': {e}")
                print("Continuing without teacher comparison.")
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


# ---------------------------
# Evaluation metrics
# ---------------------------

@torch.no_grad()
def evaluate_student(
    model,
    tokenizer,
    eval_items: List[Dict],
    mode: Literal["system1", "system2"] = "system1",
    max_new_tokens: int = 256,
    max_eval: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate a student model on a list of {instruction, input, output} items.
    Returns a dict with:
      - perplexity: average perplexity on reference outputs
      - bleu: corpus-level BLEU score
      - rouge_l: average ROUGE-L F1 score
      - num_evaluated: number of items evaluated
    """
    import math

    # Try to import evaluation libraries; provide helpful error if missing
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        print("NLTK not installed. Run: pip install nltk")
        return {"error": "nltk not installed"}
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("rouge-score not installed. Run: pip install rouge-score")
        return {"error": "rouge-score not installed"}

    items = eval_items[:max_eval]
    device = model.device

    # --- Perplexity ---
    total_loss = 0.0
    total_tokens = 0
    for item in items:
        text = format_prompt(item["instruction"], item.get("input", ""), mode=mode)
        full_text = text + " " + item.get("output", "")
        enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        labels = enc["input_ids"].clone()
        outputs = model(**enc, labels=labels)
        total_loss += outputs.loss.item() * labels.shape[1]
        total_tokens += labels.shape[1]

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

    # --- Generate predictions ---
    predictions = []
    references = []
    for item in items:
        pred = infer_student(model, tokenizer, item["instruction"], mode=mode, max_new_tokens=max_new_tokens)
        predictions.append(pred)
        references.append(item.get("output", ""))

    # --- BLEU ---
    ref_tokens = [[ref.split()] for ref in references]
    pred_tokens = [pred.split() for pred in predictions]
    smooth = SmoothingFunction().method1
    bleu = corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)

    # --- ROUGE-L ---
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)]
    avg_rouge_l = sum(rouge_scores) / max(len(rouge_scores), 1)

    results = {
        "perplexity": round(perplexity, 2),
        "bleu": round(bleu, 4),
        "rouge_l": round(avg_rouge_l, 4),
        "num_evaluated": len(items),
    }

    # Pretty-print
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Perplexity:    {results['perplexity']}")
    print(f"  BLEU:          {results['bleu']}")
    print(f"  ROUGE-L (F1):  {results['rouge_l']}")
    print(f"  Samples:       {results['num_evaluated']}")
    print("=" * 50)
    return results
