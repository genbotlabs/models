# train_and_evaluate_qolora.py

import os
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training  # 🔧 변경됨
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, Repository, HfFolder

# ─── 0. 환경 설정 및 W&B 초기화 ─────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
wandb.login()
wandb.init(
    project="google_gemma-3-1B_project",
    name="google_gemma-3-1B"
)

# ─── 1. JSONL 파일 로드 ─────────────────────────────────
datasets = load_dataset(
    "json",
    data_files={
        "train": "TS_random_child.jsonl",
        "valid": "VS_random_child.jsonl"
    }
)
train_ds = datasets["train"]
valid_ds = datasets["valid"]

# ─── 2. 모델 로드: 4bit 양자화 + LoRA(QLoRA) ─────────────────────────
base_model = "google/gemma-3-1b-it"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
base.config.use_cache = False
base.gradient_checkpointing_enable()

# 🔧 핵심: LoRA 적용 전 양자화 모델 준비
base = prepare_model_for_kbit_training(base)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base, lora_cfg).to("cuda")
model.config.use_cache = False
model.print_trainable_parameters()

# ─── 3. 토크나이저 및 데이터 콜레이터 설정 ─────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
data_collator = default_data_collator

# ─── 4. 전처리 함수 정의 ─────────────────────────────────
def preprocess(example):
    msgs = [{"role": "system", "content": "당신은 친절하고 정확한 고객상담 챗봇입니다."}] + example["messages"]
    input_ids_list, labels_list = [], []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        context = "".join(
            ("<|user|>" if prev["role"] == "user" else "<|assistant|>")
            + prev["content"] + tokenizer.eos_token
            for prev in msgs[:i]
        )
        reply = m["content"] + tokenizer.eos_token
        ids = tokenizer(context + reply, add_special_tokens=False).input_ids
        lbl = (
            [-100] * len(tokenizer(context, add_special_tokens=False).input_ids)
            + tokenizer(reply, add_special_tokens=False).input_ids
        )
        input_ids_list.append(ids)
        labels_list.append(lbl)
    return {"input_ids": input_ids_list, "labels": labels_list}

# ─── 5. 데이터셋 전처리 및 플래트닝 ─────────────────────────────────
train_proc = train_ds.map(preprocess, batched=False, remove_columns=["messages"])
valid_proc = valid_ds.map(preprocess, batched=False, remove_columns=["messages"])
all_in = sum(train_proc["input_ids"], [])
all_lbl = sum(train_proc["labels"], [])
train_flat = Dataset.from_dict({"input_ids": all_in, "labels": all_lbl})

# ─── 6. QLoRA 전용 Trainer 정의 ─────────────────────────────────
class QLoRATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# ─── 7. Trainer 초기화 ─────────────────────────────────
training_args = TrainingArguments(
    output_dir="./google_gemma-3-1B",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    num_train_epochs=2,
    learning_rate=3e-4,
    bf16=True,
    logging_steps=50,
    save_steps=300,
    save_total_limit=3,
    run_name="google_gemma-3-1B",
    report_to="wandb"
)

trainer = QLoRATrainer(
    model=model,
    args=training_args,
    train_dataset=train_flat,
    eval_dataset=valid_proc,
    data_collator=data_collator
)

# ─── 8. 학습 실행 및 저장 ─────────────────────────────────
trainer.train()

local_dir = "./google_gemma-3-1B"
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)
print("저장 완료")
