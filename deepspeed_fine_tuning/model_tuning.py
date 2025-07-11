# -*- coding: utf-8 -*-
import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from torch.distributed import destroy_process_group
import evaluate

# — Metric instances
accuracy_metric  = evaluate.load("accuracy")
bertscore_metric = evaluate.load("bertscore")
rouge_metric     = evaluate.load("rouge")

# Helper to check if this is the master process (rank 0)
def is_master():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def main():
    # — WANDB 설정 (master에서만)
    if is_master():
        wandb.login()
        wandb.init(project="solar_project", name="solar10_7B")

    # — 메모리 단편화 완화
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # — 4bit quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # — load base model & LoRA setup
    model_name = "upstage/SOLAR-10.7B-Instruct-v1.0"
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=None
    )
    print(model_name)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(base, lora_cfg).to("cuda")
    model.config.use_cache = False
    model.train()

    # — tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # — dataset load & preprocess
    raw = load_dataset(
        "json",
        data_files={"train": "card_consult_finetune_messages.jsonl"},
        split="train"
    )
    def preprocess(example):
        msgs = [{"role": "system", "content": "당신은 친절하고 정확한 고객상담 챗봇입니다."}] + example["messages"]
        input_ids, labels = [], []
        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue
            ctx = ""
            for prev in msgs[:i]:
                tag = "<|user|>" if prev["role"] == "user" else "<|assistant|>"
                ctx += tag + prev["content"] + tokenizer.eos_token
            resp = msg["content"] + tokenizer.eos_token
            in_ids = tokenizer(ctx + resp, add_special_tokens=False).input_ids
            lab_ids = [-100] * tokenizer(ctx, add_special_tokens=False).input_ids.__len__() + tokenizer(resp, add_special_tokens=False).input_ids
            input_ids.append(in_ids)
            labels.append(lab_ids)
        return {"input_ids": input_ids, "labels": labels}

    proc = raw.map(preprocess, batched=False, remove_columns=["messages"])
    all_in  = sum(proc["input_ids"], [])
    all_lbl = sum(proc["labels"], [])

    raw_dataset = Dataset.from_dict({"input_ids": all_in, "labels": all_lbl})
    splits = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_ds, valid_ds = splits["train"], splits["test"]

    # — collate_fn
    def collate_fn(batch):
        return tokenizer.pad(
            {"input_ids": [b["input_ids"] for b in batch],
             "labels":    [b["labels"]    for b in batch]},
            padding=True,
            return_tensors="pt"
        )

    # — compute_metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        mask  = labels != -100

        acc = accuracy_metric.compute(
            predictions=preds[mask].flatten().tolist(),
            references=labels[mask].flatten().tolist()
        )['accuracy']

        pred_strs  = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bert = bertscore_metric.compute(
            predictions=pred_strs,
            references=label_strs,
            lang="ko"
        )
        bert_f1 = sum(bert['f1']) / len(bert['f1'])

        rouge = rouge_metric.compute(predictions=pred_strs, references=label_strs)

        return {
            'accuracy': acc,
            'bertscore_f1': bert_f1,
            'rougeL': rouge['rougeL'].mid.fmeasure,
        }

    # — training arguments
    training_args = TrainingArguments(
        output_dir="./solar10_7B",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=3e-4,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        report_to="wandb" if is_master() else [],
        run_name="solar10_7B" if is_master() else None,
        logging_dir="./logs",
        remove_unused_columns=False,
        deepspeed="ds_config.json",
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    torch.cuda.empty_cache()

    # — train with forced checkpoint on interruption
    try:
        trainer.train()
    except Exception as e:
        if is_master():
            print(f"❌ Training error: {e}")
        raise
    finally:
        if is_master():
            # 강제 체크포인트 저장
            #torch.save(model.module.state_dict(), "solar10_7b_fallback_state_dict.pth")
            model.save_pretrained("./solar10_7B_interrupted_test")
            tokenizer.save_pretrained("./solar10_7B_interrupted_test")
            print("✅ Fallback checkpoint saved at interruption.")

    # — 분산 학습 종료
    if torch.distributed.is_initialized():
        destroy_process_group()

    return model, tokenizer

if __name__ == "__main__":
    main()
