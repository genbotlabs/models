import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer as DRTTokenizer

# ─── 1. 모델 및 토크나이저 로드 ─────────────────────────────
model_path = "./google_gemma-3-1B"
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# ─── 2. 평가 데이터셋 로드 ───────────────────────────────
eval_data = load_dataset("json", data_files="VS_random_child.jsonl", split="train")

# ─── 3. 전처리 함수 ───────────────────────────────
def preprocess(example):
    msgs = [{"role": "system", "content": "당신은 친절하고 정확한 고객상담 챗봇입니다."}]
    msgs += example["messages"]
    input_ids, labels = [], []

    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        context = "".join(
            ("<|user|>" if prev["role"] == "user" else "<|assistant|>")
            + prev["content"] + tokenizer.eos_token
            for prev in msgs[:i]
        )
        context += "<|assistant|>"
        response = m["content"] + tokenizer.eos_token
        ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
        rsp_ids = tokenizer(response, add_special_tokens=False).input_ids
        ids = ctx_ids + rsp_ids
        lbl = [-100] * len(ctx_ids) + rsp_ids
        input_ids.append(ids)
        labels.append(lbl)

    return {"input_ids": input_ids, "labels": labels}

# ─── 4. 전처리 및 데이터 준비 ───────────────────────────────
proc = eval_data.map(preprocess, batched=False, remove_columns=["messages"])
all_inputs = sum(proc["input_ids"], [])
all_labels = sum(proc["labels"], [])

# ─── 5. 배치 평가용 DataLoader 구성 ───────────────────────────────
BATCH_SIZE = 1
tokenized_inputs = tokenizer.pad({"input_ids": all_inputs}, padding=True, return_tensors="pt")
tokenized_labels = tokenizer.pad({"input_ids": all_labels}, padding=True, return_tensors="pt")

dataset = torch.utils.data.TensorDataset(tokenized_inputs["input_ids"], tokenized_labels["input_ids"])
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

preds_all, labels_all = [], []
with torch.no_grad():
    for input_ids, label_ids in tqdm(loader, desc="Running batches"):
        input_ids = input_ids.to("cuda")
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        preds = logits.argmax(dim=-1).cpu()
        preds_all.extend(preds)
        labels_all.extend(label_ids)

# ─── 6. 평가 지표 계산 ───────────────────────────────
# ⬛ Accuracy: 토큰 정확도
accuracy_metric = load("accuracy")
# ⬛ BERTScore: 의미 기반 유사도
bertscore_metric = load("bertscore")
# ⬛ BLEU: n-gram 기계번역 기반 정량 점수
bleu_metric = load("bleu")
# ⬛ ROUGE: 요약 평가용 longest overlap
rouge_metric = load("rouge")

preds_all = torch.stack(preds_all).tolist()
labels_all = torch.stack(labels_all).tolist()

# ── 마스킹 제거
flat_preds, flat_labels = [], []
for p, l in zip(preds_all, labels_all):
    for pi, li in zip(p, l):
        if li != -100:
            flat_preds.append(pi)
            flat_labels.append(li)

acc = accuracy_metric.compute(predictions=flat_preds, references=flat_labels)["accuracy"]

# ── 텍스트 디코딩
pred_strs = tokenizer.batch_decode(preds_all, skip_special_tokens=True)
label_strs = tokenizer.batch_decode(labels_all, skip_special_tokens=True)

bert = bertscore_metric.compute(predictions=pred_strs, references=label_strs, lang="ko")
bleu = bleu_metric.compute(
    predictions=[p.split() for p in pred_strs],
    references=[[l.split()] for l in label_strs]
)
rouge = rouge_metric.compute(predictions=pred_strs, references=label_strs)

# ─── 7. DialogRPT 평가 (사전학습된 대화 적절성 평가모델) ───────────────
dialogrpt_tokenizer = DRTTokenizer.from_pretrained("microsoft/DialogRPT-updown")
dialogrpt_model = AutoModelForSequenceClassification.from_pretrained("microsoft/DialogRPT-updown").to("cuda")
dialogrpt_model.eval()

scores = []
for ctx, rsp in tqdm(zip(label_strs, pred_strs), total=len(pred_strs), desc="DialogRPT"):
    text = ctx + " " + rsp
    enc = dialogrpt_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
    with torch.no_grad():
        logit = dialogrpt_model(**enc).logits.squeeze().item()
    scores.append(logit)

dialogrpt_avg = sum(scores) / len(scores)

# ─── 8. 결과 출력 ───────────────────────────────
print("\n📊 Evaluation Results")
print(f"DialogRPT (avg):        {dialogrpt_avg:.4f}   # 사전학습 기반 응답 자연스러움 평가")
print(f"Accuracy:               {acc:.4f}             # 토큰 단위 정확도")
print(f"BERTScore F1:           {sum(bert['f1']) / len(bert['f1']):.4f}   # 의미 기반 유사도")
print(f"BLEU:                   {bleu['bleu']:.4f}     # n-gram 정량 비교")
print(f"ROUGE-L:                {rouge['rougeL']:.4f}  # 요약 평가 기반")
