# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

app = FastAPI()

# 1. 모델 로드 (최초 1회만)
model_name = "K-intelligence/Midm-2.0-Base-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name)


# 2. 요청 형식 정의
class ChatRequest(BaseModel):
    question: str


# 3. 채팅 엔드포인트
@app.post("/chat")
def chat(req: ChatRequest):
    user_prompt = req.question

    messages = [
        {"role": "system", "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."},
        {"role": "user", "content": user_prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    output = model.generate(
        input_ids,
        generation_config=generation_config,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"answer": response}
