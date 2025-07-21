import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# 모델 로딩
model_name = "K-intelligence/Midm-2.0-Base-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name)

# ✅ 테스트용 질문 리스트
questions = [
    "KT에 대해 소개해줘",
    "소상공인을 위한 지원 정책은 뭐가 있어?",
    "인터넷 요금제 변경은 어떻게 하나요?",
    "휴대폰 분실 시 대처 방법 알려줘",
    "KT 멤버십 혜택은 어떤 게 있어?"
]

# ✅ 반복적으로 inference 수행
for i, prompt in enumerate(questions):
    messages = [
        {"role": "system", 
         "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."},
        {"role": "user", "content": prompt}
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
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        do_sample=False,
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"\n🟡 질문 {i+1}: {prompt}")
    print(f"🟢 답변: {decoded_output}")
