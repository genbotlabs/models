import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ─── 모델 불러오기 ─────────────────────────────
model_name = "K-intelligence/Midm-2.0-Base-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name)

# ─── Vector DB 로드 (FAISS) ─────────────────────────────
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
vectorstore = FAISS.load_local("./vectordb", embedding_model, allow_dangerous_deserialization=True)

# ─── 대화 루프 시작 ─────────────────────────────
print("💬 VectorDB 기반 Midm 챗봇입니다. 종료하려면 'exit' 또는 'quit' 입력.\n")

chat_history = []

while True:
    user_input = input("🙋 사용자: ")

    if user_input.lower() in ["exit", "quit"]:
        print("👋 대화를 종료합니다.")
        break

    # 문자열로 안전하게 변환 + 공백 제거
    query = str(user_input).strip()

    # 빈 입력 방지
    if not query:
        print("❗ 유효한 질문을 입력해 주세요.\n")
        continue

    # 🔍 유사 문서 검색 (예외 방지)
    try:
        results = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in results]) if results else "관련 정보를 찾지 못했습니다."
    except Exception as e:
        print(f"❌ 문서 검색 중 오류 발생: {e}")
        continue

    # 시스템 메시지에 context 삽입
    context_message = {
        "role": "system",
        "content": f"""
1. **친절한 인사**  
   - 사용자가 처음 질문할 때는 “안녕하세요! 무엇을 도와드릴까요?” 등으로 시작하세요.

2. **명확한 이해 확인**  
   - 질문의 의도가 모호하면, “죄송하지만 조금 더 구체적으로 어떤 정보를 찾으시는지 알려주실 수 있을까요?”라고 여쭤보세요.

3. **한글·정중체 유지**  
   - 존댓말을 사용하고 한국어로만 답변해줘.

4. **추가 안내**  
   - 답변 후 “더 궁금한 점이 있으시면 언제든 질문해 주세요.”와 같이 대화를 이어갈 여지를 남기세요.

5. **문서 기반 응답**  
   - 아래 문서 내용을 참고해서 정확하게 답변하세요.

[참고 정보]
{context}
"""
    }

    # 메시지 구성
    current_messages = [context_message, {"role": "user", "content": query}]

    input_ids = tokenizer.apply_chat_template(
        current_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # 모델 응답 생성
    output = model.generate(
        input_ids,
        generation_config=generation_config,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        do_sample=False,
    )

    # 응답 후처리
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    assistant_reply = response_text.split("<|assistant|>\n")[-1].strip()

    print(f"🤖 Mi:dm: {assistant_reply}\n")

    # 대화 기록 저장
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": assistant_reply})
