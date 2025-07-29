# rag_api.py

from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ 변경
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os

# ─── 환경변수 로드 ─────────────────────────────────
load_dotenv()

# ─── FAISS 벡터DB 및 임베딩 로딩 ─────────────────
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
vectorstore = FAISS.load_local("card_QA_faiss_db", embedding_model, allow_dangerous_deserialization=True)

# ─── Midm 모델 로딩 ───────────────────────────────
midm_model_name = "kakaocorp/kanana-1.5-8b-instruct-2505"
midm_tokenizer = AutoTokenizer.from_pretrained(midm_model_name)
midm_model = AutoModelForCausalLM.from_pretrained(
    midm_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
midm_generation_config = GenerationConfig.from_pretrained(midm_model_name)

# ─── Midm 응답 생성 함수 ──────────────────────────
def generate_with_midm(question: str, context: str) -> str:
    system_prompt = """
1. **친절한 인사**  
   - 사용자가 처음 질문할 때는 “안녕하세요! 무엇을 도와드릴까요?” 등으로 시작하세요.

2. **명확한 이해 확인**  
   - 질문의 의도가 모호하면, “죄송하지만 조금 더 구체적으로 어떤 정보를 찾으시는지 알려주실 수 있을까요?”라고 여쭤보세요.

3. **한글·정중체 유지**  
   - 존댓말을 사용하고 한국어로만 답변해줘.

4. **추가 안내**  
   - 답변 후 “더 궁금한 점이 있으시면 언제든 질문해 주세요.”와 같이 대화를 이어갈 여지를 남기세요.

"""

    user_prompt = f"문서: {context}\n질문: {question}\n위 문서들을 참고해서 질문에 답변해줘."

    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
    ]

    # ✅ 입력 텐서를 생성하고 GPU 또는 CPU로 이동
    input_ids = midm_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # ✅ attention_mask와 pad_token_id 설정 (경고 방지 및 안정적 출력)
    attention_mask = input_ids.ne(midm_tokenizer.pad_token_id)

    output = midm_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,                    # 🔧 추가
        max_new_tokens=1024,
        do_sample=True,                                   # 🔧 샘플링 켜기
        temperature=0.8,
        top_p=0.75,
        top_k=20,
        pad_token_id=midm_tokenizer.pad_token_id,         # 🔧 추가
        eos_token_id=midm_tokenizer.eos_token_id
    )

    decoded = midm_tokenizer.decode(output[0], skip_special_tokens=True)

    # ✅ 디버깅용 출력
    print("🔍 Decoded output:", decoded)

    if not decoded.strip():
        print("❌ 모델이 빈 응답을 생성했습니다. Prompt 확인 필요")

    return decoded


# ─── LangGraph State 정의 ─────────────────────────
class GraphState(TypedDict):
    question: Annotated[str, "질문"]
    answer: Annotated[str, "답변"]
    score: Annotated[float, "유사도 점수"]
    retriever_docs: Annotated[List[Document], "유사도 상위문서"]

# ─── LangGraph 노드 정의 ─────────────────────────
# def retriever_node(state: GraphState) -> GraphState:
#     docs = vectorstore.similarity_search_with_score(state["question"], k=3)
#     retrieved_docs = [doc for doc, _ in docs]
#     score = docs[0][1]
#     print("[question]:", state["question"])
#     print("[retriever_node] 유사도:", score)
#     return {
#         "question": state["question"],
#         "retriever_docs": retrieved_docs,
#         "score": score,
#         "answer": ""
#     }

# 2. 노드 정의
def retriever_node(state: GraphState) -> GraphState:
    docs = vectorstore.similarity_search_with_score(state["question"], k=3)
    retrieved_docs = [doc for doc, _ in docs]
    score = docs[0][1]
    # print("\n[retriever_node] 문서:", [doc.page_content for doc in retrieved_docs])
    print("[queston]:", state["question"])
    print("[retriever_node] 상위 문서의 제목:\n",docs)
    print("[retriever_node] 유사도 점수:", score)
    # print('-'*100)
    # print(docs)
    # print('-'*100)
    # print(retrieved_docs[0])
    return GraphState(score=score, retriever_docs=retrieved_docs)



def grade_documents_node(state: GraphState) -> GraphState:
    return GraphState()

def llm_answer_node(state: GraphState) -> GraphState:
    docs_content = "\n---\n".join([doc.page_content for doc in state["retriever_docs"]])
    answer = generate_with_midm(state["question"], docs_content)
    print("[llm_answer_node] 생성된 답변:", answer)
    return GraphState(answer=answer)


def query_rewrite_node(state: GraphState) -> GraphState:
    prompt = f"질문: {state['question']}\n위 질문의 핵심은 유지하면서, 유사 문서를 더 잘 찾을 수 있도록 질문을 다시 써줘."
    messages = [
        # {"role": "system", "content": "당신은 질문을 리라이팅해주는 도우미입니다."},
        {"role": "user", "content": prompt}
    ]

    input_ids = midm_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    output = midm_model.generate(
        input_ids,
        generation_config=midm_generation_config,
        max_new_tokens=512,
        top_p=0.75,
        do_sample=False,
        eos_token_id=midm_tokenizer.eos_token_id
    )

    new_question = midm_tokenizer.decode(output[0], skip_special_tokens=True)
    print("[query_rewrite_node] 리라이팅:", new_question)
    return {
        "question": new_question,
        "retriever_docs": [],
        "score": 0.0,
        "answer": ""
    }

def decide_to_generate(state: GraphState) -> str:
    return "llm_answer" if state["score"] <= 0.5 else "query_rewrite"

# ─── LangGraph 구성 ───────────────────────────────
workflow = StateGraph(GraphState)
workflow.add_node("retriever", retriever_node)
workflow.add_node("llm_answer", llm_answer_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("query_rewrite", query_rewrite_node)
workflow.set_entry_point("retriever")
workflow.add_conditional_edges("retriever", decide_to_generate, {
    "llm_answer": "llm_answer",
    "query_rewrite": "query_rewrite"
})
workflow.add_edge("query_rewrite", "retriever")
app_graph = workflow.compile()

response = app_graph.invoke({"question": "연회비 청구기준 알려줘 ", "answer": "", "score": 0.0, "retriever_docs": []})
print("\n[최종 답변]:", response["answer"])

