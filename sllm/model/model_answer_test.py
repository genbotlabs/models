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
vectorstore = FAISS.load_local("card", embedding_model, allow_dangerous_deserialization=True)

# ─── Midm 모델 로딩 ───────────────────────────────
midm_model_name = "K-intelligence/Midm-2.0-Base-Instruct"
midm_tokenizer = AutoTokenizer.from_pretrained(midm_model_name)
midm_model = AutoModelForCausalLM.from_pretrained(
    midm_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
midm_generation_config = GenerationConfig.from_pretrained(midm_model_name)

# ─── Midm 응답 생성 함수 ──────────────────────────
def extract_answer(decoded: str) -> str:
    """
    Midm 출력에서 실제 assistant 답변 부분만 추출
    """
    if "assistant\n" in decoded:
        return decoded.split("assistant\n")[-1].strip()
    elif "assistant:" in decoded:
        return decoded.split("assistant:")[-1].strip()
    elif "답변:" in decoded:
        return decoded.split("답변:")[-1].strip()
    else:
        return decoded.strip().splitlines()[-1]


def generate_with_midm(question: str, context: str) -> str:
    system_prompt = f"""
다음 규칙에 따라 사용자 질문에 정확하게 답변하세요:

    너는 사용자의 질문에 대해 한국어로 정확하게 답변하는 어시스턴트야.
     답변에는 'assistant:'나 'user:' 같은 단어를 절대 포함하지 마.
     문답 형식 없이 자연어로만 답변해.
     오직 질문에 대한 핵심 정보만 전달해.
     맺는말이나 인삿말 없이, 정중한 존댓말로 답해.
    """

    # user_prompt = f"질문: {question}\n위 문서들을 참고해서 질문에 답변해줘."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"문서: {context}\n질문: {question}\n위 문서를 참고해서 답변해줘."}
    ]

    prompt_text = midm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    input_ids = midm_tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")
    attention_mask = input_ids.ne(midm_tokenizer.pad_token_id)

    output = midm_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.8,
        top_p=0.75,
        top_k=20,
        pad_token_id=midm_tokenizer.pad_token_id,
        eos_token_id=midm_tokenizer.eos_token_id
    )

    decoded = midm_tokenizer.decode(output[0], skip_special_tokens=True)
    final_answer = decoded

    print("✅ 최종 답변:", final_answer)
    return final_answer





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

    return extract_answer(decoded)


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
    
    # ✅ 한 군데에서만 출력
    print("🤖 최종 LLM 응답:", answer)

    return {
        "question": state["question"],
        "retriever_docs": state["retriever_docs"],
        "score": state["score"],
        "answer": answer
    }



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

# ─── 지속적인 멀티턴 대화 루프 ───────────────────────────────
print("🧠 sLLM RAG 챗봇에 오신 걸 환영합니다! 'exit'을 입력하면 종료됩니다.\n")

conversation_history = []

while True:
    user_input = input("🙋 사용자 질문: ").strip()
    if user_input.lower() in {"exit", "quit", "종료"}:
        print("👋 대화를 종료합니다.")
        break

    # LangGraph 파이프라인 실행
    response = app_graph.invoke({
        "question": user_input,
        "answer": "",
        "score": 0.0,
        "retriever_docs": []
    })

    assistant_answer = response.get("answer", "")
    print(f"🤖 sLLM 응답: {assistant_answer}\n")

    # 필요하다면 history 저장
    conversation_history.append({
        "user": user_input,
        "assistant": assistant_answer
    })

