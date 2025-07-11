from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# 환경 변수 설정
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")  

# FAISS 벡터 DB 불러오기
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
vectorstore = FAISS.load_local("card_QA_faiss_db", embedding_model,allow_dangerous_deserialization=True)


# 1. State 정의
class GraphState(TypedDict):
    question: Annotated[str, "질문"]
    answer: Annotated[str, "답변"]
    score: Annotated[float, "유사도 점수"]
    retriever_docs: Annotated[List[Document], "유사도 상위문서"]

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
    prompt = ChatPromptTemplate.from_template(
        """
        문서: {docs}
        질문: {question}
        위 문서들을 참고해서 질문에 답변해줘.
        """
    )
    docs_content = "\n---\n".join([doc.page_content for doc in state["retriever_docs"]])
    chain = prompt | ChatOpenAI(model="gpt-4.1-mini-2025-04-14")
    answer = chain.invoke({"docs": docs_content, "question": state["question"]}).content
    print("\n[llm_answer_node] 생성된 답변:", answer)
    return GraphState(answer=answer)

def query_rewrite_node(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """
        원본 질문: {question}
        위 질문의 핵심은 유지하면서, 유사 문서를 더 잘 찾을 수 있도록 질문을 다시 써줘.
        """
    )
    chain = prompt | ChatOpenAI(model="gpt-4.1-mini-2025-04-14")
    new_question = chain.invoke({"question": state["question"]}).content
    print("\n[query_rewrite_node] :", new_question)
    return GraphState(question=new_question)

# 3. 노드 분기 함수 정의
def decide_to_generate(state: GraphState) -> str:
    """
    문서의 유사도 점수에 따라 다음 노드를 결정
    - score가 0.23 이하면 'llm_answer'로 이동
    - score가 0.23 초과면 'query_rewrite'로 이동
    """
    if state["score"] <= 0.23:
        return "llm_answer"
    else:
        return "query_rewrite"


# 4. LangGraph 구성 및 연결
workflow = StateGraph(GraphState)
workflow.add_node("retriever", retriever_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("llm_answer", llm_answer_node)
workflow.add_node("query_rewrite", query_rewrite_node)
workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "llm_answer": "llm_answer",
        "query_rewrite": "query_rewrite",
    },
)
workflow.add_edge("query_rewrite", "retriever")

# 실행 
app = workflow.compile()

response = app.invoke({"question": "비번 변경", "answer": "", "score": 0.0, "retriever_docs": []})
print("\n[최종 답변]:", response["answer"])