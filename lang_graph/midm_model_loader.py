# model_loader.py 또는 sllm_model.py

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
import torch
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import cohere
# 1. LLM 허브 모델 ID
hub_model_id = "seoungji/sllm_midm_model"

# 2. 임베딩 모델
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# 3. FAISS 벡터스토어 로드
vectorstore = FAISS.load_local("abab", embedding_model,allow_dangerous_deserialization=True)

COHERE_API_KEY = os.environ["COHERE_API_KEY"]
co = cohere.Client(COHERE_API_KEY)
# 전체 문서 (BM25용)
all_docs = vectorstore.docstore._dict.values()

# BM25 retriever
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 5

# faiss retriever
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})



# Hybrid (ensemble) retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)


# 4. LLM 로딩 함수
_tokenizer = None
_model = None
_generation_config = None

def get_midm_model():
    global _tokenizer, _model, _generation_config
    if _tokenizer is None or _model is None:
        print("🔄 모델 로딩 중...")
        _tokenizer = AutoTokenizer.from_pretrained(
            hub_model_id,
            trust_remote_code=True,
            use_fast=False
        )
        _model = AutoModelForCausalLM.from_pretrained(
            hub_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        try:
            _generation_config = GenerationConfig.from_pretrained(hub_model_id)
        except:
            _generation_config = GenerationConfig()
        print("✅ 모델 로딩 완료")
    return _tokenizer, _model, _generation_config
