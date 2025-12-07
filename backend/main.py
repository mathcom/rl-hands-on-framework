import os
import re
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RL Agent RAG Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 설정 상수 ---
# OLLAMA_HOST 환경변수를 사용하므로 base_url은 코드에서 제거해도 되지만, 
# 명시적인 참조를 위해 변수만 남겨둡니다.
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"
DATA_PATH = "/app/data"

# 전역 변수
retriever = None

def extract_from_constants_ts():
    """
    src/constants.ts 파일을 읽어 그 안에 하드코딩된
    코드 템플릿과 요구사항 문자열을 추출하여 Document로 변환합니다.
    """
    # [수정 1] 변수 초기화를 맨 위로 올림 (Scope 에러 방지)
    extracted_docs = []
    constants_path = os.path.join(DATA_PATH, "src/constants.ts")

    if not os.path.exists(constants_path):
        logger.warning(f"⚠️ constants.ts 파일을 찾을 수 없습니다: {constants_path}")
        return extracted_docs

    try:
        with open(constants_path, "r", encoding="utf-8") as f:
            content = f.read()

        # [수정 2] 새로운 상수 이름(LEVEL1_CODE 등)에 맞는 정규표현식 정의
        patterns = {
            "agent_tabular.py": r'export const LEVEL1_CODE = `([\s\S]*?)`;',
            "agent_dqn.py": r'export const LEVEL2_CODE = `([\s\S]*?)`;',
            "agent_ppo.py": r'export const LEVEL3_CODE = `([\s\S]*?)`;',
            "requirements.txt": r'export const REQUIREMENTS_TXT = `([\s\S]*?)`;',
            "run_guide.md": r'export const RUN_GUIDE_MD = `([\s\S]*?)`;'
        }

        for filename, regex in patterns.items():
            match = re.search(regex, content)
            if match:
                code_content = match.group(1)
                extracted_docs.append(Document(
                    page_content=code_content,
                    metadata={"source": f"{filename} (Virtual)"}
                ))
                logger.info(f"   ✅ constants.ts에서 '{filename}' 추출 완료")
            else:
                logger.warning(f"   ⚠️ '{filename}' 패턴을 찾지 못했습니다.")

    except Exception as e:
        logger.error(f"❌ constants.ts 파싱 중 오류: {e}")

    return extracted_docs

def load_and_index_data():
    global retriever
    logger.info("🔄 문서 로딩 및 인덱싱 시작...")

    documents = []

    # 1. 가상 파일 로드 (constants.ts 파싱)
    virtual_docs = extract_from_constants_ts()
    documents.extend(virtual_docs)

    # 2. 실제 물리 파일 로드 (README.md 등)
    physical_files = ["README.md"] # 필요한 파일 추가
    for relative_path in physical_files:
        full_path = os.path.join(DATA_PATH, relative_path)
        if os.path.exists(full_path):
            try:
                loader = TextLoader(full_path, encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"   ✅ 파일 로드 성공: {relative_path}")
            except Exception as e:
                logger.warning(f"   ⚠️ 로드 실패 ({relative_path}): {e}")

    if not documents:
        logger.error("❌ 학습할 데이터가 없습니다.")
        return

    # 3. 분할 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    # 4. 임베딩 & 저장 (base_url 제거 -> 환경변수 사용)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        collection_name="rl_class_codebase"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info("🎉 RAG 시스템 초기화 완료!")

@app.on_event("startup")
async def startup_event():
    try:
        load_and_index_data()
    except Exception as e:
        logger.error(f"❌ RAG 초기화 실패: {e}")

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG Not Ready")

    # LLM 초기화 (base_url 제거 -> 환경변수 사용)
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)

    template = """
    당신은 강화학습 실습 수업의 AI 조교입니다.
    아래 [Context]는 이 프로젝트의 소스코드와 문서입니다.
    이를 바탕으로 질문에 답변하세요.

    [Context]
    {context}

    [Question]
    {question}
    
    답변은 한국어로 작성하세요.
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response = rag_chain.invoke(request.message)
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))