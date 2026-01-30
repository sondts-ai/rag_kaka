import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes
from src.base.llm_model import get_ollama_llm
#from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA
from src.chat.main import build_chat_chain



llm = get_ollama_llm(model_name="llama3.2")
# COLAB_URL = "https://48fdb4c47b72.ngrok-free.app"
# llm = get_ollama_llm(base_url=COLAB_URL, model_name="llama3.1")
genai_docs = "./data_source/generative_ai"

# --------- Chains----------------

genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

chat_chain = build_chat_chain(llm, 
                              history_folder="./chat_histories",
                              max_history_length=6)


# --------- App - FastAPI ----------------

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --------- Routes - FastAPI ----------------

@app.get("/check")
async def check():
    return {"status": "ok"}


@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}


# --------- Langserve Routes - Playground ----------------
add_routes(app, 
           genai_chain, 
           path="/generative_ai",
           input_type=InputQA,    # <--- Thêm dòng này (Truyền Class, KHÔNG có ngoặc đơn)
           output_type=OutputQA)
add_routes(app,
           chat_chain,
           path="/chat")