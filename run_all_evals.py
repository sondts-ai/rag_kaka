import os
import json
import pandas as pd
from datasets import Dataset

# Import LLM, Loader
from src.base.llm_model import get_ollama_llm
from src.rag.file_loader import Loader

# Import 2 hệ thống RAG
from src.rag.offline_rag import Offline_RAG as Naive_RAG
from src.rag.crag import Offline_RAG as CRAG

# --- ĐÃ SỬA: Import 2 chiến lược tìm kiếm (Thô và Tinh) ---
from src.rag.vectorstore2 import VectorDB as VectorDB_Similarity # File chỉ dùng Similarity (K=3)
from src.rag.vectorstore import VectorDB as VectorDB_Rerank    # File đã cấu hình Similarity (K=15) + BAAI Reranker (Top 3)

# Import Ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatResult

# ======================================================================
# LỚP VÁ LỖI CHO DEEPSEEK (Ép n=1)
# ======================================================================
class SafeDeepSeekChat(ChatOpenAI):
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        n_requested = kwargs.pop('n', 1) 
        kwargs['n'] = 1        
        if n_requested > 1:
            generations = []
            for _ in range(n_requested):
                res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                generations.extend(res.generations)
            return ChatResult(generations=generations)
        else:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        n_requested = kwargs.pop('n', 1)
        kwargs['n'] = 1
        if n_requested > 1:
            generations = []
            for _ in range(n_requested):
                res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                generations.extend(res.generations)
            return ChatResult(generations=generations)
        else:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

# ======================================================================
# CHƯƠNG TRÌNH CHÍNH
# ======================================================================
if __name__ == "__main__":
    # 1. Khởi tạo model và VectorDB
    print("Khởi tạo model và VectorDB...")
    # Vì đang chạy trên V100, bạn dùng qwen2.5:7b sẽ cho câu trả lời mượt hơn rất nhiều so với 1.5b
    llm = get_ollama_llm(model_name="qwen2.5:7b") 
    doc_loaded = Loader().load_dir("./data_source/generative_ai", workers=2)
    
    # --- ĐÃ SỬA: Khởi tạo 2 loại Retriever ---
    # Retriever 1: Thuần Similarity, lấy luôn K=3 (File vectorstore2.py)
    retriever_sim = VectorDB_Similarity(documents=doc_loaded).get_retriever(search_kwargs={"k": 3})
    
    # Retriever 2: Gọi thẳng hàm đã nâng cấp Re-rank, bên trong file bạn đã set sẵn k=15 và Top=3 rồi (File vectorstore.py)
    retriever_rerank = VectorDB_Rerank(documents=doc_loaded).get_retriever()

    # Khởi tạo 4 cấu hình RAG để thi đấu
    chains = {
        "Naive_RAG_Similarity": Naive_RAG(llm).get_chain(retriever_sim),
        "Naive_RAG_Rerank": Naive_RAG(llm).get_chain(retriever_rerank),
        "CRAG_Similarity": CRAG(llm).get_chain(retriever_sim),
        "CRAG_Rerank": CRAG(llm).get_chain(retriever_rerank)
    }

    # 2. Đọc dữ liệu thi
    with open(r"test100.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_questions = test_data["question"]
    ground_truths = test_data["ground_truth"]

    # 3. Cấu hình Giám khảo DeepSeek
    os.environ["DEEPSEEK_API_KEY"] = "sk-24d7353b773b4653b642a0300fccaaeb" # Nhớ thay API của bạn nếu cần
    evaluator_llm = SafeDeepSeekChat(model="deepseek-chat", api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com", temperature=0)
    ragas_llm = LangchainLLMWrapper(evaluator_llm)
    # Ragas cần embedding để tính answer_relevancy
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")) 
    safe_config = RunConfig(max_workers=1, max_retries=3)

    # ĐỂ TRÁNH LỖI DEEPSEEK, CHIA METRICS LÀM 2 LÔ (BATCHES)
    metrics_batch_1 = [faithfulness, context_precision]
    metrics_batch_2 = [answer_relevancy, context_recall]

    all_results_df = []

    # 4. Cho từng thí sinh thi
    for chain_name, chain in chains.items():
        print(f"\n{'='*50}\nĐANG CHẠY KỊCH BẢN: {chain_name}\n{'='*50}")
        answers, contexts = [], []
        
        # Xác định dùng retriever nào để lấy đúng context cho Ragas chấm
        current_retriever = retriever_rerank if "Rerank" in chain_name else retriever_sim

        # Xin câu trả lời từ RAG
        for q in test_questions:
            ans = chain.invoke({"question": q})
            answers.append(ans)
            docs = current_retriever.invoke(q)
            contexts.append([doc.page_content for doc in docs])

        # Đóng gói Dataset
        dataset = Dataset.from_dict({
            "question": test_questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

        # --- CHẤM ĐIỂM ĐỢT 1 ---
        print(f"[{chain_name}] Đang nhờ DeepSeek chấm ĐỢT 1 (faithfulness, context_precision)...")
        res_batch_1 = evaluate(
            dataset=dataset, metrics=metrics_batch_1, llm=ragas_llm, embeddings=ragas_embeddings, run_config=safe_config, raise_exceptions=False
        )
        df_batch_1 = res_batch_1.to_pandas()

        # --- CHẤM ĐIỂM ĐỢT 2 ---
        print(f"[{chain_name}] Đang nhờ DeepSeek chấm ĐỢT 2 (answer_relevancy, context_recall)...")
        res_batch_2 = evaluate(
            dataset=dataset, metrics=metrics_batch_2, llm=ragas_llm, embeddings=ragas_embeddings, run_config=safe_config, raise_exceptions=False
        )
        df_batch_2 = res_batch_2.to_pandas()

        # --- GỘP HAI KẾT QUẢ ---
        df_merged = df_batch_1.copy()
        
        # Lấy trực tiếp cột điểm của Đợt 2 đắp sang Đợt 1 an toàn
        if 'answer_relevancy' in df_batch_2.columns:
            df_merged['answer_relevancy'] = df_batch_2['answer_relevancy']
        if 'context_recall' in df_batch_2.columns:
            df_merged['context_recall'] = df_batch_2['context_recall']

        # Gắn tên chiến lược (Experiment) để phân biệt sau này
        df_merged['Experiment'] = chain_name
        all_results_df.append(df_merged)

        # In kết quả trung bình
        print(f"\n=> KẾT QUẢ TỔNG HỢP CỦA {chain_name}:")
        cols_to_mean = [col for col in ['faithfulness', 'context_precision', 'answer_relevancy', 'context_recall'] if col in df_merged.columns]
        print(df_merged[cols_to_mean].mean())

    # 5. Lưu toàn bộ file CSV
    final_df = pd.concat(all_results_df, ignore_index=True)
    final_df.to_csv("rag_comparison_results.csv", index=False, encoding="utf-8-sig")
    print("\n✅ HOÀN TẤT! Đã lưu toàn bộ kết quả của 4 hệ thống vào 'rag_comparison_results.csv'.")