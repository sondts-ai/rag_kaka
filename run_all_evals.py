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

# Import 2 chiến lược tìm kiếm
from src.rag.vectorstore import VectorDB as VectorDB_MMR
from src.rag.vectorstore2 import VectorDB as VectorDB_Similarity

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
    # 1. Khởi tạo LLM và VectorDB
    print("Khởi tạo model và VectorDB...")
    llm = get_ollama_llm(model_name="qwen2.5:7b")
    doc_loaded = Loader().load_dir("./data_source/generative_ai", workers=2)
    
    retriever_mmr = VectorDB_MMR(documents=doc_loaded).get_retriever(search_kwargs={"k": 3})
    retriever_sim = VectorDB_Similarity(documents=doc_loaded).get_retriever(search_kwargs={"k": 3})

    # Khởi tạo 4 cấu hình RAG để thi đấu
    chains = {
        "Naive_RAG_Similarity": Naive_RAG(llm).get_chain(retriever_sim),
        "Naive_RAG_MMR": Naive_RAG(llm).get_chain(retriever_mmr),
        "CRAG_Similarity": CRAG(llm).get_chain(retriever_sim),
        "CRAG_MMR": CRAG(llm).get_chain(retriever_mmr)
    }

    # 2. Đọc dữ liệu thi
    with open(r"test100.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_questions = test_data["question"]
    ground_truths = test_data["ground_truth"]

    # 3. Cấu hình Giám khảo DeepSeek
    os.environ["DEEPSEEK_API_KEY"] = "sk-0f2090983b3c492588c88bd91e7c0b57" # Nhớ thay API của bạn
    evaluator_llm = SafeDeepSeekChat(model="deepseek-chat", api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com", temperature=0)
    ragas_llm = LangchainLLMWrapper(evaluator_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert"))
    
    # 🚀 ĐÃ SỬA: Tăng số luồng gọi API DeepSeek lên 8 (nhanh gấp 8 lần)
    safe_config = RunConfig(max_workers=4, max_retries=2, timeout=90)

    # ĐỂ TRÁNH LỖI DEEPSEEK, CHIA METRICS LÀM 2 LÔ (BATCHES)
    metrics_batch_1 = [faithfulness, context_precision]
    metrics_batch_2 = [answer_relevancy, context_recall]

    all_results_df = []

    # 4. Cho từng thí sinh thi
    for chain_name, chain in chains.items():
        print(f"\n{'='*50}\nĐANG CHẠY KỊCH BẢN: {chain_name}\n{'='*50}")
        
        # Xác định dùng retriever nào để lấy đúng context cho log
        current_retriever = retriever_mmr if "MMR" in chain_name else retriever_sim

        # 🚀 ĐÃ SỬA: Xử lý song song bằng cơ chế batch() thay vì vòng lặp for
        print(f"[{chain_name}] Đang sinh câu trả lời (chạy batch song song)...")
        inputs = [{"question": q} for q in test_questions]
        concurrency = 2 if "CRAG" in chain_name else 4
        answers = chain.batch(inputs, config={"max_concurrency": concurrency})
        
        print(f"[{chain_name}] Đang truy xuất tài liệu (chạy batch song song)...")
        docs_list = current_retriever.batch(test_questions, config={"max_concurrency": 4})
        contexts = [[doc.page_content for doc in docs] for docs in docs_list]

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

        # --- GỘP HAI KẾT QUẢ LẠI BẰNG PANDAS ---
        df_merged = df_batch_1.copy()
        
        print(f"[{chain_name}] Các cột trả về ở Đợt 2: {df_batch_2.columns.tolist()}")

        if 'answer_relevancy' in df_batch_2.columns:
            df_merged['answer_relevancy'] = df_batch_2['answer_relevancy']
        else:
            df_merged['answer_relevancy'] = None
            print(f"⚠️ Cảnh báo: Không tìm thấy 'answer_relevancy' trong Đợt 2 của {chain_name}")
            
        if 'context_recall' in df_batch_2.columns:
            df_merged['context_recall'] = df_batch_2['context_recall']
        else:
            df_merged['context_recall'] = None
            print(f"⚠️ Cảnh báo: Không tìm thấy 'context_recall' trong Đợt 2 của {chain_name}")

        # Gắn tên chiến lược (Experiment) để phân biệt sau này
        df_merged['Experiment'] = chain_name
        all_results_df.append(df_merged)

        # In kết quả trung bình của mô hình này
        print(f"\n=> KẾT QUẢ TỔNG HỢP CỦA {chain_name}:")
        cols_to_mean = [col for col in ['faithfulness', 'context_precision', 'answer_relevancy', 'context_recall'] if col in df_merged.columns]
        print(df_merged[cols_to_mean].mean())

    # 5. Lưu toàn bộ file Excel/CSV
    final_df = pd.concat(all_results_df, ignore_index=True)
    final_df.to_csv("rag_comparison_results.csv", index=False, encoding="utf-8-sig")
    print("\n✅ HOÀN TẤT! Đã lưu toàn bộ kết quả của 4 hệ thống vào 'rag_comparison_results.csv'.")