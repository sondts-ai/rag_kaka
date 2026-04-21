from typing import Union, Type
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- THÊM 3 DÒNG IMPORT NÀY CHO RE-RANKER ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class VectorDB:
    def __init__(self, documents=None, vector_db: Union[Chroma, FAISS] = Chroma,
        embedding=HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")) -> None:
        self.vector_db = vector_db
        self.embedding = embedding
        self.db = self._built_db(documents)

    def _built_db(self, documents):
        db = self.vector_db.from_documents(documents, embedding=self.embedding)
        return db
    
    # --- NÂNG CẤP HÀM GET_RETRIEVER ---
    def get_retriever(self, search_type: str = "similarity", search_kwargs: dict = {"k": 15}):
        """
        Chiến thuật Retrieve & Re-rank:
        1. Truy xuất thô 15 đoạn văn (tăng độ phủ, không bỏ sót).
        2. Dùng mô hình Cross-Encoder chấm điểm lại.
        3. Giữ lại đúng 3 đoạn có điểm khớp cao nhất đưa cho LLM.
        """
        # 1. Bộ tìm kiếm thô (Lấy 15 đoạn tốt nhất theo Vector)
        base_retriever = self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        # 2. Khởi tạo "Giám khảo" Reranker (Dùng model BAAI chuyên trị đa ngôn ngữ, cực nhạy tiếng Việt)
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        
        # Chỉ chắt lọc lại đúng 3 đoạn xuất sắc nhất
        compressor = CrossEncoderReranker(model=model, top_n=3)

        # 3. Gộp bộ tìm kiếm và giám khảo lại thành một cỗ máy duy nhất
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        
        return compression_retriever