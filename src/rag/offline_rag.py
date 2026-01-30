import re
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate # <--- THÊM CÁI NÀY

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    def extract_answer(self, text_response: str,
                       pattern: str = r"Answer:\s*(.*)") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response
            
class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        
        # --- SỬA ĐOẠN NÀY: Tự tạo Prompt thay vì dùng hub ---
        # Đây là nội dung chuẩn của "rlm/rag-prompt"
        template = """Bạn là một trợ lý AI trung thực. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên đoạn văn bản ngữ cảnh được cung cấp bên dưới.
        
        Quy tắc bắt buộc:
        1. Chỉ trả lời dựa trên thông tin có trong "Ngữ cảnh".
        2. Nếu thông tin không có trong "Ngữ cảnh", hãy trả lời ngắn gọn: "Xin lỗi, tài liệu không cung cấp thông tin về vấn đề này."
        3. Không tự bịa ra thông tin sai lệch (như khoảng cách địa lý sai).
        
        Ngữ cảnh:
        {context}
        
        Câu hỏi: {question}
        
        Câu trả lời:"""
        
        # Chuyển string thành PromptTemplate object để dùng được dấu |
        self.prompt = PromptTemplate.from_template(template)
        # ----------------------------------------------------
        
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs, 
            "question": RunnablePassthrough()
        }
        
        # Bây giờ self.prompt là Object, nên phép | sẽ hoạt động
        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.str_parser
        )
        return rag_chain

    def format_docs(self, docs):
       
        print(f"\n[DEBUG] Số lượng đoạn văn tìm thấy: {len(docs)}")
        if len(docs) > 0:
            print("[DEBUG] Nội dung đoạn đầu tiên:")
            print("-" * 50)
            print(docs[0].page_content[:300]) # In thử 300 ký tự đầu
            print("-" * 50)
        else:
            print("[DEBUG] ⚠️ KHÔNG TÌM THẤY TÀI LIỆU NÀO KHỚP CÂU HỎI!")
        # ---------------------------
        
        return "\n\n".join(doc.page_content for doc in docs)