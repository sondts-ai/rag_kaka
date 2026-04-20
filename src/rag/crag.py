# import re
# from typing import List, TypedDict
# from langchain_core.runnables import RunnableLambda
# from langchain_core.prompts import PromptTemplate
# from langgraph.graph import END, StateGraph

# # 1. MÁY CHÉM VĂN PHONG (Giữ nguyên của bạn để dọn dẹp rác từ LLM nhỏ)
# def may_chem_van_phong(text: str) -> str:
#     # 1. Dọn dẹp thẻ HTML và ký tự đặc biệt của mô hình
#     clean_text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    
#     # 2. XỬ LÝ LỖI HIỆN \n: Biến chuỗi "\\n" (2 ký tự) thành dấu xuống dòng thật (\n)
#     # Điều này giải quyết việc thỉnh thoảng Bot trả về text chứa chữ \n thay vì xuống dòng
#     clean_text = clean_text.replace('\\n', '\n')
    
#     # 3. Xóa sạch mọi ký tự Tiếng Trung/Nhật/Hàn (giữ nguyên logic của bạn)
#     clean_text = re.sub(r'[\u4e00-\u9fff]+', '', clean_text).strip()
    
#     # 4. QUÉT SẠCH LANH CHANH (Kể cả khi web bị stream)
#     pattern = r"^(Đúng rồi.*?|Bạn nói đúng.*?|Đúng vậy.*?|Chính xác.*?|Câu hỏi.*?|Dạ đúng.*?)(?:,|\.|!|\n|\s)+"
#     while re.match(pattern, clean_text, re.IGNORECASE):
#         clean_text = re.sub(pattern, "", clean_text, count=1, flags=re.IGNORECASE).strip()
    
#     # 5. Gọt nốt cái đuôi ngáo ngơ (giữ nguyên logic của bạn)
#     duoi_ngao_ngo = "không cần trả lời thêm bất kỳ thông tin nào khác"
#     if duoi_ngao_ngo in clean_text.lower():
#         clean_text = re.sub(r'(?i)[.,]?\s*' + duoi_ngao_ngo + r'.*$', '.', clean_text).strip()

#     # 6. DỌN DẸP XUỐNG DÒNG THỪA: 
#     # Thay thế 3 dấu xuống dòng trở lên bằng 2 dấu xuống dòng để văn bản gọn gàng
#     clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()

#     # 7. Viết hoa chữ cái đầu tiên
#     if len(clean_text) > 0:
#         clean_text = clean_text[0].upper() + clean_text[1:]
        
#     return clean_text

# # 2. KHAI BÁO TRẠNG THÁI (STATE) CỦA ĐỒ THỊ CRAG
# class GraphState(TypedDict):
#     question: str
#     documents: List[any]
#     generation: str
#     fallback: bool


# # 3. CLASS HỆ THỐNG CRAG
# class Offline_RAG:
#     def __init__(self, llm) -> None:
#         self.llm = llm

#     def get_chain(self, retriever):
        
#         # ==========================================
#         # KHAI BÁO CÁC TRẠM XỬ LÝ (NODES) TRONG CRAG
#         # ==========================================
        
#         def retrieve_node(state: GraphState):
#             """Trạm 1: Tìm kiếm tài liệu từ VectorDB"""
#             question = state["question"]
#             print(f"\n[CRAG] 🔎 TRẠM 1: Tìm kiếm tài liệu cho câu hỏi: '{question}'")
#             docs = retriever.invoke(question)
#             return {"documents": docs, "question": question}

#         def grade_documents_node(state: GraphState):
#             """Trạm 2: LLM làm Giám khảo đánh giá tài liệu (Corrective)"""
#             print("[CRAG] ⚖️ TRẠM 2: Đánh giá độ chính xác của tài liệu...")
#             question = state["question"]
#             documents = state["documents"]
            
#             # SỬA PROMPT: Rõ ràng hơn, cho phép "yes" nếu tài liệu có liên quan dù chỉ một phần
#             prompt = PromptTemplate(
#                 template="""Bạn là một giám khảo khách quan và cẩn thận. Nhiệm vụ của bạn là đánh giá xem TÀI LIỆU có liên quan hoặc hữu ích để trả lời CÂU HỎI hay không.
                
# TÀI LIỆU: {document}
# CÂU HỎI: {question}

# Hãy suy nghĩ kỹ. Nếu tài liệu chứa bất kỳ từ khóa, ngữ cảnh hoặc thông tin nào có thể giúp trả lời dù chỉ một phần câu hỏi, hãy chấm là 'yes'. Nếu hoàn toàn lạc đề, chấm là 'no'.
# Chỉ xuất ra MỘT TỪ DUY NHẤT (yes hoặc no), không giải thích gì thêm:""",
#                 input_variables=["document", "question"],
#             )
            
#             grader_chain = prompt | self.llm
#             filtered_docs = []

#             for doc in documents:
#                 # Ép kiểu và bắt lỗi tốt hơn cho chuỗi trả về
#                 score_response = grader_chain.invoke({"question": question, "document": doc.page_content}).content
#                 score = score_response.strip().lower()
                
#                 # Nới lỏng kiểm tra: Chỉ cần LLM có thốt ra chữ "yes" trong output
#                 if "yes" in score:
#                     print("   -> 🟢 Tài liệu hợp lệ: Giữ lại")
#                     filtered_docs.append(doc)
#                 else:
#                     print(f"   -> 🔴 Tài liệu rác (Điểm: {score}): Vứt bỏ")

#             fallback = len(filtered_docs) == 0
#             return {"documents": filtered_docs, "fallback": fallback}

#         def generate_node(state: GraphState):
#             """Trạm 3A: Tạo câu trả lời nếu tài liệu tốt"""
#             print("[CRAG] 📝 TRẠM 3A: Tổng hợp câu trả lời...")
#             question = state["question"]
#             documents = state["documents"]
            
#             context = "\n\n".join([f"[Nguồn: {d.metadata.get('dia_diem', 'Chung')}]\n{d.page_content}" for d in documents])
            
#             # SỬA PROMPT: Bê nguyên cấu trúc chặt chẽ của Offline_RAG sang
#             prompt = PromptTemplate(
#                 template="""Bạn là một trợ lý AI cung cấp thông tin. BẠN BỊ CẤM SỬ DỤNG KIẾN THỨC BÊN NGOÀI.
# Tài liệu duy nhất bạn được phép dùng:
# <ngu_canh>
# {context}
# </ngu_canh>

# Câu hỏi của khách: {question}

# QUY TẮC BẮT BUỘC:
# 1. Nếu <ngu_canh> có thông tin, hãy tóm tắt CHÍNH XÁC thông tin đó. TUYỆT ĐỐI KHÔNG tự ý suy diễn, không tự bịa thêm mốc thời gian, không thêm thắt bình luận cá nhân.
# 2. Nếu <ngu_canh> KHÔNG đủ thông tin để trả lời hoàn chỉnh, BẮT BUỘC trả lời đúng nguyên văn: 'Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác'.
# Trả lời:""",
#                 input_variables=["context", "question"],
#             )
            
#             rag_chain = prompt | self.llm
#             response = rag_chain.invoke({"context": context, "question": question}).content
#             return {"generation": response}

#         def refuse_node(state: GraphState):
#             """Trạm 3B: Từ chối trả lời nếu tài liệu sai (Luật Zero-Hallucination)"""
#             print("[CRAG] 🛑 TRẠM 3B: Kích hoạt luật Zero-Hallucination (Từ chối trả lời)")
#             tu_choi = "Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác."
#             return {"generation": tu_choi}

#         # ==========================================
#         # ĐỊNH NGHĨA RẼ NHÁNH (CONDITIONAL EDGES)
#         # ==========================================
#         def decide_to_generate(state: GraphState):
#             if state["fallback"]:
#                 return "refuse"
#             return "generate"

#         # ==========================================
#         # XÂY DỰNG VÀ ĐÓNG GÓI ĐỒ THỊ CRAG
#         # ==========================================
#         workflow = StateGraph(GraphState)
        
#         workflow.add_node("retrieve", retrieve_node)
#         workflow.add_node("grade", grade_documents_node)
#         workflow.add_node("generate", generate_node)
#         workflow.add_node("refuse", refuse_node)
        
#         workflow.set_entry_point("retrieve")
#         workflow.add_edge("retrieve", "grade")
#         workflow.add_conditional_edges("grade", decide_to_generate, {
#             "generate": "generate",
#             "refuse": "refuse"
#         })
#         workflow.add_edge("generate", END)
#         workflow.add_edge("refuse", END)
        
#         crag_app = workflow.compile()

#         # ==========================================
#         # BỌC LẠI THÀNH RUNNABLE ĐỂ KHÔNG PHÁ VỠ APP.PY
#         # ==========================================
#         def run_crag_flow(inputs: dict) -> str:
#             question = inputs.get("question", "")
#             # Chạy đồ thị
#             final_state = crag_app.invoke({"question": question})
#             # Lấy text thô từ đồ thị
#             raw_generation = final_state["generation"]
#             # Cho qua máy chém văn phong trước khi xuất ra
#             final_generation = may_chem_van_phong(raw_generation)
#             return final_generation

#         # Trả về một chuỗi tương thích hoàn toàn với LangServe / FastAPI
#         return RunnableLambda(run_crag_flow)
import re
from typing import List, TypedDict
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

# 1. MÁY CHÉM VĂN PHONG (Giữ nguyên để dọn dẹp rác từ LLM nhỏ)
def may_chem_van_phong(text: str) -> str:
    # 1. Dọn dẹp thẻ HTML và ký tự đặc biệt của mô hình
    clean_text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    
    # 2. XỬ LÝ LỖI HIỆN \n: Biến chuỗi "\\n" (2 ký tự) thành dấu xuống dòng thật (\n)
    clean_text = clean_text.replace('\\n', '\n')
    
    # 3. Xóa sạch mọi ký tự Tiếng Trung/Nhật/Hàn
    clean_text = re.sub(r'[\u4e00-\u9fff]+', '', clean_text).strip()
    
    # 4. QUÉT SẠCH LANH CHANH (Kể cả khi web bị stream)
    pattern = r"^(Đúng rồi.*?|Bạn nói đúng.*?|Đúng vậy.*?|Chính xác.*?|Câu hỏi.*?|Dạ đúng.*?)(?:,|\.|!|\n|\s)+"
    while re.match(pattern, clean_text, re.IGNORECASE):
        clean_text = re.sub(pattern, "", clean_text, count=1, flags=re.IGNORECASE).strip()
    
    # 5. Gọt nốt cái đuôi ngáo ngơ
    duoi_ngao_ngo = "không cần trả lời thêm bất kỳ thông tin nào khác"
    if duoi_ngao_ngo in clean_text.lower():
        clean_text = re.sub(r'(?i)[.,]?\s*' + duoi_ngao_ngo + r'.*$', '.', clean_text).strip()

    # 6. DỌN DẸP XUỐNG DÒNG THỪA
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()

    # 7. Viết hoa chữ cái đầu tiên
    if len(clean_text) > 0:
        clean_text = clean_text[0].upper() + clean_text[1:]
        
    return clean_text

# 2. KHAI BÁO TRẠNG THÁI (STATE) CỦA ĐỒ THỊ CRAG
class GraphState(TypedDict):
    question: str
    documents: List[any]
    generation: str
    fallback: bool


# 3. CLASS HỆ THỐNG CRAG
class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm

    def get_chain(self, retriever):
        
        # ==========================================
        # KHAI BÁO CÁC TRẠM XỬ LÝ (NODES) TRONG CRAG
        # ==========================================
        
        def retrieve_node(state: GraphState):
            """Trạm 1: Tìm kiếm tài liệu từ VectorDB"""
            question = state["question"]
            print(f"\n[CRAG] 🔎 TRẠM 1: Tìm kiếm tài liệu cho câu hỏi: '{question}'")
            docs = retriever.invoke(question)
            return {"documents": docs, "question": question}

        def grade_documents_node(state: GraphState):
            """Trạm 2: Đánh giá tài liệu (Chỉ báo động, TUYỆT ĐỐI KHÔNG vứt tài liệu)"""
            print("[CRAG] ⚖️ TRẠM 2: Đánh giá độ chính xác của tài liệu...")
            question = state["question"]
            documents = state["documents"] # Lấy danh sách tài liệu từ Trạm 1
            
            prompt = PromptTemplate(
                template="""Bạn là một giám khảo khách quan và cẩn thận. Nhiệm vụ của bạn là đánh giá xem TÀI LIỆU có liên quan hoặc hữu ích để trả lời CÂU HỎI hay không.
                
TÀI LIỆU: {document}
CÂU HỎI: {question}

Hãy suy nghĩ kỹ. Nếu tài liệu chứa bất kỳ từ khóa, ngữ cảnh hoặc thông tin nào có thể giúp trả lời dù chỉ một phần câu hỏi, hãy chấm là 'yes'. Nếu hoàn toàn lạc đề, chấm là 'no'.
Chỉ xuất ra MỘT TỪ DUY NHẤT (yes hoặc no), không giải thích gì thêm:""",
                input_variables=["document", "question"],
            )
            
            grader_chain = prompt | self.llm
            has_valid_doc = False # Cờ đánh dấu xem có ít nhất 1 tài liệu xài được không

            for doc in documents:
                score_response = grader_chain.invoke({"question": question, "document": doc.page_content}).content
                score = score_response.strip().lower()
                
                if "yes" in score:
                    print("   -> 🟢 LLM đánh giá: Có liên quan")
                    has_valid_doc = True
                else:
                    print(f"   -> 🔴 LLM đánh giá: Không liên quan (Điểm: {score})")

            # SỬA QUAN TRỌNG NHẤT Ở ĐÂY:
            # Nếu LLM bảo tất cả đều là rác -> fallback = True (Kích hoạt Trạm Cứu Hộ)
            fallback = not has_valid_doc
            
            if fallback:
                print("   => ⚠️ CẢNH BÁO: Không có tài liệu nào tốt, đi viết lại câu hỏi!")
            else:
                print("   => ✅ Gửi TOÀN BỘ tài liệu lên Trạm 3A!")

            # TRẢ VỀ NGUYÊN SI DANH SÁCH 'documents' GỐC, KHÔNG VỨT CÁI NÀO ĐI CẢ
            return {"documents": documents, "fallback": fallback}

        def rewrite_node(state: GraphState):
            """Trạm Cứu Hộ: Viết lại câu hỏi nếu tìm kiếm lần 1 thất bại"""
            print("[CRAG] 🔄 TRẠM CỨU HỘ: Viết lại câu hỏi để tìm kiếm lại...")
            question = state["question"]
            
            # Prompt ép LLM viết lại câu hỏi rõ nghĩa hơn
            prompt = PromptTemplate(
                template="""Bạn là một chuyên gia ngôn ngữ. Hãy viết lại câu hỏi sau thành một câu hỏi rõ ràng hơn, tối ưu hơn để tìm kiếm từ khóa trong cơ sở dữ liệu.
Chỉ trả về đúng MỘT câu hỏi mới, TUYỆT ĐỐI không giải thích hay nói thêm gì khác.

Câu hỏi gốc: {question}
Câu hỏi mới:""",
                input_variables=["question"],
            )
            
            better_question = (prompt | self.llm).invoke({"question": question}).content.strip()
            print(f"   -> Câu hỏi mới: {better_question}")
            
            # Dùng câu hỏi mới để tìm kiếm lại trong VectorDB
            new_docs = retriever.invoke(better_question)
            
            return {"documents": new_docs, "question": better_question}

        def generate_node(state: GraphState):
            """Trạm 3A: Tạo câu trả lời nếu tài liệu tốt"""
            print("[CRAG] 📝 TRẠM 3A: Tổng hợp câu trả lời...")
            question = state["question"]
            documents = state["documents"]
            
            context = "\n\n".join([f"[Nguồn: {d.metadata.get('dia_diem', 'Chung')}]\n{d.page_content}" for d in documents])
            
            # PROMPT ĐÃ SIẾT CHẶT: Bê nguyên cấu trúc chặt chẽ của Offline_RAG sang
            prompt = PromptTemplate(
                template="""Hãy đọc kỹ <ngu_canh> dưới đây để trả lời <câu_hỏi>. 

<ngu_canh>
{context}
</ngu_canh>

<câu_hỏi>
{question}
</câu_hỏi>

Hướng dẫn trả lời:
- Hãy tìm thông tin trong <ngu_canh> khớp với ý nghĩa của <câu_hỏi> (không cần phải giống hệt từng chữ, ví dụ "có nghề làm đồ mã" có thể dùng để trả lời cho câu hỏi "bán đồ mã").
- Viết câu trả lời ngắn gọn, chính xác dựa trên <ngu_canh>.
- Chỉ khi nào <ngu_canh> hoàn toàn lạc đề và không có bất kỳ thông tin nào liên quan, bạn mới được phép trả lời đúng câu này: 'Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác'.
Trả lời:""",
                input_variables=["context", "question"],
            )
            
            rag_chain = prompt | self.llm
            response = rag_chain.invoke({"context": context, "question": question}).content
            return {"generation": response}

        def refuse_node(state: GraphState):
            """Trạm 3B: Từ chối trả lời nếu tài liệu sai (Luật Zero-Hallucination)"""
            print("[CRAG] 🛑 TRẠM 3B: Kích hoạt luật Zero-Hallucination (Từ chối trả lời)")
            tu_choi = "Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác."
            return {"generation": tu_choi}

        # ==========================================
        # ĐỊNH NGHĨA RẼ NHÁNH (CONDITIONAL EDGES)
        # ==========================================
        def decide_to_generate(state: GraphState):
            # NẾU THẤT BẠI TRONG ĐÁNH GIÁ (FALLBACK = TRUE) -> CHẠY VÀO TRẠM CỨU HỘ
            if state["fallback"]:
                return "rewrite"
            return "generate"

        # ==========================================
        # XÂY DỰNG VÀ ĐÓNG GÓI ĐỒ THỊ CRAG
        # ==========================================
        workflow = StateGraph(GraphState)
        
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade", grade_documents_node)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("refuse", refuse_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade")
        
        workflow.add_conditional_edges("grade", decide_to_generate, {
            "generate": "generate",
            "rewrite": "rewrite"
        })
        
        workflow.add_edge("rewrite", "generate")
        workflow.add_edge("generate", END)
        workflow.add_edge("refuse", END)
        
        crag_app = workflow.compile()

        # ==========================================
        # BỌC LẠI THÀNH RUNNABLE ĐỂ KHÔNG PHÁ VỠ APP.PY
        # ==========================================
        def run_crag_flow(inputs: dict) -> str:
            question = inputs.get("question", "")
            # Chạy đồ thị
            final_state = crag_app.invoke({"question": question})
            # Lấy text thô từ đồ thị
            raw_generation = final_state["generation"]
            # Cho qua máy chém văn phong trước khi xuất ra
            final_generation = may_chem_van_phong(raw_generation)
            return final_generation

        # Trả về một chuỗi tương thích hoàn toàn với LangServe / FastAPI
        return RunnableLambda(run_crag_flow)