import re
from typing import List, TypedDict
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

# 1. MÁY CHÉM VĂN PHONG (Giữ nguyên để dọn dẹp rác từ LLM nhỏ)
def may_chem_van_phong(text: str) -> str:
    clean_text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    clean_text = clean_text.replace('\\n', '\n')
    clean_text = re.sub(r'[\u4e00-\u9fff]+', '', clean_text).strip()
    
    pattern = r"^(Đúng rồi.*?|Bạn nói đúng.*?|Đúng vậy.*?|Chính xác.*?|Câu hỏi.*?|Dạ đúng.*?)(?:,|\.|!|\n|\s)+"
    while re.match(pattern, clean_text, re.IGNORECASE):
        clean_text = re.sub(pattern, "", clean_text, count=1, flags=re.IGNORECASE).strip()
    
    duoi_ngao_ngo = "không cần trả lời thêm bất kỳ thông tin nào khác"
    if duoi_ngao_ngo in clean_text.lower():
        clean_text = re.sub(r'(?i)[.,]?\s*' + duoi_ngao_ngo + r'.*$', '.', clean_text).strip()

    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()

    if len(clean_text) > 0:
        clean_text = clean_text[0].upper() + clean_text[1:]
        
    return clean_text

# 2. KHAI BÁO TRẠNG THÁI (STATE) CỦA ĐỒ THỊ CRAG
class GraphState(TypedDict):
    question: str
    documents: List[any]
    generation: str
    fallback: bool
    retry_count: int # Thêm biến đếm số lần cứu hộ


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
            
            # Sửa lỗi NoneType
            retry_count = state.get("retry_count") or 0 
            return {"documents": docs, "question": question, "retry_count": retry_count}

        def grade_documents_node(state: GraphState):
            """Trạm 2: Đánh giá và LỌC tài liệu"""
            print("[CRAG] ⚖️ TRẠM 2: Đánh giá độ chính xác của tài liệu...")
            question = state["question"]
            documents = state["documents"]
            
            retry_count = state.get("retry_count") or 0
            
            # PROMPT ĐÁNH GIÁ: Đã thêm ví dụ số 3 để dạy model giữ lại tài liệu phản biện
            prompt = PromptTemplate(
                template="""Bạn là một trợ lý kiểm tra dữ liệu. Hãy xem tài liệu có chứa thông tin để trả lời câu hỏi hay không.
Chỉ trả lời DUY NHẤT một từ tiếng Anh là "yes" hoặc "no". TUYỆT ĐỐI KHÔNG dùng tiếng Việt, KHÔNG giải thích gì thêm.

Ví dụ 1:
TÀI LIỆU: Hồ Gươm nằm ở trung tâm thủ đô Hà Nội.
CÂU HỎI: Hồ Gươm ở đâu?
TRẢ LỜI: yes

Ví dụ 2:
TÀI LIỆU: Phở là món ăn truyền thống của Việt Nam.
CÂU HỎI: Ai là người xây dựng Văn Miếu?
TRẢ LỜI: no

Ví dụ 3:
TÀI LIỆU: 82 bia tiến sĩ được dựng từ năm 1484 đến 1780.
CÂU HỎI: 82 bia tiến sĩ được dựng cùng 1 năm đúng không?
TRẢ LỜI: yes

Bây giờ đến lượt bạn:
TÀI LIỆU: {document}
CÂU HỎI: {question}
TRẢ LỜI:""",
                input_variables=["document", "question"],
            )
            
            grader_chain = prompt | self.llm
            filtered_docs = []

            for doc in documents:
                score_response = grader_chain.invoke({"question": question, "document": doc.page_content}).content
                score = score_response.strip().lower()
                
                # Bắt nới lỏng thêm tiếng Việt đề phòng LLM cứng đầu
                if "yes" in score or "có" in score or "đạt" in score:
                    print("   -> 🟢 Đạt: Có liên quan -> Giữ lại")
                    filtered_docs.append(doc)
                else:
                    print(f"   -> 🔴 Loại bỏ rác (LLM output: {score})")

            # Nếu không còn tài liệu nào sau khi lọc -> Kích hoạt Fallback
            fallback = len(filtered_docs) == 0
            
            if fallback:
                print("   => ⚠️ KHÔNG có tài liệu tốt. Chuẩn bị cứu hộ!")
            else:
                print("   => ✅ Đã lọc xong, chuyển tài liệu sạch lên Trạm 3A!")

            return {"documents": filtered_docs, "fallback": fallback, "retry_count": retry_count}

        def rewrite_node(state: GraphState):
            """Trạm Cứu Hộ: Viết lại câu hỏi"""
            print("[CRAG] 🔄 TRẠM CỨU HỘ: Viết lại câu hỏi để tìm kiếm lại...")
            question = state["question"]
            
            # Sửa lỗi NoneType và tăng biến đếm
            retry_count = (state.get("retry_count") or 0) + 1 
            
            prompt = PromptTemplate(
                template="""Hãy trích xuất từ khóa quan trọng nhất từ câu hỏi dưới đây để tìm kiếm. Chỉ in ra từ khóa, không in gì thêm.
Câu hỏi gốc: {question}
Từ khóa:""",
                input_variables=["question"],
            )
            
            better_question = (prompt | self.llm).invoke({"question": question}).content.strip()
            print(f"   -> Câu hỏi mới (Từ khóa): {better_question}")
            
            new_docs = retriever.invoke(better_question)
            
            return {"documents": new_docs, "question": better_question, "retry_count": retry_count}

        def generate_node(state: GraphState):
            """Trạm 3A: Tổng hợp câu trả lời tiêu chuẩn (Cho phép dùng logic)"""
            print("[CRAG] 📝 TRẠM 3A: Tổng hợp câu trả lời...")
            question = state["question"]
            documents = state["documents"]
            
            context = "\n\n".join([f"[Nguồn: {d.metadata.get('dia_diem', 'Chung')}]\n{d.page_content}" for d in documents])
            
            # PROMPT SINH CÂU TRẢ LỜI: Đã cởi trói để model được dùng logic cơ bản
            prompt = PromptTemplate(
                template="""Sử dụng <ngu_canh> dưới đây để trả lời câu hỏi. 
- Chỉ trả lời trực tiếp vào trọng tâm, ngắn gọn, súc tích.
- Bạn được phép dùng logic cơ bản để đối chiếu thông tin trong <ngu_canh> nhằm xác nhận Đúng/Sai cho các câu hỏi nghi vấn.
- TUYỆT ĐỐI KHÔNG thêm các câu chào hỏi, xưng hô.
- Nếu <ngu_canh> hoàn toàn không chứa dữ kiện nào để suy luận, BẮT BUỘC trả lời chính xác nguyên văn: 'Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác'.

<ngu_canh>
{context}
</ngu_canh>

Câu hỏi: {question}
Trả lời:""",
                input_variables=["context", "question"],
            )
            
            rag_chain = prompt | self.llm
            response = rag_chain.invoke({"context": context, "question": question}).content
            return {"generation": response}

        def refuse_node(state: GraphState):
            """Trạm 3B: Từ chối trả lời (Luật Zero-Hallucination)"""
            print("[CRAG] 🛑 TRẠM 3B: Kích hoạt luật Zero-Hallucination (Từ chối trả lời)")
            tu_choi = "Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác."
            return {"generation": tu_choi}

        # ==========================================
        # ĐỊNH NGHĨA RẼ NHÁNH (CONDITIONAL EDGES)
        # ==========================================
        def decide_to_generate(state: GraphState):
            if state["fallback"]:
                retry_count = state.get("retry_count") or 0
                
                # Nếu đã cứu hộ 1 lần mà VẪN fallback -> ĐẦU HÀNG, KHÔNG LẶP VÔ HẠN
                if retry_count >= 1:
                    print("   => ❌ Cứu hộ thất bại, từ chối trả lời!")
                    return "refuse"
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
            "rewrite": "rewrite",
            "refuse": "refuse"
        })
        
        # SỬA QUAN TRỌNG: Viết lại câu hỏi xong phải quay lại đánh giá tài liệu mới
        workflow.add_edge("rewrite", "grade") 
        workflow.add_edge("generate", END)
        workflow.add_edge("refuse", END)
        
        crag_app = workflow.compile()

        # ==========================================
        # BỌC LẠI THÀNH RUNNABLE
        # ==========================================
        def run_crag_flow(inputs: dict) -> str:
            question = inputs.get("question", "")
            final_state = crag_app.invoke({"question": question})
            raw_generation = final_state["generation"]
            final_generation = may_chem_van_phong(raw_generation)
            return final_generation

        return RunnableLambda(run_crag_flow)