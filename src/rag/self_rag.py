import re
from typing import Dict, TypedDict, List
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

# Máy chém văn phong chuẩn của bạn
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
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    if len(clean_text) > 0:
        clean_text = clean_text[0].upper() + clean_text[1:]
    return clean_text

# Cấu trúc Trạng thái của LangGraph
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    loop_step: int

class Self_RAG:
    def __init__(self, llm):
        self.llm = llm
        self.max_retries = 2 # Cho phép AI tự sửa lỗi tối đa 2 lần
        
        # 1. Prompt Trả lời câu hỏi
        self.qa_prompt = PromptTemplate(
            template="""Bạn là NPC hướng dẫn viên lịch sử Hà Nội. Dựa vào NGỮ CẢNH sau, hãy trả lời CÂU HỎI. 
Tuyệt đối KHÔNG sử dụng kiến thức bên ngoài. Nếu NGỮ CẢNH không có thông tin, bắt buộc trả lời: "Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác."

NGỮ CẢNH: 
{context}

CÂU HỎI: {question}
TRẢ LỜI:""",
            input_variables=["context", "question"]
        )
        
        # 2. Prompt Giám khảo (Kiểm tra Ảo giác - Hallucination)
        self.hallucination_prompt = PromptTemplate(
            template="""Nhiệm vụ: Đánh giá xem CÂU TRẢ LỜI có bịa đặt thông tin không có trong NGỮ CẢNH hay không.
Chỉ xuất ra 'yes' (nếu CÂU TRẢ LỜI hoàn toàn dựa vào NGỮ CẢNH) hoặc 'no' (nếu CÂU TRẢ LỜI có chi tiết bịa đặt).

NGỮ CẢNH: {context}
CÂU TRẢ LỜI: {generation}
ĐÁNH GIÁ (chỉ ghi yes hoặc no):""",
            input_variables=["context", "generation"]
        )

    def get_chain(self, retriever):
        self.retriever = retriever

        # Xây dựng luồng đồ thị Self-RAG
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("fallback", self.fallback_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        
        # Sau khi generate, rẽ nhánh đi chấm điểm xem có bịa đặt không
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "useful": END,               # Không bịa đặt -> Kết thúc và trả về
                "hallucination": "generate", # Phát hiện bịa đặt -> Bắt viết lại
                "fallback": "fallback"       # Hết số lần cho phép -> Từ chối trả lời an toàn
            }
        )
        workflow.add_edge("fallback", END)

        app = workflow.compile()
        
        # Đóng gói thành RunnableLambda để dùng chung luồng code cũ
        def run_chain(inputs: dict):
            question = inputs.get("question", "")
            result = app.invoke({"question": question, "loop_step": 0})
            return may_chem_van_phong(result["generation"])

        return RunnableLambda(run_chain)

    def retrieve_node(self, state: GraphState):
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question, "loop_step": state.get("loop_step", 0)}

    def generate_node(self, state: GraphState):
        question = state["question"]
        documents = state["documents"]
        loop_step = state.get("loop_step", 0)
        
        context = "\n\n".join([doc.page_content for doc in documents])
        chain = self.qa_prompt | self.llm
        generation = chain.invoke({"context": context, "question": question})
        
        if hasattr(generation, 'content'):
            generation = generation.content
            
        return {"generation": generation, "loop_step": loop_step + 1}
        
    def fallback_node(self, state: GraphState):
        return {"generation": "Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác."}

    def grade_generation(self, state: GraphState):
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        loop_step = state["loop_step"]
        
        # Nếu bot tự nhận không biết, coi như hợp lệ (không bịa đặt)
        if "tôi không biết" in generation.lower() or "tôi không có thông tin" in generation.lower():
            return "useful"

        # Kiểm tra Ảo giác (Hallucination)
        context = "\n\n".join([doc.page_content for doc in documents])
        chain = self.hallucination_prompt | self.llm
        score = chain.invoke({"context": context, "generation": generation})
        
        score_text = score.content.lower() if hasattr(score, 'content') else str(score).lower()

        if "yes" in score_text:
            return "useful"
        else:
            if loop_step <= self.max_retries:
                print(f"\n[Self-RAG] ⚠️ CẢNH BÁO: Phát hiện ảo giác. Đang bắt AI viết lại (Lần {loop_step})...")
                return "hallucination"
            else:
                print("\n[Self-RAG] ❌ AI cố tình bịa đặt quá số lần. Kích hoạt Fallback.")
                return "fallback"