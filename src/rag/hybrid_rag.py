import re
from typing import Dict, TypedDict, List
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

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

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    loop_step: int

class Hybrid_RAG:
    def __init__(self, llm):
        self.llm = llm
        self.max_retries = 2
        
        # 1. Prompt Trả lời 
        self.qa_prompt = PromptTemplate(
            template="""Bạn là NPC lịch sử Hà Nội. Dựa vào NGỮ CẢNH, hãy trả lời CÂU HỎI. 
Nếu không có thông tin, trả lời: "Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác."
NGỮ CẢNH: {context}
CÂU HỎI: {question}
TRẢ LỜI:""",
            input_variables=["context", "question"]
        )
        
        # 2. Prompt Trạm 1: CRAG (Chấm tài liệu)
        self.retrieval_grader_prompt = PromptTemplate(
            template="""Đánh giá xem NGỮ CẢNH có chứa thông tin để trả lời CÂU HỎI không.
NGỮ CẢNH: {context}
CÂU HỎI: {question}
Chỉ trả lời [HOP_LE] nếu có liên quan, hoặc [KHONG_LIEN_QUAN] nếu hoàn toàn lạc đề.
ĐÁNH GIÁ:""",
            input_variables=["context", "question"]
        )

        # 3. Prompt Trạm 2: Self-RAG (Chấm ảo giác)
        self.hallucination_prompt = PromptTemplate(
            template="""Đánh giá xem CÂU TRẢ LỜI có bịa đặt thông tin ngoài NGỮ CẢNH không.
NGỮ CẢNH: {context}
CÂU TRẢ LỜI: {generation}
Chỉ trả lời [HOP_LE] nếu hoàn toàn dựa vào ngữ cảnh, hoặc [BIA_DAT] nếu có chi tiết chế thêm.
ĐÁNH GIÁ:""",
            input_variables=["context", "generation"]
        )

    def get_chain(self, retriever):
        self.retriever = retriever

        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("grade_documents", self.grade_documents_node) # Node mới của CRAG
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("fallback", self.fallback_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Trạm 1: Có tài liệu mới cho generate, không thì Fallback
        workflow.add_conditional_edges(
            "grade_documents",
            self.check_document_relevance,
            {
                "relevant": "generate",
                "irrelevant": "fallback"
            }
        )
        
        # Trạm 2: Generate xong thì chấm xem có bịa đặt không
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "useful": END,
                "hallucination": "generate",
                "fallback": "fallback"
            }
        )
        workflow.add_edge("fallback", END)

        app = workflow.compile()
        
        def run_chain(inputs: dict):
            question = inputs.get("question", "")
            result = app.invoke({"question": question, "loop_step": 0})
            return may_chem_van_phong(result["generation"])

        return RunnableLambda(run_chain)

    def retrieve_node(self, state: GraphState):
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question, "loop_step": state.get("loop_step", 0)}

    # TRẠM 1: ĐÁNH GIÁ TÀI LIỆU (CRAG)
    def grade_documents_node(self, state: GraphState):
        question = state["question"]
        documents = state["documents"]
        
        if not documents:
            return {"documents": []}
            
        context = "\n\n".join([doc.page_content for doc in documents])
        chain = self.retrieval_grader_prompt | self.llm
        score = chain.invoke({"context": context, "question": question})
        score_text = score.content.lower() if hasattr(score, 'content') else str(score).lower()
        
        if "hop_le" in score_text or "hợp lệ" in score_text or "có" in score_text:
            return {"documents": documents} # Giữ tài liệu
        else:
            print(f"\n[CRAG] 🛑 Tài liệu tìm được không liên quan. Bỏ qua bước sinh văn bản.")
            return {"documents": []} # Vứt tài liệu đi

    def check_document_relevance(self, state: GraphState):
        if len(state["documents"]) == 0:
            return "irrelevant"
        return "relevant"

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

    # TRẠM 2: ĐÁNH GIÁ CÂU TRẢ LỜI (Self-RAG)
    def grade_generation(self, state: GraphState):
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        loop_step = state["loop_step"]
        
        if "tôi không biết" in generation.lower() or "tôi không có thông tin" in generation.lower():
            return "useful"

        context = "\n\n".join([doc.page_content for doc in documents])
        chain = self.hallucination_prompt | self.llm
        score = chain.invoke({"context": context, "generation": generation})
        score_text = score.content.lower() if hasattr(score, 'content') else str(score).lower()
        
        if "hop_le" in score_text or "hợp lệ" in score_text or "không bịa" in score_text or "đúng" in score_text:
            return "useful"
        else:
            if loop_step <= self.max_retries:
                print(f"\n[Self-RAG] ⚠️ Trạm 2 báo cáo: Phát hiện ảo giác. Đang viết lại lần {loop_step}...")
                return "hallucination"
            else:
                print("\n[Self-RAG] ❌ Ép sửa nhiều lần không thành công. Kích hoạt Fallback.")
                return "fallback"