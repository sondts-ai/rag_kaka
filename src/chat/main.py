from pydantic import BaseModel, Field
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.chat.history import create_session_factory
from src.chat.output_parser import Str_OutputParser

# 1. THÊM LUẬT CHỐNG BỊA VÀ NGỮ CẢNH VÀO PROMPT
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là một hướng dẫn viên du lịch ảo tại Hà Nội. BẠN BỊ CẤM SỬ DỤNG KIẾN THỨC BÊN NGOÀI.\n"
                   "Chỉ được phép trả lời dựa trên nội dung trong phần <ngu_canh> dưới đây.\n\n"
                   "<ngu_canh>\n{context}\n</ngu_canh>\n\n"
                   "QUY TẮC BẮT BUỘC: Nếu thông tin không có trong <ngu_canh>, TUYỆT ĐỐI KHÔNG TỰ BỊA ĐẶT mà hãy trả lời nguyên văn: "
                   "'Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác'."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{human_input}"),
    ]
)

class InputChat(BaseModel):
    human_input: str = Field(
        ...,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "human_input"}},
    )

def format_docs_for_chat(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 2. THÊM BIẾN `retriever` VÀO HÀM ĐỂ TÌM TÀI LIỆU
def build_chat_chain(llm, retriever, history_folder, max_history_length):
    # Setup luồng kết hợp: Tìm tài liệu -> Bỏ vào Prompt -> Đưa cho LLM
    chain = (
        {
            "context": itemgetter("human_input") | retriever | format_docs_for_chat,
            "human_input": itemgetter("human_input"),
            "chat_history": itemgetter("chat_history")
        }
        | chat_prompt 
        | llm 
        | Str_OutputParser()
    )
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        create_session_factory(base_dir=history_folder, 
                               max_history_length=max_history_length),
        input_messages_key="human_input",
        history_messages_key="chat_history",
    )
    return chain_with_history