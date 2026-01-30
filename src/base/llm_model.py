# [File: src/base/llm_model.py]
from langchain_community.llms import Ollama

def get_ollama_llm(model_name: str = "llama3.2", base_url: str = "http://localhost:11434", **kwargs):
    """
    Kết nối tới Ollama đang chạy Local (trên máy tính của bạn).
    Mặc định dùng model 'llama3.2' (bản 3B) và cổng 11434.
    """
    print(f"🔌 Đang kết nối tới Ollama tại {base_url} với model: {model_name}...")
    
    try:
        llm = Ollama(
            base_url=base_url,
            model=model_name,
            # Các tham số cấu hình thêm (nếu có)
            temperature=kwargs.get("temperature", 0.7),
            **kwargs
        )
        return llm
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo Ollama: {e}")
        return None

# --- Phần này để test nhanh khi chạy trực tiếp file này ---
if __name__ == "__main__":
    # Test thử kết nối
    print("--- ĐANG TEST KẾT NỐI OLLAMA ---")
    try:
        # Gọi hàm khởi tạo
        my_llm = get_ollama_llm() # Mặc định là llama3.2
        
        # Thử hỏi một câu
        question = "Giải thích RAG là gì cho sinh viên CNTT một cách ngắn gọn."
        print(f"❓ Câu hỏi: {question}")
        
        response = my_llm.invoke(question)
        print("\n🤖 Trả lời:")
        print(response)
        print("\n✅ THÀNH CÔNG! Ollama đang hoạt động tốt.")
    except Exception as e:
        print("\n❌ THẤT BẠI! Hãy kiểm tra xem bạn đã bật App Ollama chưa?")
        print(f"Lỗi chi tiết: {e}")