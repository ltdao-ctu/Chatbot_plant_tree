import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import faiss
import json
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

# Giả sử bạn đã có:
# - embedder: SentenceTransformer
# - index: FAISS index
# - docs: danh sách metadata (docs.json đã load sẵn)


MODEL_NAME = "intfloat/multilingual-e5-small"
INDEX_FILE = "faiss.index"
META_FILE = "docs.json"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --------------------------------------------
# 🧠 Load model + dữ liệu
embedder = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_FILE)
reranker = CrossEncoder(RERANK_MODEL)
with open(META_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)




# === Function: retrieve context ===
def retrieve(query, top_k=30, rerank_top_n=9):
    # 1. Tính embedding cho query
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")

    # 2. Lấy top_k từ FAISS
    D, I = index.search(qv, top_k)
    candidates = [docs[idx] for idx in I[0] if idx != -1]

    # 3. Rerank bằng CrossEncoder
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    # 4. Ghép điểm và sắp xếp
    reranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 5. Trả về top_n kết quả cuối cùng
    results = [
        {
            "rank": i+1,
            "source": r[0]["source"],
            "rep_type": r[0]["rep_type"],
            "score": float(r[1]),
            "text": r[0]["text"][:500] + "..."  # cắt ngắn khi in
        }
        for i, r in enumerate(reranked[:rerank_top_n])
    ]

    return results



# === Function: build prompt ===
# def make_prompt(query, retrieved):
#     parts = []
#     for i, r in enumerate(retrieved, 1):
#         parts.append(f"[{i}] (source: {r['source']}) {r['text'][:800]}")
#     context = "\n\n".join(parts)
#     prompt = f"Bạn là trợ lý trồng cây. Dựa trên thông tin sau:\n{context}\n\nCâu hỏi: {query}\nTrả lời ngắn gọn và nêu nguồn."
#     return prompt

def make_prompt(query: str, retrieved: list, role: str = "trợ lý trồng cây") -> str:
    """
    Tạo prompt hoàn chỉnh cho LLM dựa trên các đoạn văn được truy xuất.
    
    Args:
        query (str): Câu hỏi người dùng.
        retrieved (list): Danh sách tài liệu (đã qua retrieve + rerank).
        role (str): Vai trò của hệ thống trợ lý.
    
    Returns:
        str: Prompt hoàn chỉnh sẵn sàng gửi vào LLM.
    """

    if not retrieved:
        return f"Không tìm thấy thông tin liên quan cho câu hỏi: {query}"

    # 🧩 Ghép ngữ cảnh từ các đoạn truy xuất
    parts = []
    for i, r in enumerate(retrieved, 1):
        # Giới hạn text để tránh prompt quá dài
        snippet = (r.get("text") or "").strip().replace("\n", " ")
        snippet = snippet[:800] + ("..." if len(snippet) > 800 else "")

        parts.append(f"[{i}] (Nguồn: {r.get('source', 'không rõ')})\n{snippet}")

    context = "\n\n".join(parts)

    # 🧠 Cấu trúc prompt chuẩn RAG
    prompt = (
        f"Bạn là {role}, có nhiệm vụ trả lời câu hỏi dựa trên thông tin đã cho.\n\n"
        f"=== NGỮ CẢNH ===\n{context}\n\n"
        f"=== CÂU HỎI ===\n{query}\n\n"
        f"=== YÊU CẦU ===\n"
        f"- Trả lời ngắn gọn, chính xác.\n"
        f"- Nếu có thể, hãy nêu rõ nguồn (số thứ tự trong ngoặc vuông).\n"
    )

    return prompt

# === Function: call Ollama API ===
# def call_ollama(prompt, model="gemma:2b"):
#     url = "http://localhost:11434/api/generate"
#     payload = {
#         "model": model,   # bạn có thể đổi sang "mistral", "gemma:2b", v.v.
#         "prompt": prompt,
#         "stream": False
#     }
#     resp = requests.post(url, json=payload)
#     if resp.status_code == 200:
#         return resp.json()["response"]
#     else:
#         return f"Lỗi Ollama API: {resp.text}"

def call_ollama(prompt: str, model: str = "gemma:2b", temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """
    Gọi API của Ollama để sinh phản hồi từ mô hình ngôn ngữ cục bộ.

    Args:
        prompt (str): Chuỗi prompt đầu vào (đã bao gồm context và câu hỏi).
        model (str): Tên mô hình Ollama (vd: "gemma:2b", "mistral", "llama3", ...).
        temperature (float): Mức độ sáng tạo của mô hình (0.0 - 1.0).
        max_tokens (int): Giới hạn số token sinh ra.

    Returns:
        str: Phản hồi văn bản từ mô hình hoặc thông báo lỗi.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()  # ném lỗi nếu status != 200
        data = resp.json()
        return data.get("response", "(Không có phản hồi từ mô hình)")
    except requests.exceptions.ConnectionError:
        return "❌ Không thể kết nối đến Ollama. Hãy đảm bảo dịch vụ đang chạy (ollama serve)."
    except requests.exceptions.Timeout:
        return "⚠️ Yêu cầu tới Ollama bị quá thời gian chờ."
    except requests.exceptions.JSONDecodeError:
        return f"⚠️ Phản hồi không hợp lệ: {resp.text[:200]}"
    except Exception as e:
        return f"⚠️ Lỗi không xác định: {e}"
    

# === Main answer function ===
# def answer(query, top_k=3, model="gemma:2b"):
#     retrieved = retrieve(query, top_k=top_k)
#     if not retrieved:
#         return "Xin lỗi, tôi không tìm thấy thông tin trong cơ sở dữ liệu."

#     # In ra top-k để theo dõi
#     print("\n=== Retrieved context ===")
#     for i, r in enumerate(retrieved, 1):
#         print(f"[{i}] (source: {r['source']}) {r['text'][:200]}...")
#     print("=========================\n")

#     prompt = make_prompt(query, retrieved)
#     print(f"[DEBUG] Prompt length: {len(prompt)} chars\n")

#     return call_ollama(prompt, model=model)



def answer(query: str, top_k: int = 5, model: str = "gemma:2b", debug: bool = True) -> str:
    """
    Truy vấn hệ thống RAG: retrieve → tạo prompt → gọi Ollama → trả lời.

    Args:
        query (str): Câu hỏi người dùng.
        top_k (int): Số đoạn văn lấy từ FAISS (trước khi rerank).
        model (str): Mô hình Ollama cần gọi (vd: "gemma:2b", "mistral").
        debug (bool): Nếu True, in log truy xuất và prompt.

    Returns:
        str: Câu trả lời được sinh ra bởi mô hình.
    """
    try:
        # 1️⃣ Retrieve (lấy dữ liệu liên quan)
        retrieved = retrieve(query, top_k=top_k)
        if not retrieved:
            return "❌ Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

        if debug:
            print("\n=== 📚 Retrieved context ===")
            for i, r in enumerate(retrieved, 1):
                snippet = r["text"].replace("\n", " ")[:200]
                print(f"[{i}] ({r.get('rep_type', '-')}) {r['source']}: {snippet}...")
            print("============================\n")

        # 2️⃣ Tạo prompt cho LLM
        prompt = make_prompt(query, retrieved)
        if debug:
            print(f"[DEBUG] Prompt length: {len(prompt)} chars\n")

        # 3️⃣ Gọi Ollama LLM để sinh câu trả lời
        answer = call_ollama(prompt, model=model)
        return answer

    except Exception as e:
        return f"⚠️ Lỗi khi xử lý câu hỏi: {e}"
