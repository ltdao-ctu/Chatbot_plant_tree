# # # ingest.py (phiên bản multi-representation RAG - không chunk tự động)
# # # ---------------------------------------------------------------
# # # Mục tiêu:
# # #   - Duyệt qua tất cả file .docx (hoặc văn bản) trong thư mục chỉ định
# # #   - Mỗi file được embedding nhiều tầng biểu diễn (multi-representation)
# # #       + raw: toàn bộ nội dung
# # #       + summary: tóm tắt nội dung (ngắn hơn)
# # #       + keywords: trích xuất các chủ từ chính
# # #   - Lưu vào FAISS + metadata JSON để phục vụ RAG đa tầng
# # import sys, io
# # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# # import os, json, uuid
# # import numpy as np
# # import faiss
# # from sentence_transformers import SentenceTransformer
# # from extractors import auto_extract  # tự động trích xuất nội dung file docx, pdf, txt, v.v.
# # from utils import extract_summary, extract_keywords  # giả định bạn có 2 hàm tóm tắt & trích chủ từ

# # # ---------------------------------------------------------------
# # # 🔧 Cấu hình
# # DATA_DIR = "data_output"             # thư mục chứa file cần ingest
# # INDEX_FILE = "faiss.index"
# # META_FILE = "docs.json"

# # # model embedding nhẹ, hỗ trợ đa ngôn ngữ (tốt cho tiếng Việt)
# # MODEL_NAME = "intfloat/multilingual-e5-small"
# # embedder = SentenceTransformer(MODEL_NAME)
# # dimension = embedder.get_sentence_embedding_dimension()

# # # ---------------------------------------------------------------
# # # 🧠 Tạo hoặc load FAISS index
# # if os.path.exists(INDEX_FILE):
# #     index = faiss.read_index(INDEX_FILE)
# #     with open(META_FILE, "r", encoding="utf-8") as f:
# #         docs = json.load(f)
# # else:
# #     # Dùng cosine similarity (inner product + normalize vector)
# #     index = faiss.IndexFlatIP(dimension)
# #     docs = []

# # # ---------------------------------------------------------------
# # # 📄 Hàm xử lý 1 file đơn lẻ
# # def ingest_file(path):
# #     text = auto_extract(path)
# #     if not text.strip():
# #         print(f"[!] Bỏ qua file rỗng: {path}")
# #         return

# #     # Tạo các biểu diễn khác nhau cho cùng 1 tài liệu
# #     representations = {
# #         "raw": text,
# #         "summary": extract_summary(text),
# #         "keywords": extract_keywords(text)
# #     }

# #     metas = []
# #     vecs = []

# #     # embedding từng tầng
# #     for rep_type, rep_text in representations.items():
# #         doc_id = str(uuid.uuid4())
# #         vec = embedder.encode(rep_text, normalize_embeddings=True)

# #         meta = {
# #             "id": doc_id,
# #             "source": os.path.basename(path),
# #             "rep_type": rep_type,   # loại biểu diễn (raw / summary / keywords)
# #             "text": rep_text
# #         }

# #         metas.append(meta)
# #         vecs.append(vec)

# #     # Thêm vào FAISS và lưu metadata
# #     vecs_np = np.vstack(vecs).astype("float32")
# #     index.add(vecs_np)
# #     docs.extend(metas)

# #     faiss.write_index(index, INDEX_FILE)
# #     with open(META_FILE, "w", encoding="utf-8") as f:
# #         json.dump(docs, f, ensure_ascii=False, indent=2)

# #     print(f"[*] Ingested {path}: {len(metas)} representations")

# # # ---------------------------------------------------------------
# # # 🚀 Ingest toàn bộ file trong thư mục chỉ định
# # def ingest_folder(folder=DATA_DIR):
# #     for fname in os.listdir(folder):
# #         p = os.path.join(folder, fname)
# #         if os.path.isfile(p):
# #             ingest_file(p)

# # # ---------------------------------------------------------------
# # if __name__ == "__main__":
# #     print("🚀 Multi-representation RAG Embedding Started...\n")
# #     ingest_folder(DATA_DIR)
# #     print("\n✅ Hoàn tất embedding tất cả file.")

# # ingest.py (phiên bản multi-representation RAG - có thanh tiến trình)
# # ---------------------------------------------------------------
# # Mục tiêu:
# #   - Duyệt qua tất cả file .docx (hoặc văn bản) trong thư mục chỉ định
# #   - Mỗi file được embedding nhiều tầng biểu diễn (multi-representation)
# #   - Hiển thị tiến độ ingest bằng thanh tiến trình tqdm
# import sys, io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# import os, json, uuid
# import numpy as np
# import faiss
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# from extractors import auto_extract
# from utils import extract_summary, extract_keywords

# # ---------------------------------------------------------------
# # 🔧 Cấu hình
# DATA_DIR = "data_output"
# INDEX_FILE = "faiss.index"
# META_FILE = "docs.json"

# MODEL_NAME = "intfloat/multilingual-e5-small"
# embedder = SentenceTransformer(MODEL_NAME)
# dimension = embedder.get_sentence_embedding_dimension()

# # ---------------------------------------------------------------
# # 🧠 Load hoặc tạo FAISS index
# if os.path.exists(INDEX_FILE):
#     index = faiss.read_index(INDEX_FILE)
#     with open(META_FILE, "r", encoding="utf-8") as f:
#         docs = json.load(f)
# else:
#     index = faiss.IndexFlatIP(dimension)
#     docs = []

# # ---------------------------------------------------------------
# def ingest_file(path):
#     text = auto_extract(path)
#     if not text.strip():
#         print(f"[!] Bỏ qua file rỗng: {path}")
#         return

#     representations = {
#         "raw": text,
#         "summary": extract_summary(text),
#         "keywords": extract_keywords(text)
#     }

#     metas = []
#     vecs = []

#     for rep_type, rep_text in representations.items():
#         doc_id = str(uuid.uuid4())
#         vec = embedder.encode(rep_text, normalize_embeddings=True)

#         meta = {
#             "id": doc_id,
#             "source": os.path.basename(path),
#             "rep_type": rep_type,
#             "text": rep_text
#         }
#         metas.append(meta)
#         vecs.append(vec)

#     vecs_np = np.vstack(vecs).astype("float32")
#     index.add(vecs_np)
#     docs.extend(metas)

#     faiss.write_index(index, INDEX_FILE)
#     with open(META_FILE, "w", encoding="utf-8") as f:
#         json.dump(docs, f, ensure_ascii=False, indent=2)

# # ---------------------------------------------------------------
# def ingest_folder(folder=DATA_DIR):
#     files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
#     if not files:
#         print("⚠️ Không có file nào trong thư mục cần ingest.")
#         return

#     print(f"📁 Đang ingest {len(files)} file trong thư mục: {folder}\n")

#     for path in tqdm(files, desc="🔄 Đang xử lý", unit="file", ncols=90):
#         ingest_file(path)

# # ---------------------------------------------------------------
# if __name__ == "__main__":
#     print("🚀 Multi-representation RAG Embedding Started...\n")
#     ingest_folder(DATA_DIR)
#     print("\n✅ Hoàn tất embedding tất cả file.")





# ingest.py (phiên bản 2-vector: summary + keywords, chung 1 FAISS, lưu raw text)
# ---------------------------------------------------------------
# Mục tiêu:
#   - Duyệt qua tất cả file .docx / .txt / .pdf trong thư mục DATA_DIR
#   - Mỗi file sinh 2 vector embedding:
#       + summary: tóm tắt nội dung
#       + keywords: trích xuất các chủ từ chính
#   - Cả hai vector cùng nằm trong 1 FAISS index
#   - Metadata (docs.json) lưu raw text đầy đủ
#   - Hiển thị tiến trình ingest bằng tqdm
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os, json, uuid
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.extractors import auto_extract
from src.utils import extract_summary, extract_keywords

# ---------------------------------------------------------------
# 🔧 Cấu hình
DATA_DIR = "data_output"
INDEX_FILE = "faiss.index"
META_FILE = "docs.json"

MODEL_NAME = "intfloat/multilingual-e5-small"
embedder = SentenceTransformer(MODEL_NAME)
dimension = embedder.get_sentence_embedding_dimension()

# ---------------------------------------------------------------
# 🧠 Tạo hoặc load FAISS index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)
else:
    index = faiss.IndexFlatIP(dimension)  # cosine similarity
    docs = []

# ---------------------------------------------------------------
# 📄 Hàm xử lý 1 file đơn
def ingest_file(path):
    raw_text = auto_extract(path)
    if not raw_text.strip():
        print(f"[!] Bỏ qua file rỗng: {path}")
        return

    # Tạo 2 biểu diễn: summary + keywords
    summary_text = extract_summary(raw_text)
    keyword_text = extract_keywords(raw_text)

    representations = {
        "summary": summary_text,
        "keywords": keyword_text
    }

    metas, vecs = [], []

    # Duyệt qua 2 biểu diễn
    for rep_type, rep_text in representations.items():
        doc_id = str(uuid.uuid4())
        vec = embedder.encode(rep_text, normalize_embeddings=True)

        meta = {
            "id": doc_id,
            "source": os.path.basename(path),
            "rep_type": rep_type,   # summary / keywords
            "text": raw_text        # luôn lưu raw text
        }

        metas.append(meta)
        vecs.append(vec)

    # Thêm 2 vector vào FAISS
    vecs_np = np.vstack(vecs).astype("float32")
    index.add(vecs_np)
    docs.extend(metas)

    # Lưu lại
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------
# 🚀 Ingest toàn bộ thư mục
def ingest_folder(folder=DATA_DIR):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not files:
        print("⚠️ Không có file nào trong thư mục cần ingest.")
        return

    print(f"📁 Đang ingest {len(files)} file trong thư mục: {folder}\n")

    for path in tqdm(files, desc="🔄 Đang xử lý", unit="file", ncols=90):
        ingest_file(path)

# ---------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Multi-representation (summary + keywords) Embedding Started...\n")
    ingest_folder(DATA_DIR)
    print("\n✅ Hoàn tất embedding tất cả file.")
