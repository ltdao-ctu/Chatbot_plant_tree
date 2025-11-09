
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os, json, uuid
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from extractors import auto_extract
from utils import extract_summary, extract_keywords
from config_loader import load_config

config = load_config()


# ---------------------------------------------------------------
# 🔧 Cấu hình

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # thư mục chứa file hiện tại
INDEX_FILE = os.path.join(BASE_DIR, "..", "faiss.index")
META_FILE = os.path.join(BASE_DIR, "..", "docs.json")
DATA_DIR = os.path.join(BASE_DIR, "..", "data_output")

MODEL_NAME = config["model"]["embedding_model"]
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
