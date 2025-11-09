# 🚀 Hướng dẫn khởi chạy hệ thống RAG

## 1️⃣ Chuẩn bị dữ liệu
- Đặt tất cả các tệp **`.docx`** đầu vào vào thư mục:  
    ./data/


## 2️⃣ Cài đặt thư viện cần thiết
Chạy lệnh sau trong thư mục dự án:
    pip install -r requirements.txt

3️⃣ Tiền xử lý dữ liệu (tách nhỏ văn bản)
Chạy lệnh: 
    python ./src/preprocessing.py

Kết quả sẽ được lưu tại thư mục: 
    ./data_output/

4️⃣ Tạo vector embedding và cơ sở dữ liệu FAISS
Chạy: 
    python ./src/ingest.py
Kết quả:
    File FAISS: ./data_output/faiss.index
    Metadata: ./data_output/docs.json

5️⃣ Khởi chạy server API
Chạy:
    python ./src/server.py
Mặc định server chạy tại:
👉 http://127.0.0.1:8000

6️⃣ Khởi chạy giao diện Streamlit
Chạy:
    streamlit run ./src/streamlit_app.py
Truy cập giao diện tại:
👉 http://localhost:8501