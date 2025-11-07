1. Dữ liệu đầu vào đặt ở folder data dưới dạng *.docx.
2. Chạy pip install -r requirements.txt
3. Chạy file preprocessing.py (python ./src/preprossing.py) để cắt nhỏ dữ liệu và lưu vào trong data_output.
4. Chạy file ingest.py (python ./src/ingest.py) để embedding dữ liệu và lưu vào docs, faiss
5. Chạy server.py (python ./server.py) để khởi chạy hệ thống.
6. Chạy streamlit_app.py (Streamlit run streamlit_app.py) để khởi chạy giao diện tương tác.