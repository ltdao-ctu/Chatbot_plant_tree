# server.py (rút gọn)
from fastapi import FastAPI, File, UploadFile
import uvicorn
from src.ingest import ingest_file
from src.qa import answer

app = FastAPI()

# @app.post("/upload")
# async def upload(file: UploadFile = File(...)):
#     contents = await file.read()
#     # lưu tạm và ingest
#     path = f"data/{file.filename}"
#     with open(path, "wb") as f:
#         f.write(contents)
#     ingest_file(path)
#     return {"status":"ok", "filename": file.filename}

@app.post("/ask")
async def ask(q: dict):
    query = q.get("query")
    return {"answer": answer(query)}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)