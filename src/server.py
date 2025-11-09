# server.py (rút gọn)
from fastapi import FastAPI, File, UploadFile
import uvicorn
from ingest import ingest_file
from qa import answer

app = FastAPI()



@app.post("/ask")
async def ask(q: dict):
    query = q.get("query")
    return {"answer": answer(query)}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)