from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
import requests

import os

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET = os.environ.get("BUCKET")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.post("/upload")
async def upload(file: UploadFile = File(...), theme: str = Form(...)):
    contents = await file.read()
    filename = file.filename
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{filename}"
    res = requests.put(upload_url, headers={
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/octet-stream"
    }, data=contents)
    if res.status_code in [200, 201]:
        return {"success": True,
                "url": f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{filename}",
                "theme": theme}
    return {"success": False, "error": res.text}