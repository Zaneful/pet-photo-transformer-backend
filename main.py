from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json, os, aiohttp
from supabase import create_client, Client

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("prompts.json", "r") as f:
    PROMPTS = json.load(f)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("BUCKET")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"

@app.get("/prompts")
def get_prompts():
    return [{"id": p["id"], "title": p["title"]} for p in PROMPTS]

@app.post("/generate")
async def generate(prompt_id: int, file: UploadFile = File(...)):
    prompt = next((p["promptText"] for p in PROMPTS if p["id"] == prompt_id), None)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    image_data = await file.read()

    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": file.content_type, "data": image_data.decode("latin1")}}
            ]
        }]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(GEMINI_API_URL, headers=headers, json=body) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail=await resp.text())
            gemini_response = await resp.json()
            result = gemini_response["candidates"][0]["content"]["parts"][0]["text"]

    filename = f"result_{file.filename}"
    supabase.storage.from_(SUPABASE_BUCKET).upload(filename, image_data)
    public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
    return {"image_url": public_url, "gemini_result": result}