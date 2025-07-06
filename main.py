from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json, os, base64
from supabase import create_client, Client

# You will need to install the Vertex AI library: pip install google-cloud-aiplatform
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image

app = FastAPI()

# --- Define the clients globally ---
supabase: Client | None = None
generation_model: ImageGenerationModel | None = None

# --- Run this after the app starts ---
@app.on_event("startup")
def startup_event():
    global supabase, generation_model

    # Load environment variables
    GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
    GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    IMAGEN_MODEL_NAME = os.getenv("IMAGEN_MODEL_NAME")

    # Check for missing environment variables
    if not all([GOOGLE_PROJECT_ID, GOOGLE_LOCATION, SUPABASE_URL, SUPABASE_KEY, IMAGEN_MODEL_NAME]):
        print("ERROR: One or more environment variables are missing. Please ensure all are set.")
        if not GOOGLE_PROJECT_ID: print("  - GOOGLE_PROJECT_ID is missing")
        if not GOOGLE_LOCATION: print("  - GOOGLE_LOCATION is missing")
        if not SUPABASE_URL: print("  - SUPABASE_URL is missing")
        if not SUPABASE_KEY: print("  - SUPABASE_KEY is missing")
        if not IMAGEN_MODEL_NAME: print("  - IMAGEN_MODEL_NAME is missing")
        return

    try:
        print("Initializing Supabase and Vertex AI clients...")
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        # Log the model name before attempting to load it
        print(f"Loading model: {IMAGEN_MODEL_NAME}")
        generation_model = ImageGenerationModel.from_pretrained(IMAGEN_MODEL_NAME)

        print("Clients initialized successfully.")
    except Exception as e:
        print(f"FATAL: Error during client initialization: {e}")

# Enable CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load prompts from JSON
with open("prompts.json", "r") as f:
    PROMPTS = json.load(f)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Pet Photo Transformer API!"}

@app.get("/prompts")
def get_prompts():
    return [{"id": p["id"], "title": p["title"]} for p in PROMPTS]

@app.post("/generate-image")
async def generate_image(prompt_id: int, file: UploadFile = File(...)):
    if not generation_model or not supabase:
        raise HTTPException(status_code=503, detail="Service not ready. AI or Database client failed to initialize. Check server logs.")

    prompt = next((p["promptText"] for p in PROMPTS if p["id"] == prompt_id), None)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    image_bytes = await file.read()
    source_image = Image(image_bytes=image_bytes)

    try:
        response = generation_model.edit_image(
            base_image=source_image,
            prompt=prompt,
            number_of_images=1,
        )
        new_image_data_raw = response.images[0]._image_bytes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    try:
        new_image_bytes = new_image_data_raw
        filename = f"generated_{prompt_id}_{os.urandom(4).hex()}.png"
        bucket_name_supabase = os.getenv("BUCKET")
        if not bucket_name_supabase:
            raise ValueError("SUPABASE BUCKET environment variable (BUCKET) is not set.")

        from io import BytesIO
        supabase.storage.from_(bucket_name_supabase).upload(
            file=BytesIO(new_image_bytes),
            path=filename,
            file_options={"content-type": "image/png"}
        )

        public_url = supabase.storage.from_(bucket_name_supabase).get_public_url(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload to Supabase failed: {e}")

    return {"image_url": public_url}

