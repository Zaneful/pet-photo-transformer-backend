from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from io import BytesIO
from supabase import create_client, Client

# Vertex AI SDK
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image

app = FastAPI()

# --- Global clients ---
supabase: Client | None = None
generation_model: ImageGenerationModel | None = None

# --- Startup event to initialize clients ---
@app.on_event("startup")
def startup_event():
    global supabase, generation_model

    # Load environment variables
    GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
    GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    IMAGEN_MODEL_NAME = os.getenv("IMAGEN_MODEL_NAME")
    
    # Check required env vars
    if not all([GOOGLE_PROJECT_ID, GOOGLE_LOCATION, SUPABASE_URL, SUPABASE_KEY, IMAGEN_MODEL_NAME]):
        print("ERROR: Missing environment variables:")
        if not GOOGLE_PROJECT_ID: print("  - GOOGLE_PROJECT_ID")
        if not GOOGLE_LOCATION: print("  - GOOGLE_LOCATION")
        if not SUPABASE_URL: print("  - SUPABASE_URL")
        if not SUPABASE_KEY: print("  - SUPABASE_SERVICE_KEY")
        if not IMAGEN_MODEL_NAME: print("  - IMAGEN_MODEL_NAME")
        return

    try:
        print("Initializing Vertex AI and Supabase clients...")
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        print(f"Loading image generation model: {IMAGEN_MODEL_NAME}")
        generation_model = ImageGenerationModel.from_pretrained(IMAGEN_MODEL_NAME)
        print("Clients initialized successfully.")
    except Exception as e:
        print(f"FATAL: Client initialization error: {e}")

# --- Enable CORS for frontend requests ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load prompts from JSON file ---
with open("prompts.json", "r") as f:
    PROMPTS = json.load(f)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Pet Photo Transformer API!"}

@app.get("/prompts")
def get_prompts():
    # Return prompt IDs and titles only
    return [{"id": p["id"], "title": p["title"]} for p in PROMPTS]

@app.post("/generate-image")
async def generate_image(prompt_id: int, file: UploadFile = File(...)):
    # Ensure clients ready
    if not generation_model or not supabase:
        raise HTTPException(status_code=503, detail="Service not ready. AI or Database client failed to initialize.")

    # Find prompt text by ID
    prompt = next((p["promptText"] for p in PROMPTS if p["id"] == prompt_id), None)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    # Read uploaded image bytes
    image_bytes = await file.read()
    source_image = Image(image_bytes=image_bytes)

    try:
        # Generate edited image using Vertex AI model
        response = generation_model.edit_image(
            base_image=source_image,
            prompt=prompt,
            number_of_images=1,
        )
        # Extract raw bytes of generated image (first image)
        new_image_bytes = response.images[0]._image_bytes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    # Upload generated image to Supabase storage
    try:
        filename = f"generated_{prompt_id}_{os.urandom(4).hex()}.png"
        bucket_name = os.getenv("BUCKET")
        if not bucket_name:
            raise ValueError("Environment variable BUCKET is not set for Supabase storage.")

        supabase.storage.from_(bucket_name).upload(
            path=filename,
            file=BytesIO(new_image_bytes),  # Wrap bytes in BytesIO
            file_options={"content-type": "image/png"},
        )

        public_url = supabase.storage.from_(bucket_name).get_public_url(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload to Supabase failed: {e}")

    # Return public URL for frontend to display
    return {"image_url": public_url}
