from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json, os, base64
from supabase import create_client, Client

# You will need to install the Vertex AI library: pip install google-cloud-aiplatform
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image

app = FastAPI()

# --- We will define the clients here, but initialize them later ---
supabase: Client | None = None
generation_model: ImageGenerationModel | None = None

# --- NEW: This function will run AFTER the app has started ---
@app.on_event("startup")
def startup_event():
    global supabase, generation_model

    # --- Environment Variables are read inside the startup event ---
    GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
    GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    
    # --- NEW: Get the IMAGEN_MODEL_NAME environment variable ---
    IMAGEN_MODEL_NAME = os.getenv("IMAGEN_MODEL_NAME")

    # Check if all environment variables are loaded
    # Add IMAGEN_MODEL_NAME to your check
    if not all([GOOGLE_PROJECT_ID, GOOGLE_LOCATION, SUPABASE_URL, SUPABASE_KEY, IMAGEN_MODEL_NAME]):
        print("ERROR: One or more environment variables are missing. Please ensure all are set.")
        # Print which ones are missing for better debugging
        if not GOOGLE_PROJECT_ID: print("  - GOOGLE_PROJECT_ID is missing")
        if not GOOGLE_LOCATION: print("  - GOOGLE_LOCATION is missing")
        if not SUPABASE_URL: print("  - SUPABASE_URL is missing")
        if not SUPABASE_KEY: print("  - SUPABASE_KEY is missing")
        if not IMAGEN_MODEL_NAME: print("  - IMAGEN_MODEL_NAME is missing")
        return # Exit the function if variables are missing

    try:
        print("Initializing Supabase and Vertex AI clients...")
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # --- THIS IS THE CRITICAL CHANGE ---
        # Use the environment variable to get the model name
        generation_model = ImageGenerationModel.from_pretrained(IMAGEN_MODEL_NAME)
        
        print("Clients initialized successfully.")
    except Exception as e:
        # This will print the exact error if initialization fails
        print(f"FATAL: Error during client initialization: {e}")
        # Consider re-raising or setting a flag if you want the app to be unhealthy
        # raise  # If you want the app to fail startup completely

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Prompts ---
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
    # Check if the clients were initialized successfully
    if not generation_model or not supabase:
        raise HTTPException(status_code=503, detail="Service not ready. AI or Database client failed to initialize. Check server logs.")

    # 1. Get the prompt text from your JSON file
    prompt = next((p["promptText"] for p in PROMPTS if p["id"] == prompt_id), None)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    # 2. Prepare the user's uploaded image for the AI
    image_bytes = await file.read()
    source_image = Image(image_bytes=image_bytes)

    # 3. Call the Imagen Model to EDIT the image
    try:
        response = generation_model.edit_image(
            base_image=source_image,
            prompt=prompt,
            number_of_images=1,
        )
        # Assuming response.images[0]._image_bytes is already base64 encoded or raw bytes
        # The documentation for `Image.image_bytes` suggests it stores raw bytes.
        # If it's already base64, this line might be slightly different.
        new_image_data_raw = response.images[0]._image_bytes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    # 4. Upload the new image data to Supabase
    try:
        # If new_image_data_raw is raw bytes, no need for base64.b64decode here
        new_image_bytes = new_image_data_raw # Directly use the bytes

        filename = f"generated_{prompt_id}_{os.urandom(4).hex()}.png"

        # IMPORTANT: You're using os.getenv("BUCKET") - ensure this environment variable is set in Render!
        bucket_name_supabase = os.getenv("BUCKET")
        if not bucket_name_supabase:
            raise ValueError("SUPABASE BUCKET environment variable (BUCKET) is not set.")

        supabase.storage.from_(bucket_name_supabase).upload(
            file=new_image_bytes,
            path=filename,
            file_options={"content-type": "image/png"}
        )
        
        public_url = supabase.storage.from_(bucket_name_supabase).get_public_url(filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload to Supabase failed: {e}")

    # 5. Return the URL of the NEWLY GENERATED image
    return {"image_url": public_url}
