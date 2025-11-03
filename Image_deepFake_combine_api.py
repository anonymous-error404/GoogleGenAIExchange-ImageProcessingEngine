from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image, ImageOps
import numpy as np
import requests
import easyocr
import validators
from io import BytesIO
import os

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("❌ HF_TOKEN not found in environment variables. Please add it to your .env file.")

# -------------------------------
# FastAPI Configuration
# -------------------------------
app = FastAPI(
    title="🧠 DeepFake + OCR Detection API",
    description="API for detecting AI-generated images and extracting text using EasyOCR + Hugging Face.",
    version="1.0.0"
)

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Constants
# -------------------------------
API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model"
THRESHOLD_FAKE = 0.20

# -------------------------------
# Load EasyOCR (once at startup)
# -------------------------------
print("🔤 Loading EasyOCR model (may take a few seconds)...")
reader = easyocr.Reader(['en'], gpu=False)
print("✅ EasyOCR model loaded successfully!")

# -------------------------------
# Helper Functions
# -------------------------------
def load_image_from_url(url: str) -> Image.Image:
    """Load image from a valid URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {str(e)}")

def query_huggingface_api(image: Image.Image):
    """Send image to Hugging Face inference endpoint."""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg"
    }

    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_bytes = buffered.getvalue()

    try:
        response = requests.post(API_URL, headers=headers, data=img_bytes, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face API Error: {e}")

def classify_image(image: Image.Image):
    """Classify image as Real or Fake based on Hugging Face output."""
    api_results = query_huggingface_api(image)
    if not api_results or not isinstance(api_results, list):
        return {"verdict": "Error", "confidence": 0.0}

    all_results = [(r.get("label", "").lower(), float(r.get("score", 0))) for r in api_results]
    fake_score = max((s for l, s in all_results if "fake" in l), default=0)
    real_score = max((s for l, s in all_results if "real" in l), default=0)

    if fake_score > real_score and fake_score > THRESHOLD_FAKE:
        return {"verdict": "Fake", "confidence": fake_score}
    elif real_score >= fake_score:
        return {"verdict": "Real", "confidence": real_score}
    else:
        return {"verdict": "Error", "confidence": 0.0}

def extract_text_from_image(image: Image.Image):
    """Extract visible text from image using EasyOCR."""
    gray_image = ImageOps.grayscale(image)
    gray_np = np.array(gray_image)
    result = reader.readtext(gray_np, detail=0)
    return "\n".join(result) if result else None

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "🧠 DeepFake + OCR Detection API is running!"}

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(None, description="Upload an image file (.jpg, .jpeg, .png)"),
    url: str = Form(None, description="Or provide an image URL")
):
    """Analyze image for text and deepfake classification."""
    if not file and not url:
        raise HTTPException(status_code=400, detail="Please provide an image file or a valid URL.")

    # Load image
    if file:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        source = file.filename
    elif url:
        if not validators.url(url):
            raise HTTPException(status_code=400, detail="Invalid image URL.")
        image = load_image_from_url(url)
        source = url
    else:
        raise HTTPException(status_code=400, detail="No valid input provided.")

    # Process image
    extracted_text = extract_text_from_image(image)
    classification = classify_image(image)

    # Return result
    return JSONResponse(
        content={
            "source": source,
            "verdict": classification["verdict"],
            "confidence": round(classification["confidence"], 3),
            "extracted_text": extracted_text or "No text detected.",
        }
    )

# -------------------------------
# Cloud Run Compatible Start
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
