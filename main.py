import cv2
import numpy as np
import easyocr
import re
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
import logging
import time
from typing import List
from pydantic import BaseModel
import os 
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Arabization OCR", description="Extract Aarabic text")

# API key for security
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.warning("API_KEY not found in environment variables")
api_key_header = APIKeyHeader(name="X-API-Key")

# Global reader - lazy initialized
reader = None

def get_reader():
    """Lazy initialization of EasyOCR reader to save memory until needed"""
    global reader
    if reader is None:
        start_time = time.time()
        logger.info("Initializing EasyOCR reader...")
        reader = easyocr.Reader(
            ['ar'], 
            gpu=False,
            model_storage_directory='./model',
            download_enabled=True,
            detector=True
        )
        logger.info(f"EasyOCR initialized in {time.time() - start_time:.2f} seconds")
    return reader

class OCRResponse(BaseModel):
    extracted_text: List[str]
    cleaned_text: List[str]
    processing_time: dict
    error: str = None

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Optimize image for OCR processing"""
    try:
        start_time = time.time()
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply light denoising (reduced intensity to save memory)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Apply threshold
        _, threshold = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Resize image (reduced height to save memory)
        h, w = threshold.shape
        new_h = 600  # Reduced from 800
        resized = cv2.resize(threshold, (int(w * new_h / h), new_h))
        
        logger.info(f"Image preprocessing completed in {time.time() - start_time:.2f} seconds")
        return resized
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def extract_text(image: np.ndarray, confidence_threshold: float = 0.5) -> List[str]:
    """Extract text from preprocessed image"""
    try:
        start_time = time.time()
        # Get reader only when needed
        ocr = get_reader()
        
        result = ocr.readtext(
            image,
            detail=1,
            paragraph=True,
            batch_size=2,  # Reduced batch size
            contrast_ths=0.5,
            adjust_contrast=0.7
        )
        
        if not result:
            logger.warning("No text detected in the image")
            return []
            
        extracted_text = []
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                extracted_text.append(item[1])
            else:
                logger.warning(f"Unexpected result format: {item}")
                
        logger.info(f"Text extraction completed in {time.time() - start_time:.2f} seconds")
        return extracted_text
    except Exception as e:
        logger.error(f"Error during text extraction: {e}")
        raise

def clean_text(text_list: List[str]) -> List[str]:
    """Clean and normalize extracted text"""
    try:
        start_time = time.time()
        cleaned = [re.sub(r'[^\u0600-\u06FF\s]', '', text).strip() for text in text_list]
        cleaned = [t for t in cleaned if t]
        logger.info(f"Text cleaning completed in {time.time() - start_time:.2f} seconds")
        return cleaned
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise

@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(
    image: UploadFile = File(...),
    api_key: str = Depends(api_key_header)
):
    """
    Extract text from a medicine box image.
    :param image: Uploaded image file (jpg, png, etc.).
    :param api_key: API key for authentication.
    :return: JSON response with extracted text, cleaned text, and timing.
    """
    # Verify API key
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        total_start_time = time.time()
        
        # Validate image
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image.")

        # Process image
        timing = {}
        start_time = time.time()
        preprocessed_img = preprocess_image(img)
        timing['preprocessing'] = time.time() - start_time

        # Extract text
        start_time = time.time()
        extracted_text = extract_text(preprocessed_img)
        timing['extraction'] = time.time() - start_time

        # Clean text
        start_time = time.time()
        cleaned_text = clean_text(extracted_text)
        timing['cleaning'] = time.time() - start_time

        # Total time
        timing['total'] = time.time() - total_start_time

        return OCRResponse(
            extracted_text=extracted_text,
            cleaned_text=cleaned_text,
            processing_time=timing
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return OCRResponse(
            extracted_text=[],
            cleaned_text=[],
            processing_time={},
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)