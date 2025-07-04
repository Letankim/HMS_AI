import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import requests
from io import BytesIO
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from starlette.responses import JSONResponse
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nutrition_prediction_api.log')
    ]
)

class NutritionPredictionError(Exception):
    pass

# Pydantic model for response
class NutritionPredictionResponse(BaseModel):
    food_name: str
    quantity: int
    predictions: Dict[str, float]
    metrics: Dict[str, str]
    is_food_image: bool
    food_confidence: float

app = FastAPI(title="Nutrition Prediction API", description="API for predicting nutritional content of food images")

# Global variables for model and scalera
MODEL_PATH = 'food_nutrition_cnn.keras'
CSV_FILE = 'weights_nutrition.csv'
VAL_CSV_FILE = 'validation_predictions.csv'
YOLO_MODEL_PATH = './resultsV1/yolo11n.pt'

# Initialize Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables")
    raise NutritionPredictionError("GEMINI_API_KEY not found in environment variables")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logging.info("Gemini API initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Gemini API: {e}")
    raise NutritionPredictionError(f"Failed to initialize Gemini API: {e}")

# Load CNN model
try:
    cnn_model = load_model(MODEL_PATH)
    logging.info("CNN model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load CNN model: {e}")
    raise NutritionPredictionError(f"Failed to load CNN model: {e}")

def load_scaler_and_metrics(csv_file: str, val_csv_file: str = None) -> Tuple[StandardScaler, Dict]:
    try:
        if not os.path.exists(csv_file):
            raise NutritionPredictionError(f"CSV file {csv_file} does not exist")
        df = pd.read_csv(csv_file)
        required_columns = ['total_weight', 'calories', 'fat', 'carbs', 'protein']
        if not all(col in df.columns for col in required_columns):
            raise NutritionPredictionError(f"CSV missing required columns: {required_columns}")
        
        labels = df[required_columns].values
        scaler = StandardScaler()
        scaler.fit(labels)
        logging.info("Scaler loaded and fitted successfully")
        
        metrics = {}
        if val_csv_file and os.path.exists(val_csv_file):
            val_df = pd.read_csv(val_csv_file)
            if all(col in val_df.columns for col in required_columns + ['pred_weight', 'pred_calories', 'pred_fat', 'pred_carbs', 'pred_protein']):
                y_true = val_df[required_columns].values
                y_pred = val_df[['pred_weight', 'pred_calories', 'pred_fat', 'pred_carbs', 'pred_protein']].values
                r2 = r2_score(y_true, y_pred, multioutput='raw_values')
                mape = mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
                labels = ['weight', 'calories', 'fat', 'carbs', 'protein']
                metrics = {f'r2_{label}': f"{score:.3f}" for label, score in zip(labels, r2)}
                metrics.update({f'mape_{label}': f"{score * 100:.2f}%" for label, score in zip(labels, mape)})
                logging.info(f"Accuracy metrics calculated: {metrics}")
            else:
                logging.warning("Validation CSV missing required columns for metrics")
        else:
            logging.warning("Validation CSV not provided or does not exist; metrics unavailable")
            metrics = {f'r2_{label}': 'N/A' for label in ['weight', 'calories', 'fat', 'carbs', 'protein']}
            metrics.update({f'mape_{label}': 'N/A' for label in ['weight', 'calories', 'fat', 'carbs', 'protein']})
        
        return scaler, metrics
    except Exception as e:
        raise NutritionPredictionError(f"Failed to load scaler or metrics: {e}")

try:
    scaler, metrics = load_scaler_and_metrics(CSV_FILE, VAL_CSV_FILE)
except Exception as e:
    logging.error(f"Failed to initialize scaler or metrics: {e}")
    raise NutritionPredictionError(f"Failed to initialize scaler or metrics: {e}")

def download_image(url: str) -> BytesIO:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.RequestException as e:
        raise NutritionPredictionError(f"Failed to download image from {url}: {e}")

def preprocess_image(image_source: BytesIO, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    try:
        img = load_img(image_source, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        logging.info("Image preprocessed successfully")
        return img_array
    except Exception as e:
        raise NutritionPredictionError(f"Failed to preprocess image: {e}")

def verify_food_image(image_source: BytesIO) -> Tuple[bool, float]:
    try:
        image_source.seek(0)
        img = Image.open(image_source)
        response = gemini_model.generate_content([
            "Is this an image of food? Provide a confidence score between 0 and 1.",
            {"mime_type": "image/jpeg", "data": image_source.getvalue()}
        ])
        text = response.text.lower()
        is_food = "yes" in text or "food" in text
        # Extract confidence score (assuming Gemini returns a numeric value in the response)
        confidence = 0.5  # Default if not found
        for word in text.split():
            try:
                if 0 <= float(word) <= 1:
                    confidence = float(word)
                    break
            except ValueError:
                continue
        logging.info(f"Gemini food verification: is_food={is_food}, confidence={confidence}")
        return is_food, confidence
    except Exception as e:
        logging.error(f"Gemini food verification failed: {e}")
        raise NutritionPredictionError(f"Gemini food verification failed: {e}")

def detect_food_items(image_source: BytesIO, model_path: str = YOLO_MODEL_PATH) -> Tuple[str, int]:
    try:
        yolo_model = YOLO(model_path)
        logging.info(f"YOLO model {model_path} loaded successfully")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            img = Image.open(image_source)
            img.save(tmp_file.name)
            results = yolo_model.predict(source=tmp_file.name, conf=0.5)
        
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                os.unlink(tmp_file.name)
                logging.info(f"Temporary file {tmp_file.name} deleted successfully")
                break
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(1)
                else:
                    logging.warning(f"Failed to delete temporary file {tmp_file.name} after {max_attempts} attempts")
        
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = results[0].names
            
            unique_classes = np.unique(class_ids, return_counts=True)
            if len(unique_classes[0]) > 0:
                dominant_class_id = unique_classes[0][np.argmax(unique_classes[1])]
                food_name = class_names[dominant_class_id]
                quantity = len(boxes)
                logging.info(f"Detected {quantity} {food_name}(s)")
                return food_name, quantity
            else:
                logging.warning("No valid food classes detected")
                return "Unknown", 0
        else:
            logging.warning("No objects detected in the image")
            return "Unknown", 0
    except Exception as e:
        raise NutritionPredictionError(f"Failed to detect food items: {e}")

def predict_nutrition(model: tf.keras.Model, image_array: np.ndarray, scaler: StandardScaler, image_source: BytesIO) -> Dict:
    try:
        # CNN predictions
        cnn_predictions = model.predict(image_array, verbose=0)
        cnn_predictions = scaler.inverse_transform(cnn_predictions)[0]
        labels = ['total_weight', 'calories', 'fat', 'carbs', 'protein']
        cnn_result = {label: float(max(0, pred)) for label, pred in zip(labels, cnn_predictions)}
        
        # Gemini predictions (for refinement)
        image_source.seek(0)
        gemini_response = gemini_model.generate_content([
            "Analyze this food image and estimate its nutritional content (total weight, calories, fat, carbs, protein).",
            {"mime_type": "image/jpeg", "data": image_source.getvalue()}
        ])
        gemini_text = gemini_response.text
        gemini_result = {}
        for label in labels:
            # Placeholder: Parse Gemini response for nutritional values
            # This assumes Gemini returns values in a structured format; adjust based on actual response
            gemini_result[label] = cnn_result[label]  # Fallback to CNN if parsing fails
            for line in gemini_text.split('\n'):
                if label in line.lower():
                    try:
                        value = float(line.split(':')[-1].strip().split()[0])
                        gemini_result[label] = max(0, value)
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Combine predictions (e.g., weighted average)
        combined_result = {}
        for label in labels:
            # Example: 70% CNN + 30% Gemini
            combined_value = 0.7 * cnn_result[label] + 0.3 * gemini_result.get(label, cnn_result[label])
            combined_result[label] = float(max(0, combined_value))
        
        logging.info(f"Combined predictions: {combined_result}")
        return combined_result
    except Exception as e:
        raise NutritionPredictionError(f"Prediction failed: {e}")

@app.exception_handler(NutritionPredictionError)
async def nutrition_prediction_exception_handler(request, exc: NutritionPredictionError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.post("/predict/", response_model=NutritionPredictionResponse)
async def predict_nutrition_endpoint(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[HttpUrl] = Query(None)
):
    try:
        if (file is None and image_url is None) or (file is not None and image_url is not None):
            raise NutritionPredictionError("Provide either an image file or an image URL, not both or neither")
        
        # Handle image input
        if file:
            image_data = BytesIO(await file.read())
            logging.info(f"Received uploaded file: {file.filename}")
        else:
            image_data = download_image(image_url)
            logging.info(f"Downloaded image from URL: {image_url}")
        
        # Verify if image contains food using Gemini
        is_food, food_confidence = verify_food_image(image_data)
        if not is_food:
            raise NutritionPredictionError("Image does not contain food")
        
        # Preprocess image
        image_array = preprocess_image(image_data)
        
        # Detect food items
        image_data.seek(0)
        food_name, quantity = detect_food_items(image_data)
        
        # Predict nutrition
        image_data.seek(0)
        predictions = predict_nutrition(cnn_model, image_array, scaler, image_data)
        
        # Return response
        return NutritionPredictionResponse(
            food_name=food_name,
            quantity=quantity,
            predictions=predictions,
            metrics=metrics,
            is_food_image=is_food,
            food_confidence=food_confidence
        )
    except NutritionPredictionError as e:
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Nutrition Prediction API. Use /predict/ to upload an image or provide a URL."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)