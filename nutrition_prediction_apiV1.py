import os
import socket
import threading
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

class NutritionPredictionResponse(BaseModel):
    food_name: str
    quantity: int
    predictions: Dict[str, float]
    metrics: Dict[str, str]
    is_food_image: bool
    food_confidence: float

app = FastAPI(title="Nutrition Prediction API", description="API for predicting nutritional content of food images")

MODEL_PATH = 'food_nutrition_cnn.keras'
CSV_FILE = 'weights_nutrition.csv'
VAL_CSV_FILE = 'validation_predictions.csv'
YOLO_MODEL_PATH = './resultsV1/yolo11n.pt'

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

cnn_model = None
yolo_model = None

def get_cnn_model():
    if cnn_model is None:
        raise NutritionPredictionError("CNN model is not ready yet.")
    return cnn_model

def get_yolo_model():
    if yolo_model is None:
        raise NutritionPredictionError("YOLO model is not ready yet.")
    return yolo_model

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

scaler, metrics = load_scaler_and_metrics(CSV_FILE, VAL_CSV_FILE)

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
        confidence = 0.5
        for word in text.split():
            try:
                if 0 <= float(word) <= 1:
                    confidence = float(word)
                    break
            except ValueError:
                continue
        return is_food, confidence
    except Exception as e:
        raise NutritionPredictionError(f"Gemini food verification failed: {e}")

def detect_food_items(image_source: BytesIO) -> Tuple[str, int]:
    try:
        yolo = get_yolo_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            img = Image.open(image_source)
            img.save(tmp_file.name)
            results = yolo.predict(source=tmp_file.name, conf=0.5)
        os.unlink(tmp_file.name)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = results[0].names
            unique_classes = np.unique(class_ids, return_counts=True)
            if len(unique_classes[0]) > 0:
                dominant_class_id = unique_classes[0][np.argmax(unique_classes[1])]
                food_name = class_names[dominant_class_id]
                quantity = len(boxes)
                return food_name, quantity
        return "Unknown", 0
    except Exception as e:
        raise NutritionPredictionError(f"Failed to detect food items: {e}")

def predict_nutrition(model: tf.keras.Model, image_array: np.ndarray, scaler: StandardScaler, image_source: BytesIO) -> Dict:
    try:
        cnn_predictions = model.predict(image_array, verbose=0)
        cnn_predictions = scaler.inverse_transform(cnn_predictions)[0]
        labels = ['total_weight', 'calories', 'fat', 'carbs', 'protein']
        cnn_result = {label: float(max(0, pred)) for label, pred in zip(labels, cnn_predictions)}
        image_source.seek(0)
        gemini_response = gemini_model.generate_content([
            "Analyze this food image and estimate its nutritional content (total weight, calories, fat, carbs, protein).",
            {"mime_type": "image/jpeg", "data": image_source.getvalue()}
        ])
        gemini_text = gemini_response.text
        gemini_result = {}
        for label in labels:
            gemini_result[label] = cnn_result[label]
            for line in gemini_text.split('\n'):
                if label in line.lower():
                    try:
                        value = float(line.split(':')[-1].strip().split()[0])
                        gemini_result[label] = max(0, value)
                        break
                    except (ValueError, IndexError):
                        continue
        combined_result = {}
        for label in labels:
            combined_value = 0.7 * cnn_result[label] + 0.3 * gemini_result.get(label, cnn_result[label])
            combined_result[label] = float(max(0, combined_value))
        return combined_result
    except Exception as e:
        raise NutritionPredictionError(f"Prediction failed: {e}")

@app.post("/predict/", response_model=NutritionPredictionResponse)
async def predict_nutrition_endpoint(file: Optional[UploadFile] = File(None), image_url: Optional[HttpUrl] = Query(None)):
    try:
        if (file is None and image_url is None) or (file and image_url):
            raise NutritionPredictionError("Provide either an image file or an image URL, not both or neither")
        image_data = BytesIO(await file.read()) if file else download_image(image_url)
        is_food, food_confidence = verify_food_image(image_data)
        if not is_food:
            raise NutritionPredictionError("Image does not contain food")
        image_array = preprocess_image(image_data)
        image_data.seek(0)
        food_name, quantity = detect_food_items(image_data)
        image_data.seek(0)
        cnn = get_cnn_model()
        predictions = predict_nutrition(cnn, image_array, scaler, image_data)
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

@app.on_event("startup")
async def load_models_background():
    def load_models():
        global cnn_model, yolo_model
        try:
            logging.info("🔄 Loading CNN model...")
            cnn_model = load_model(MODEL_PATH)
            logging.info("✅ CNN model loaded.")
            logging.info("🔄 Loading YOLO model...")
            yolo_model = YOLO(YOLO_MODEL_PATH)
            logging.info("✅ YOLO model loaded.")
        except Exception as e:
            logging.error(f"❌ Failed to load models: {e}")
    threading.Thread(target=load_models).start()

@app.on_event("startup")
async def check_port_open():
    port = int(os.environ.get("PORT", 8000))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        logging.info(f"✅ Port {port} is available and bound successfully.")
        s.close()
    except Exception as e:
        logging.error(f"❌ Port binding failed: {e}")
