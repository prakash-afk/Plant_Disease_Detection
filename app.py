import io
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
MODEL_PATH = "models/plant_disease_model.keras"
CLASS_PATH = "models/class_indices.json"
IMG_SIZE = (224, 224)

# ─────────────────────────────────────
# LOAD MODEL ON START
# ─────────────────────────────────────
app = FastAPI(title="Plant Disease Detection API")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading class labels...")
with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# ─────────────────────────────────────
# IMAGE PREPROCESS
# ─────────────────────────────────────
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ─────────────────────────────────────
# ROUTES
# ─────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)

        predictions = model.predict(image)
        pred_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return JSONResponse({
            "predicted_class": class_names[pred_index],
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)