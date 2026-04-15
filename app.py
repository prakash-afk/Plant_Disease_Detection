import io
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

MODEL_PATH = "models/plant_disease_model.keras"
CLASS_PATH = "models/class_indices.json"
IMG_SIZE = (224, 224)

app = FastAPI(title="Plant Disease Detection API")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading class labels...")
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_indices = json.load(f)

# Sort by index to guarantee correct class order
class_names = [None] * len(class_indices)
for class_name, index in class_indices.items():
    class_names[index] = class_name


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_batch = np.array(image, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_batch, axis=0)
    return image_batch


@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)

        predictions = model.predict(image, verbose=0)[0]

        top_k = 3
        top_indices = np.argsort(predictions)[::-1][:top_k]

        top_predictions = []
        for idx in top_indices:
            top_predictions.append(
                {
                    "class_name": class_names[int(idx)],
                    "confidence": float(predictions[int(idx)] * 100),
                }
            )

        best_prediction = top_predictions[0]

        return JSONResponse(
            {
                "predicted_class": best_prediction["class_name"],
                "confidence": best_prediction["confidence"],
                "top_predictions": top_predictions,
            }
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
