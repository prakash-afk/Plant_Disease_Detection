import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

VALID_DIR = r"data/PlantVillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
MODEL_PATH = r"models/plant_disease_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading validation data...")
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

val_gen = val_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nRunning evaluation...")
predictions = model.predict(val_gen, verbose=1)

y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes
class_names = list(val_gen.class_indices.keys())

# Accuracy
accuracy = np.mean(y_pred == y_true)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n", report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(20, 16))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
plt.yticks(range(len(class_names)), class_names, fontsize=6)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

print("\nEvaluation done.")