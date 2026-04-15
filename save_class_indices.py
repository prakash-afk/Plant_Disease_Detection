import os
import json

TRAIN_DIR = r"data/PlantVillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

class_names = sorted(os.listdir(TRAIN_DIR))
class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

os.makedirs("models", exist_ok=True)

with open("models/class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=4)

print("class_indices.json created successfully")