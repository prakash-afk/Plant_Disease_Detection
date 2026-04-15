from dotenv import load_dotenv
load_dotenv()

import os
import zipfile

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_NAME = "vipoooool/new-plant-diseases-dataset"
DOWNLOAD_DIR = "data/raw"
EXTRACT_DIR  = "data/PlantVillage"

# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
def download_dataset():
    print("📦 Starting dataset download from Kaggle...")
    print(f"   Dataset : {DATASET_NAME}")
    print(f"   Save to : {DOWNLOAD_DIR}\n")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR,  exist_ok=True)

    os.system(f"kaggle datasets download -d {DATASET_NAME} -p {DOWNLOAD_DIR}")

    print("\n✅ Download complete!")

# ─────────────────────────────────────────────
# UNZIP
# ─────────────────────────────────────────────
def extract_dataset():
    zip_path = os.path.join(DOWNLOAD_DIR, "new-plant-diseases-dataset.zip")

    if not os.path.exists(zip_path):
        print("❌ Zip file not found. Did the download succeed?")
        return

    print(f"📂 Extracting {zip_path} → {EXTRACT_DIR} ...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print("✅ Extraction complete!")

# ─────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────

def verify_dataset():
    print("\n🔍 Verifying dataset structure...\n")
    
    actual_dir = os.path.join("data/PlantVillage", 
                              "New Plant Diseases Dataset(Augmented)",
                              "New Plant Diseases Dataset(Augmented)",
                              "train")

    if not os.path.exists(actual_dir):
        print(f"❌ Directory not found: {actual_dir}")
        return

    classes = sorted(os.listdir(actual_dir))
    print(f"✅ Found {len(classes)} disease classes:\n")

    for cls in classes:
        class_path = os.path.join(actual_dir, cls)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            print(f"   📁 {cls:<45} {num_images} images")

    total = sum(
        len(os.listdir(os.path.join(actual_dir, c)))
        for c in classes
        if os.path.isdir(os.path.join(actual_dir, c))
    )
    print(f"\n📊 Total images: {total:,}")
    
if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    verify_dataset()