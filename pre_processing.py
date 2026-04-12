import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_DIR  = "data/PlantVillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
VALID_DIR  = "data/PlantVillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

# ─────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────
def create_data_generators():
    print("🔧 Setting up data generators...\n")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    validation_set = val_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    print(f"\n✅ Training images   : {training_set.samples:,}")
    print(f"✅ Validation images : {validation_set.samples:,}")
    print(f"✅ Number of classes : {training_set.num_classes}")
    print(f"\n📋 Classes (first 5): {list(training_set.class_indices.keys())[:5]} ...")

    return training_set, validation_set

# ─────────────────────────────────────────────
# VISUALIZE SAMPLES
# ─────────────────────────────────────────────
def visualize_samples(training_set):
    print("\n🖼️  Generating sample visualization...")

    images, labels = next(training_set)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Sample Training Images (after augmentation)", fontsize=14)

    class_names = list(training_set.class_indices.keys())

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        class_idx = np.argmax(labels[i])
        ax.set_title(class_names[class_idx], fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("data/sample_images.png", dpi=100, bbox_inches='tight')
    print("✅ Saved visualization → data/sample_images.png")
    plt.show()

if __name__ == "__main__":
    train_gen, val_gen = create_data_generators()
    visualize_samples(train_gen)