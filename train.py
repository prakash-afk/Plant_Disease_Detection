
# Training

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = r"data/PlantVillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
VALID_DIR = r"data/PlantVillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

MODEL_SAVE = "models/plant_disease_model.keras"
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10

os.makedirs("models", exist_ok=True)

def get_data():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    return train_gen, val_gen


def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model, base_model


def get_callbacks():
    return [
        ModelCheckpoint(
            MODEL_SAVE,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


def plot_history(history, phase_name):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(phase_name)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_gen, val_gen = get_data()

    print(f"Train classes: {train_gen.num_classes}")
    print(f"Valid classes: {val_gen.num_classes}")

    model, base_model = build_model(train_gen.num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=get_callbacks(),
        verbose=1
    )

    plot_history(history1, "Phase 1")

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        callbacks=get_callbacks(),
        verbose=1
    )

    plot_history(history2, "Phase 2")

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"Final Accuracy: {val_acc * 100:.2f}%")

    model.save(MODEL_SAVE)
    print(f"Model saved at: {MODEL_SAVE}")