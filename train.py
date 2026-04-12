"""
PLANT DISEASE DETECTION — MODEL TRAINING
==========================================
Imports data generators from pre_processing.py
Run on Google Colab with GPU for best results.
Runtime → Change runtime type → T4 GPU
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# import from our own pre_processing.py
from pre_processing import create_data_generators

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_SAVE    = "models/plant_disease_model.h5"
IMG_SIZE      = (224, 224)
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
NUM_CLASSES   = 38

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
def build_model(num_classes):
    print("\n🏗️  Building MobileNetV2 model...\n")

    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    print(f"✅ Model built!")
    print(f"   Base model params : {base_model.count_params():,} (frozen)")
    return model, base_model

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks():
    return [
        ModelCheckpoint(
            MODEL_SAVE,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

# ─────────────────────────────────────────────
# PLOT HISTORY
# ─────────────────────────────────────────────
def plot_history(history, phase_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History — {phase_name}', fontsize=14)

    ax1.plot(history.history['accuracy'],     label='Train Accuracy', color='#2d6a35')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy',   color='#a8d5a2')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train Loss', color='#c0622b')
    ax2.plot(history.history['val_loss'], label='Val Loss',   color='#f5c842')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"models/history_{phase_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"📊 Plot saved → {filename}")
    plt.show()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🌿 PLANT DISEASE DETECTION — MODEL TRAINING")
    print("=" * 60)

    # load data from pre_processing.py
    train_gen, val_gen = create_data_generators()
    model, base_model  = build_model(num_classes=NUM_CLASSES)

    # ── PHASE 1: Feature Extraction ──
    print("\n" + "─" * 60)
    print("🚀 PHASE 1: Feature Extraction (base model frozen)")
    print("─" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1
    )

    plot_history(history1, "Phase 1 Feature Extraction")

    # ── PHASE 2: Fine-Tuning ──
    print("\n" + "─" * 60)
    print("🔬 PHASE 2: Fine-Tuning (last 30 layers unfrozen)")
    print("─" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE2,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1
    )

    plot_history(history2, "Phase 2 Fine Tuning")

    # ── FINAL EVALUATION ──
    print("\n" + "=" * 60)
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"🎯 Final Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"📉 Final Validation Loss     : {val_loss:.4f}")
    print(f"💾 Model saved → {MODEL_SAVE}")
    print("🎉 Training complete!")

    # ── DOWNLOAD MODEL (Colab only) ──
    try:
        from google.colab import files
        files.download(MODEL_SAVE)
        print("📥 Model download started!")
    except ImportError:
        print("💡 Not on Colab — model saved locally.")