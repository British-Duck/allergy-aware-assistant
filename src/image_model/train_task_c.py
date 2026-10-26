# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:36:55 2026

@author: Jack Street
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =========================
# CONFIG
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "task_c_dataset")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(MODEL_DIR, "food_classifier.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
METRICS_PATH = os.path.join(RESULTS_DIR, "training_metrics.txt")
CM_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
TRAINING_PLOT_PATH = os.path.join(RESULTS_DIR, "training_history.png")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


# =========================
# DATA LOADING
# =========================
def build_datasets():
    """
    Load train / val / test datasets from folder structure.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    # Performance optimisation
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(autotune)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(autotune)
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


# =========================
# MODEL
# =========================
def build_model(num_classes: int):
    """
    Transfer-learning model using MobileNetV2.
    """
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# PLOTS / SAVE HELPERS
# =========================
def save_label_map(class_names):
    label_map = {i: name for i, name in enumerate(class_names)}
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)


def save_training_plot(history):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(TRAINING_PLOT_PATH, dpi=200)
    plt.close()


def evaluate_and_save_report(model, test_ds, class_names):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # Save report
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("TASK C - FOOD IMAGE CLASSIFIER RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    # Save confusion matrix image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title("Task C Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=200)
    plt.close()

    print("\n=== Classification Report ===")
    print(report)
    print("\nConfusion matrix saved to:", CM_PATH)
    print("Metrics report saved to:", METRICS_PATH)


# =========================
# MAIN
# =========================
def main():
    print("Loading Task C datasets...")
    train_ds, val_ds, test_ds, class_names = build_datasets()

    print("Classes found:", class_names)
    save_label_map(class_names)

    print("Building model...")
    model = build_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    print("Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("Saving training plot...")
    save_training_plot(history)

    print("Evaluating on test set...")
    evaluate_and_save_report(model, test_ds, class_names)

    print("\nTraining complete.")
    print("Saved model:", MODEL_PATH)
    print("Saved label map:", LABEL_MAP_PATH)
    print("Saved training plot:", TRAINING_PLOT_PATH)


if __name__ == "__main__":
    main()