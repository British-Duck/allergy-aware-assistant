"""
Re-evaluate the saved Task C model on the held-out test set without retraining.
Prints the same classification report and confusion matrix as train_task_c.py.
"""
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from image_model.train_task_c import TEST_DIR, MODEL_PATH, IMG_SIZE, BATCH_SIZE
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def main():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )
    class_names = test_ds.class_names
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

    model = tf.keras.models.load_model(MODEL_PATH)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
