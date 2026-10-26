# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:35:20 2026

@author: Jack Street
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "food_classifier.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

IMG_SIZE = (224, 224)

_model = None
_label_map = None


def load_task_c_model():
    global _model, _label_map

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _label_map is None:
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            _label_map = {int(k): v for k, v in loaded.items()}

    return _model, _label_map


def classify_image(image_path: str):
    """
    Return predicted class name and confidence.
    """
    model, label_map = load_task_c_model()

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    confidence = float(preds[best_idx])
    predicted_label = label_map[best_idx]

    return predicted_label, confidence