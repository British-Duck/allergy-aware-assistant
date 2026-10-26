# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:42:08 2026

@author: Jack Street
"""
import os
from typing import List, Tuple, Dict

from google.cloud import vision

from image_model.task_c_inference import classify_image


# -------------------------
# Task C extension:
# Google Cloud Vision
# -------------------------

# Mapping from raw Vision labels to your coursework class set
VISION_KEYWORD_MAP = {
    "pizza": "pizza",
    "sushi": "sushi",
    "salad": "greek_salad",
    "greek salad": "greek_salad",
    "hamburger": "hamburger",
    "burger": "hamburger",
    "ice cream": "ice_cream",
    "frozen dessert": "ice_cream",
    "steak": "steak",
    "beef": "steak",
    "meat": "steak",
}


def _normalise_text(text: str) -> str:
    return text.strip().lower()


def _map_cloud_labels_to_coursework_classes(labels: List[Tuple[str, float]]) -> Tuple[str, float]:
    """
    Convert Google Vision labels into one of the six coursework classes.
    Returns (predicted_class, confidence).

    If nothing maps cleanly, return ("unknown", 0.0).
    """
    class_scores: Dict[str, float] = {
        "pizza": 0.0,
        "sushi": 0.0,
        "greek_salad": 0.0,
        "hamburger": 0.0,
        "ice_cream": 0.0,
        "steak": 0.0,
    }

    for raw_label, score in labels:
        label_norm = _normalise_text(raw_label)

        for keyword, target_class in VISION_KEYWORD_MAP.items():
            if keyword in label_norm:
                class_scores[target_class] = max(class_scores[target_class], score)

    best_class = max(class_scores, key=class_scores.get)
    best_score = class_scores[best_class]

    if best_score <= 0:
        return "unknown", 0.0

    return best_class, best_score


def cloud_classify_image(image_path: str):
    """
    Use Google Cloud Vision label detection on a local image file.

    Returns:
        predicted_class: str
        confidence: float
        raw_labels: list of (label, score)
    """
    # Important for ADC-based local development:
    # do not force GOOGLE_APPLICATION_CREDENTIALS here.
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Official client library pattern:
    # response = client.label_detection(image=image)
    response = client.label_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    raw_labels = []
    for label in response.label_annotations:
        raw_labels.append((label.description, float(label.score)))

    predicted_class, confidence = _map_cloud_labels_to_coursework_classes(raw_labels)

    return predicted_class, confidence, raw_labels


def compare_local_and_cloud(image_path: str):
    """
    Run both the local coursework model and the cloud extension.
    """
    local_label, local_conf = classify_image(image_path)
    cloud_label, cloud_conf, raw_labels = cloud_classify_image(image_path)

    return {
        "local_label": local_label,
        "local_confidence": local_conf,
        "cloud_label": cloud_label,
        "cloud_confidence": cloud_conf,
        "raw_cloud_labels": raw_labels[:5],  # top 5 for readability
    }
