# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:38:45 2026

@author: Jack Street
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from image_model.task_c_inference import classify_image

SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "task_c_sample_images")


def main():
    print("=== Task C Sample Image Test ===")

    if not os.path.exists(SAMPLE_DIR):
        print("Sample image folder not found:", SAMPLE_DIR)
        return

    files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not files:
        print("No sample images found.")
        return

    for filename in sorted(files):
        path = os.path.join(SAMPLE_DIR, filename)
        label, confidence = classify_image(path)
        print(f"{filename} -> {label} ({confidence:.2%})")


if __name__ == "__main__":
    main()