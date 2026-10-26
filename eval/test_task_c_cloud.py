# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:43:06 2026

@author: Jack Street
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from image_model.task_c_cloud import cloud_classify_image, compare_local_and_cloud

SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "task_c_sample_images")


def main():
    print("=== Task C Cloud Extension Test ===")

    files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not files:
        print("No sample images found.")
        return

    for filename in sorted(files):
        image_path = os.path.join(SAMPLE_DIR, filename)

        cloud_label, cloud_conf, raw_labels = cloud_classify_image(image_path)
        compare = compare_local_and_cloud(image_path)

        print("-" * 70)
        print("File:", filename)
        print(f"Cloud only -> {cloud_label} ({cloud_conf:.2%})")
        print(f"Local vs cloud -> local={compare['local_label']} ({compare['local_confidence']:.2%}), "
              f"cloud={compare['cloud_label']} ({compare['cloud_confidence']:.2%})")
        print("Top raw cloud labels:")
        for label, score in compare["raw_cloud_labels"]:
            print(f"   {label}: {score:.2%}")


if __name__ == "__main__":
    main()