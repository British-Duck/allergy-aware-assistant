# Data

The image data is not committed to this repository.

## Food image dataset (`data/task_c_dataset/`)

A 6-class subset of **Food-101** (Bossard, Guillaumin and Van Gool, ECCV 2014):
greek_salad, hamburger, ice_cream, pizza, steak, sushi.

| Split | Images per class | Total |
|-------|------------------|-------|
| train | 200 | 1,200 |
| val   | 50  | 300   |
| test  | 30  | 180   |

Food-101 images were collected from foodspotting.com and are distributed for
research use only. Check the licence terms on the dataset page before reuse:
https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

### How to rebuild it

1. Download and extract Food-101 from the link above.
2. Copy the images listed in `data/split_manifest.csv` (split, class, Food-101
   image ID) from `food-101/images/<class>/<id>.jpg` into
   `data/task_c_dataset/<split>/<class>/<id>.jpg`.
3. Run `python src/image_model/train_task_c.py` to retrain.

This reproduces the exact split used for the results in `results/`. Retraining
may still give slightly different numbers because GPU/CPU training is not fully
deterministic.

## Demo images (`data/task_c_sample_images/`)

Twelve Food-101 images (two per class) used by `eval/test_task_c.py`.
Put any `.jpg` food photos here to run that script.

## Text knowledge bases

The QA pairs, logic facts and evaluation sets are small hand-written CSVs and
are committed under `kb/` and `eval/`.
