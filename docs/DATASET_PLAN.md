# Dataset Plan

## 1. MS COCO
Purpose: general object detection.

Use: pretrained COCO-based YOLO model and/or quantitative evaluation.

Current local file expected:
`data/coco/val2017.zip`

Do not download the 18 GB COCO training images for this laptop unless we later decide
custom training is required.

## 2. TextOCR
Purpose: natural-scene text recognition.

Current local file expected:
`data/textocr/TextOCR_0.1_val.json`

Note: annotations alone do not contain the image pixels. For quantitative OCR
evaluation we may later need the corresponding validation images.

## 3. Indoor Objects Detection
Purpose: indoor obstacle/object testing.

Use after the MVP works.

## 4. Sidewalk Obstacle Detection
Purpose: outdoor obstacle testing.

Optional; use only if the dataset files and licensing/setup are practical.

## Important
Datasets are not automatically training the current application. The MVP uses
pretrained models. We will decide later whether fine-tuning is justified.
