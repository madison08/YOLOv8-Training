# YOLOv8-Training

This repository contains a Python project for training a YOLOv8 model using the Ultralytics library. The model is built from scratch and trained using custom data specified in a configuration file.

## Config

1. Activate your virtual environment:

   ```bash
   source env/bin/activate
   ```

2. Install the necessary dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. Add some image for train launch `labelImg`:
   ```bash
   labelImg
   ```
4. place image and image label in training folder

## Training

5. Run the training script:

   ```bash
   python train.py
   ```

## Prediction

1. Run the prediction script:
   ```bash
   python predict.py
   ```
