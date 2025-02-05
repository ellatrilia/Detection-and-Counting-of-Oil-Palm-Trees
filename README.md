# Detection-and-Counting-of-Oil-Palm-Trees

This tutorial explains how to train, detect, and count oil palm fruits using a pre-trained YOLOv8 model.

<img src="https://drive.google.com/file/d/1sqgAL01v5WR4m9GJ7rI7xzqpzFbrXyKI/view?usp=sharing" width="300" height="auto">

## Prepare Dataset
   * Source dataset: [Oil Palm Trees Annotation with Roboflow](https://universe.roboflow.com/ptc/ptc/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true).
   * The dataset has been annotated using the Roboflow platform.
   * Augmentations applied to improve model robustness:
     * Flip: Horizontal, Vertical
     * 90° Rotate: Clockwise, Counter-Clockwise
     * Rotation: Between -15° and +15
     * Shear: ±10° Horizontal, ±10° Vertical
     * Saturation: Between -25% and +25%
     * Exposure: Between -10% and +10%
     * Blur: Up to 2.5px
     * Noise: Up to 0.1% of pixels
       
   * Download dataset using Roboflow API:
     ```
     !pip install roboflow
     from roboflow import Roboflow
     rf = Roboflow(api_key="38tP3MAn9Msvn367ZMee")
     project = rf.workspace("ptc").project("ptc")
     version = project.version(1)
     dataset = version.download("yolov8")
     ```
## Train the Dataset Using YOLOv8 Pre-Trained Model

Link to train: [Notebook Train](https://colab.research.google.com/github/ellatrilia/Train-Custom-Dataset-With-YOLOv8-Pre-Trained-Model/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb#scrollTo=D2YkphuiaE7_)
  * Make sure the API key is valid, and the dataset version is correctly set before training!
  * Run the training command with YOLOv8:
    ```
    !yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=100 imgsz=800 patience=0 plots=True
    ```
    This command will:
    * Train a YOLOv8 object detection model
    * Use yolov8s.pt as the pre-trained model
    * Train for 100 epochs
    * Use an image size of 800px
    * Set patience to 0 (no early stopping)
    * Generate training plots

  * Download Trained Model
    After training is complete, the best model weights will be saved as best.pt in the Colab Files section. You can download it using:
    ```
    from google.colab import files
    files.download('runs/detect/train/weights/best.pt')
    ```
    **Now, you have a trained YOLOv8 model ready to be used for inference! 🚀**

## Detection & Counting (Jupyter Notebook)
    This repository provides a Python script to detect and count oil palm fruits using a fine-tuned YOLOv8 model.

## Requirements
Make sure you have the following dependencies installed:
```
pip install ultralytics opencv-python numpy ipython
```
## Usage
1. Modify the Paths:
   * Update the model path in the script to the location of your trained YOLOv8 model (best.pt).
   * Update the image path to the location of your test image.
2. Run the Script in Jupyter Notebook
Save the following script as a Jupyter Notebook (.ipynb) and execute it cell by cell.

##  Script Overview
Save the following code as a cell in your Jupyter Notebook:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load model 
model = YOLO(r"C:\Users\asus\Downloads\SawitPRO\Count\Model Count Sawit.pt")  # Update path accordingly
print("Model loaded successfully!")
# Load image
image_path = r"C:\Users\asus\Downloads\SawitPRO\Count\ai_assignment_20241202_count.jpeg"  # Update path accordingly
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Object detection
results = model(image, conf=0.25, iou=0.3, max_det=10000)
num_pohon = len(results[0].boxes)
print(f"Total of oil palm trees detected: {num_pohon}")

# Image annotation
annotated_image = image.copy()
for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), 5)
    cv2.putText(annotated_image, str(i + 1), (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5, cv2.LINE_AA)
# Show image results with the total number of oil palm trees
plt.figure(figsize=(10, 12))
plt.imshow(annotated_image)
plt.axis("off")
plt.title(f"Total of Oil Palm Trees: {num_pohon}", fontsize=12, fontweight="bold")
plt.show()
```
## Output
* The detected oil palm fruits will be shown with bounding boxes.
* The total number of detected palm fruits will be printed in the console.
  
![Image Description](https://drive.google.com/uc?export=view&id=1JUAgzf7OVsMG8NDlI5Uta6cRkbP5LZTk)

## License
This project is open-source and can be freely modified and distributed.





