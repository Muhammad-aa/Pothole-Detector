# Pothole Detection with YOLOv5

Requirements: Python 3.8+, NVIDIA GPU with CUDA, 8GB+ RAM  

Or just use Google Colab.

Please run in a virtualenv!!!

Setup: pip install -r requirements.txt  
Train: Update `DATASET_PATH` in `pothole_yolov5_train.py`, then run `python pothole_yolov5_train.py`  
Detect: `python pothole_detector.py image.jpg --save` (ensure best.pt in same directory)

Dataset: 665 images by [Atikur Rahman Chitholian](https://github.com/chitholian/Potholes-Detection)]  

3 images to practice also included.


Happy Coding!