# Brain Tumor Detection System

A deep learning-based brain tumor detection system using LeNet model architecture and Flask web interface.

![Brain Tumor Detection System](screenshot.png)

## Description

This application allows users to upload brain MRI images and obtain predictions on whether the image shows the presence of a brain tumor. The system uses a deep learning model trained on a dataset of MRI images to provide accurate predictions.

## Features

- Intuitive web interface for uploading MRI images
- Automatic image preprocessing to ensure compatibility with the model
- Binary classification (tumor/non-tumor) with confidence scores
- Comprehensive logging for easy debugging and auditing

## System Requirements

- Python 3.8 or newer
- TensorFlow 2.x
- Flask
- OpenCV
- NumPy
- Pillow (PIL)
- Werkzeug

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
