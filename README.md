# 🌿 Plant Disease Classification using Computer Vision and AI

## 📌 Project Overview
This project implements a machine learning-based computer vision system to automatically detect and classify plant diseases from leaf images using deep learning techniques.

Two models were developed and compared:
- Baseline Convolutional Neural Network (CNN)
- Transfer Learning Model using MobileNetV2

The aim is to evaluate which approach provides better accuracy, stability, and generalisation.

---

## 🎯 Problem Statement
Plant diseases significantly impact agricultural productivity. Traditional detection methods:
- require expert knowledge
- are time-consuming
- are prone to human error

This project aims to develop an automated AI-based solution to classify plant diseases from images efficiently.

---

## 🎯 Objectives
- Develop a deep learning-based image classification system
- Compare CNN and Transfer Learning approaches
- Evaluate models using performance metrics
- Predict plant diseases from unseen images

---

## 📊 Dataset
- Dataset: PlantVillage
- Total Images: ~20,638
- Number of Classes: 16 (diseased + healthy)
- Type: Labelled dataset (supervised learning)

Each class represents a specific plant disease or a healthy leaf.

---

## 🔀 Data Splitting Strategy

| Dataset | Percentage | Images |
|--------|------------|--------|
| Training | 64% | ~13,202 |
| Validation | 16% | ~3,302 |
| Test | 20% | ~4,134 |

---

## ⚙️ System Pipeline

Dataset → Preprocessing → Augmentation → Train/Validation/Test Split → Model Training → Evaluation → Prediction

---

## 🔧 Data Preprocessing
- Resize images to 160 × 160 pixels
- Normalise pixel values
- Apply data augmentation:
  - Horizontal flip
  - Rotation
  - Zoom

Purpose:
- Improve generalisation
- Reduce overfitting

---

## 🧠 Models Implemented

### 🔹 Baseline CNN
- Built from scratch
- Layers:
  - Convolutional layers
  - MaxPooling layers
  - Dense layers
  - Dropout
- Parameters: ~5.4 million

---

### 🔹 Transfer Learning (MobileNetV2)
- Pre-trained on ImageNet
- Base model frozen (no weight updates)
- Custom classification layers added
- Parameters: ~2.7 million

---

## 🏋️ Training Configuration

| Parameter | Value |
|----------|------|
| Epochs | 10 |
| Batch Size | 32 |
| Image Size | 160 × 160 |
| Framework | TensorFlow / Keras |

---

## 📈 Evaluation Metrics
- Accuracy
- Loss
- Validation Accuracy
- Test Accuracy

---

## 📊 Results

| Model | Accuracy | Notes |
|------|--------|------|
| Baseline CNN | ~85–89% | Slight overfitting |
| Transfer Learning | ~88–89% | More stable and consistent |

---

## 🔍 Analysis

Baseline CNN:
- Gradual learning
- Minor fluctuations
- Slight overfitting

Transfer Learning:
- Faster convergence
- Stable learning curves
- Better generalisation

---

## 🏆 Conclusion
Both models performed well. However, the transfer learning model is preferred due to:
- better stability
- improved generalisation
- faster convergence

---

## ⚠️ Limitations
- Dataset contains controlled images
- Real-world performance may vary due to lighting and background differences

---

## 🚀 Future Improvements
- Use real-world datasets
- Fine-tune base model layers
- Deploy as web/mobile application

---

## ▶️ How to Run

Install dependencies:
pip install tensorflow matplotlib pillow

Train baseline model:
python src/train_baseline.py

Train transfer model:
python src/train_transfer.py

Run prediction:
python src/predict.py

---

## 📂 Project Structure

plant_disease_project/
│
├── dataset/
├── models/
├── outputs/
├── sample_images/
├── src/
│   ├── train_baseline.py
│   ├── train_transfer.py
│   ├── predict.py
│
└── README.md

---

## 📚 References
- PlantVillage Dataset
- TensorFlow & Keras Documentation
- MobileNetV2 (Sandler et al., 2018)
- Géron, A. (2019) Hands-On Machine Learning

---

## 👨‍💻 Author
Your Name
Module: Computer Vision and AI
