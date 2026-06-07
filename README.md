# 🔍 DeepFake Detector

A deep learning web app that detects AI-generated/deepfake face images using PyTorch and Explainable AI (Grad-CAM).

## 🚀 Live Demo
[swetha-deepfake-detector.streamlit.app](https://swetha-deepfake-detector.streamlit.app)

## 🎯 Features
- **99.98% validation accuracy** on 140k face dataset
- **Grad-CAM XAI** — shows exactly which pixels triggered the detection
- **Bias-aware** — identified and addressed racial bias using UTKFace dataset
- **Streamlit UI** — upload any face image for instant Real/Fake analysis

## 🛠️ Tech Stack
| Component | Tool |
|-----------|------|
| Model | EfficientNet-B0 (PyTorch) |
| XAI | Grad-CAM |
| Dataset | 140k Real & Fake Faces + UTKFace |
| UI | Streamlit |
| Training | Google Colab T4 GPU |

## 🏗️ Architecture
