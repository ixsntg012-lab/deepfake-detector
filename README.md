# DeepFake Detector 🔍

<div align="center">

**Deep learning web app that detects AI-generated/deepfake face images — with Grad-CAM explainability**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## What It Does

1. Upload any **face image** (JPG/PNG)
2. EfficientNet-B0 CNN analyzes the image
3. Model predicts **REAL or FAKE** with confidence percentage
4. **Grad-CAM heatmap** shows exactly which pixels triggered the decision
5. Detailed explanation of why the model made that prediction

> **Problem:** AI-generated deepfake images are increasingly indistinguishable from real photos. This system detects them with 99.98% accuracy — and explains *why* it made each decision using Explainable AI (Grad-CAM).

---

## Features

| Feature | Description |
|---------|-------------|
| 99.98% accuracy | Trained on 140,000 real and AI-generated face images |
| Grad-CAM XAI | Heatmap overlay showing which pixels triggered detection |
| Confidence score | Probability percentage for every prediction |
| Bias-aware | Identified racial bias, retrained with UTKFace diverse dataset |
| Clean dark UI | Original image + heatmap + overlay side by side |
| Any face image | Upload JPG or PNG for instant analysis |

---

## How It Works

```
User uploads face image
        │
        ▼
Preprocessing — Resize 224×224 → Normalize (ImageNet stats)
        │
        ▼
EfficientNet-B0 (pretrained ImageNet, fine-tuned on 140k faces)
  Conv blocks → GlobalAvgPool → [1280 features]
  → Dropout(0.3) → Linear(1280→1) → Sigmoid
        │
        ▼
Prediction — REAL / FAKE + Confidence %
        │
        ▼
Grad-CAM — hooks into final conv layer
  Gradients × Activations → 7×7 attention map
  → Resize to 224×224 → JET colormap overlay
        │
        ▼
Streamlit UI — Original | Heatmap | Overlay side by side
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | EfficientNet-B0 (PyTorch, pretrained ImageNet) |
| XAI | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| Training Dataset | 140k Real & Fake Faces (Kaggle) + UTKFace (diversity) |
| Image Processing | OpenCV, PIL, torchvision transforms |
| UI | Streamlit |
| Training Hardware | Google Colab T4 GPU |

---

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **99.98%** |
| Train Loss (Epoch 5) | 0.0017 |
| Val Loss (Epoch 5) | 0.0007 |
| Total Parameters | 4,008,829 |
| Training Time | ~20 minutes (T4 GPU) |
| Training Images | 120,000 (100k original + 20k UTKFace) |

---

## Installation

```bash
git clone https://github.com/ixsntg012-lab/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
```

Place your trained model file in the project root:
```
deepfake-detector/
├── app.py
├── deepfake_detector_v2.pth   ← trained model
└── requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```

Open browser: `http://localhost:8501`

---

## Project Structure

```
deepfake-detector/
│
├── app.py                      ← Streamlit app + Grad-CAM
├── deepfake_detector_v2.pth    ← Trained model weights
├── requirements.txt
└── README.md
```

Training notebook (Google Colab):
```
Cell 1  — Kaggle setup + kaggle.json upload
Cell 2  — Dataset download (140k faces + UTKFace)
Cell 3  — Data verification
Cell 4  — Packages + transforms + DataLoaders
Cell 5  — UTKFace custom dataset (bias fix)
Cell 6  — EfficientNet-B0 model definition
Cell 7  — Training loop (5 epochs)
Cell 8  — Model save + download
```

---

## AI Bias Discovery

During testing, I discovered the model predicted **FAKE for darker skin tones** — including my own photo — due to dataset imbalance in the 140k StyleGAN dataset (predominantly light-skinned faces).

**Fix applied:** Retrained with UTKFace dataset (23k images across White, Black, Asian, Indian ethnicities) — adding 20k diverse real faces.

**Lesson learned:** 99.98% benchmark accuracy does not guarantee fairness across demographic groups. Diverse, representative training data is non-negotiable for production AI systems.

---

## Limitations & Future Work

**Known Limitations**
- Optimized for StyleGAN-generated fakes; newer GANs (Stable Diffusion, Midjourney) may differ
- Best results with clear frontal face portraits
- Skin tone bias documented and partially addressed — ongoing improvement

**Phase 1 — Fairness**
- [ ] FairFace dataset integration for complete bias mitigation
- [ ] Per-demographic accuracy reporting

**Phase 2 — Detection Scope**
- [ ] Two image comparison mode (Real vs Fake side by side)
- [ ] Newer GAN detection (Stable Diffusion, Midjourney artifacts)
- [ ] Video frame-by-frame deepfake analysis

**Phase 3 — Production**
- [ ] Face detection preprocessing (auto-crop + align faces)
- [ ] REST API endpoint for integration
- [ ] Batch image processing

---

## Author

**Swetha Kiran Veernapu**  
MS Computer Science — University of Central Florida  
[GitHub](https://github.com/ixsntg012-lab)

---

## License

MIT License
