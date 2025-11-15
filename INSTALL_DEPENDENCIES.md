# Installation Guide - Dependencies

## Quick Install

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually if you prefer:

```bash
pip install deepface opencv-python librosa soundfile scikit-learn joblib pandas numpy matplotlib seaborn imgaug tqdm
```

## Important Notes

### TensorFlow 2.20+ Requirements

If you have TensorFlow 2.20.0 or newer, you **must** also install `tf-keras`:

```bash
pip install tf-keras
```

**Why?** TensorFlow 2.20+ separates Keras into a separate package (`tf-keras`). DeepFace and RetinaFace require this package.

### Alternative: Downgrade TensorFlow

If you prefer to avoid `tf-keras`, you can downgrade TensorFlow:

```bash
pip install tensorflow==2.19.0
```

However, we recommend using TensorFlow 2.20+ with `tf-keras` as it's the latest stable version.

---

## Package Descriptions

### Core Dependencies

- **deepface** - Face recognition and embedding extraction
- **tensorflow** & **tf-keras** - Deep learning backend
- **opencv-python** - Image processing
- **librosa** - Audio feature extraction
- **scikit-learn** - Machine learning models and preprocessing
- **joblib** - Model serialization
- **pandas** & **numpy** - Data manipulation

### Optional Dependencies

- **shap** - Model interpretation (for product recommendation analysis)
- **imbalanced-learn** - Handling class imbalance (already used in notebooks)
- **xgboost** - Advanced ML model (used in product recommendation)

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'tf_keras'"

**Solution:**
```bash
pip install tf-keras
```

### Error: "dlib installation failed"

**Solution (Windows):**
1. Install Visual Studio Build Tools
2. Or use pre-built wheel:
```bash
pip install dlib-binary
```

**Solution (Mac/Linux):**
```bash
# Mac
brew install cmake
pip install dlib

# Linux
sudo apt-get install cmake
pip install dlib
```

### Error: "librosa installation failed"

**Solution:**
```bash
pip install librosa --upgrade
# If still fails, try:
pip install numba --upgrade
pip install librosa
```

### TensorFlow GPU Support

If you have an NVIDIA GPU and want GPU acceleration:

```bash
pip install tensorflow[and-cuda]
```

Note: Requires CUDA and cuDNN to be installed separately.

---

## Verify Installation

After installing, verify key packages:

```python
import deepface
import cv2
import librosa
import sklearn
import tensorflow as tf
import tf_keras

print("✅ All packages imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"DeepFace version: {deepface.__version__}")
```

---

## Minimal Installation (For Testing Only)

If you only want to test the system without training models:

```bash
pip install joblib pandas numpy scikit-learn
```

This allows you to:
- Load pre-saved models
- Run the system demonstration (if models are already saved)
- **Cannot** train new models or extract features

---

## Colab Installation

If running in Google Colab, most packages are pre-installed. You may only need:

```python
!pip install deepface imgaug noisereduce
```

Then restart runtime to ensure TensorFlow/Keras loads correctly.

---

## Docker (Optional)

If you prefer Docker, here's a sample Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/system_demonstration.py"]
```

---

**Last Updated:** After fixing tf-keras requirement for TensorFlow 2.20+

