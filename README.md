# Multimodal Authentication & Product Recommendation System



## 1. Introduction

This is a multimodal biometric authentication system integrated with a product recommendation engine. The system combines face recognition and voice verification to provide secure, dual-factor authentication, while simultaneously generating personalized product recommendations based on authenticated user profiles.

In an era where digital security is paramount, traditional authentication methods face significant challenges. Single-factor authentication systems are vulnerable to various attack vectors including credential theft, social engineering, and brute force attacks. This project addresses these vulnerabilities by implementing a multimodal approach that requires multiple biometric factors for user verification.


## 2. System Overview

The pipeline follows a four-stage process:

1. **Face Recognition** → Initial user identification
2. **Product Recommendation** → Generate recommendations (hidden)
3. **Voice Verification** → Secondary authentication
4. **Display Predicted Product** → Show recommendations after full authentication

This design ensures that sensitive product recommendation data is only displayed after successful dual-factor authentication, maintaining both security and user privacy.

---

## 3. System Architecture

### 3.1 Models Used

| Component | Model/Technique | Purpose |
|-----------|----------------|---------|
| **Face Recognition** | Facenet (128D embeddings) + RandomForest Classifier | Extract face features and classify user identity |
| **Voice Verification** | 91 Audio Features + NNs (Regularised & Calibrated) | Extract voice features and verify speaker identity with 96% accuracy |
| **Product Recommendation** | RandomForest Classifier | Predict product categories based on user data |

### 3.2 Technology Stack

- **Python 3.x**: Core programming language
- **DeepFace**: Face recognition library using the Facenet model
- **Librosa**: Audio feature extraction and processing
- **scikit-learn**: Machine learning algorithms (RandomForest, LogisticRegression)
- **OpenCV**: Image processing and manipulation
- **TensorFlow**: Backend for DeepFace deep learning operations
- **joblib**: Model serialization and loading
- **pandas & numpy**: Data manipulation and numerical operations

---

## 4. Implementation Pipeline

### 4.1 System Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                   START: Authentication Process                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  STEP 1: FACE       │
                    │  RECOGNITION        │
                    │                     │
                    │  • Extract face     │
                    │    embeddings       │
                    │  • Classify with    │
                    │    RandomForest     │
                    │  • Threshold: 50%   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Recognized?        │
                    └──┬──────────────┬───┘
                       │              │
                  ┌────▼────┐    ┌───▼─────┐
                  │   YES   │    │   NO    │
                  └────┬────┘    └───┬─────┘
                       │              │
                       │        ┌─────▼────────┐
                       │        │ ACCESS       │
                       │        │ DENIED       │
                       │        └──────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  STEP 2: PRODUCT     │
            │  RECOMMENDATION      │
            │                      │
            │  • Generate product  │
            │    recommendations   │
            │  • Use RandomForest  │
            │    model             │
            │  • Store (hidden)    │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  STEP 3: VOICE       │
            │  VERIFICATION        │
            │                      │
            │  • Extract audio     │
            │    features (91)     │
            │  • Classify with     │
            │    LogisticRegression│
            │  • Threshold: 70%    │
            │  • Verify identity   │
            └──────────┬───────────┘
                       │
            ┌──────────▼──────────┐
            │  Verified?          │
            └──┬──────────────┬───┘
               │              │
          ┌────▼────┐    ┌───▼─────┐
          │   YES   │    │   NO    │
          └────┬────┘    └───┬─────┘
               │              │
               │        ┌─────▼────────┐
               │        │ ACCESS       │
               │        │ DENIED       │
               │        └──────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  STEP 4: DISPLAY         │
    │  PREDICTED PRODUCT       │
    │                          │
    │  • Show top 3 categories │
    │  • Display confidence    │
    │  • ACCESS GRANTED        │
    └──────────────────────────┘
```

### 4.2 Detailed Pipeline Explanation

#### Stage 1: Face Recognition

**Input**: Face image (JPG/PNG format)



**Decision Logic**:
- If confidence ≥ 50%: Proceed to next stage
- If confidence < 50%: Deny access and terminate

**Output**: User identity (e.g., "Victoria") mapped to voice identity (e.g., "Pam")

#### Stage 2: Product Recommendation Generation

**Input**: Authenticated user identity from Stage 1



**Security Feature**:
- Recommendations are **generated but stored internally** - they are not displayed to the user yet
- This ensures voice verification must succeed before sensitive recommendation data is shown

**Output**: Top 3 product categories with confidence scores (hidden)

#### Stage 3: Voice Verification

**Input**: Audio sample 


**Decision Logic**:
- If confidence ≥ 70% **AND** identity matches face identity **AND** not detected as unknown: Proceed to final stage
- If confidence < 70% **OR** identity mismatch **OR** unknown speaker detected: Deny access and terminate

**Output**: Verified speaker identity with confidence score (or "unknown" if unrecognized)

#### Stage 4: Display Predicted Product

**Input**: Previously generated product recommendations from Stage 2

**Process**:
1. The system retrieves the stored recommendations generated in Stage 2
2. Product recommendations are displayed with full details:
   - Top recommended category
   - Confidence score for top prediction
   - Top 3 categories with probability scores

**Output**: User granted full access with personalized product recommendations


---

## 5. Conclusion

This report has presented the design and implementation of a multimodal biometric authentication system integrated with product recommendation capabilities. The system successfully addresses the limitations of traditional single-factor authentication by implementing dual-factor biometric verification using face recognition and voice verification.

**Key Achievements:**


The system demonstrates the practical application of multimodal biometric authentication in a real-world scenario, combining security with user convenience and intelligent recommendation capabilities.


---

## Appendix A: Project Structure

```
├── Models/                    # Trained model files
│   ├── face_recognition/
│   │   └── face_recognition_model.pkl
│   ├── voiceprint/
│   │   ├── voiceprint_model.pkl
│   │   ├── voiceprint_scaler.pkl
│   │   ├── voiceprint_label_encoder.pkl
│   │   └── voiceprint_feature_cols.pkl
│   └── product_recommendation/
│       ├── best_product_recommendation_model.pkl
│       ├── label_encoder.pkl
│       └── feature_info.pkl
├── Notebooks/                 # Training notebooks
│   ├── Facial_Recognition_Data_Preprocessing.ipynb
│   ├── Voiceprint_Complete_Analysis.ipynb
│   └── product_recommendation_model.ipynb
├── scripts/                   # Application scripts
│   ├── system_demonstration.py
│   ├── feature_extractors.py
│   └── save_models.py
└── data/                      # Test data
    ├── images/
    └── audio_samples/
```

---

## Appendix B: Quick Start Guide

```bash
# Install dependencies
pip install -r requirements.txt

# Run system demonstration
python scripts/system_demonstration.py
```

For detailed setup instructions, model training, and usage guides, see the documentation files in the project directory.

---

*This project is part of a formative assessment for multimodal data preprocessing and authentication systems.*
