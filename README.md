# Multimodal Authentication & Product Recommendation System
## Technical Report

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [System Overview](#4-system-overview)
5. [System Architecture](#5-system-architecture)
6. [Implementation Pipeline](#6-implementation-pipeline)
7. [Conclusion](#7-conclusion)

---

## 1. Introduction

This report presents the design and implementation of a multimodal biometric authentication system integrated with a product recommendation engine. The system combines face recognition and voice verification to provide secure, dual-factor authentication, while simultaneously generating personalized product recommendations based on authenticated user profiles.

In an era where digital security is paramount, traditional authentication methods face significant challenges. Single-factor authentication systems are vulnerable to various attack vectors including credential theft, social engineering, and brute force attacks. This project addresses these vulnerabilities by implementing a multimodal approach that requires multiple biometric factors for user verification.

---

## 2. Problem Statement

### 2.1 Context: Why Multimodal Authentication is Needed

Traditional authentication systems suffer from several critical limitations:

**Limitations of Single-Factor Authentication:**
- **Credential Vulnerability**: Passwords and PINs can be stolen, guessed, or intercepted
- **Phishing Susceptibility**: Users may unknowingly disclose credentials to malicious actors
- **Password Fatigue**: Users often reuse passwords or create weak passwords due to the burden of managing multiple accounts
- **No Intrinsic User Binding**: Possession of credentials does not guarantee the identity of the user

**Benefits of Multimodal Biometric Authentication:**

1. **Enhanced Security**: The combination of face recognition and voice verification creates multiple layers of security. Even if one biometric factor is compromised, the system remains protected by the second factor, significantly reducing the probability of unauthorized access.

2. **Spoofing Resistance**: Unlike passwords that can be stolen or guessed, biometric traits are inherently difficult to replicate. A potential attacker would need to simultaneously spoof both face and voice characteristics, which presents a significantly higher barrier than single-factor systems.

3. **Non-Repudiation**: Biometric authentication provides strong cryptographic evidence of user identity, making it substantially more difficult for users to deny their actions or transactions, which is crucial for financial and legal applications.

4. **User Convenience**: Once enrolled, users do not need to remember complex passwords or carry physical tokens. Their biometric characteristics become their credentials, reducing user friction while maintaining security.

5. **Fraud Prevention**: In financial and e-commerce applications, multimodal authentication helps prevent account takeovers and reduces fraudulent transactions by ensuring that only the legitimate user can access their account.

---

## 3. Objectives

This system is designed to achieve two primary objectives:

### 3.1 User Authentication Objective

Develop a robust, dual-factor biometric authentication system that:
- Verifies user identity through **face recognition** as the primary identification mechanism
- Validates identity through **voice verification** as the secondary authentication factor
- Implements confidence thresholds (50% for face recognition, 70% for voice verification) to ensure high-quality matches
- Provides clear access control with denial of access at any failure point in the pipeline
- Maps user identities between different biometric modalities (face identity "Victoria" maps to voice identity "Pam")

### 3.2 Product Recommendation Objective

Develop an intelligent product recommendation system that:
- Generates personalized product category predictions based on authenticated user profiles
- Utilizes machine learning models trained on customer demographic and behavioral data
- Provides top product category recommendations with confidence scores
- Integrates seamlessly with the authentication pipeline to offer recommendations only to verified users
- Maintains user privacy by generating recommendations before display and only showing them after successful authentication

---

## 4. System Overview

The system implements a sequential multimodal authentication pipeline that combines biometric verification with personalized product recommendations. The pipeline follows a four-stage process:

1. **Face Recognition** → Initial user identification
2. **Product Recommendation** → Generate recommendations (hidden)
3. **Voice Verification** → Secondary authentication
4. **Display Predicted Product** → Show recommendations after full authentication

This design ensures that sensitive product recommendation data is only displayed after successful dual-factor authentication, maintaining both security and user privacy.

---

## 5. System Architecture

### 5.1 Models Used

| Component | Model/Technique | Purpose |
|-----------|----------------|---------|
| **Face Recognition** | Facenet (128D embeddings) + RandomForest Classifier | Extract face features and classify user identity |
| **Voice Verification** | 91 Audio Features + LogisticRegression (Regularized & Calibrated) | Extract voice features and verify speaker identity with 96% accuracy |
| **Product Recommendation** | RandomForest Classifier | Predict product categories based on user data |

### 5.2 Technology Stack

- **Python 3.x**: Core programming language
- **DeepFace**: Face recognition library using Facenet model
- **Librosa**: Audio feature extraction and processing
- **scikit-learn**: Machine learning algorithms (RandomForest, LogisticRegression)
- **OpenCV**: Image processing and manipulation
- **TensorFlow**: Backend for DeepFace deep learning operations
- **joblib**: Model serialization and loading
- **pandas & numpy**: Data manipulation and numerical operations

---

## 6. Implementation Pipeline

### 6.1 System Flowchart

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

### 6.2 Detailed Pipeline Explanation

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

**Input**: Audio sample (WAV/M4A format)


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

### 6.3 Security Features

The implementation includes several security features:

1. **Dual-Factor Biometric Authentication**: Requires both face and voice verification, significantly reducing the attack surface
2. **Confidence Thresholds**: Prevents low-quality matches from being accepted (50% face, 70% voice)
3. **Unknown Speaker Rejection**: Voice model includes "unknown" class to explicitly reject unrecognized speakers
4. **Model Improvements**: Voice verification uses regularized and calibrated Logistic Regression achieving 96% accuracy with realistic confidence scores
5. **Sequential Verification**: Each stage must succeed before proceeding, with immediate denial at any failure point
6. **Identity Mapping Verification**: Ensures voice identity matches face identity, preventing cross-identity attacks
7. **Delayed Information Disclosure**: Product recommendations are generated but only displayed after full authentication, protecting user privacy
8. **Fail-Safe Design**: Any authentication failure results in immediate access denial with no information leakage

---

## 7. Conclusion

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
