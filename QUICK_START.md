# Quick Start Guide - System Demonstration

## ✅ What Has Been Implemented

All code for the system demonstration is ready! You just need to:

1. **Save models from notebooks** (5 minutes)
2. **Run the system** (1 minute)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Save Models from Notebooks

Open each notebook and run the save code (see details in `scripts/README_SAVE_MODELS.md`):

#### Facial Recognition Model:
```python
# In models/Facial_Recognition_Data_Preprocessing.ipynb, after cell 9:
import joblib
import os
os.makedirs('../models', exist_ok=True)
joblib.dump(rf_model, '../models/face_recognition_model.pkl')
face_data = {'X': X.values if isinstance(X, pd.DataFrame) else X,
             'y': y.values if isinstance(y, pd.Series) else y}
joblib.dump(face_data, '../models/face_embeddings_data.pkl')
print("✅ Face recognition model saved!")
```

#### Voiceprint Model:
```python
# In models/Voiceprint_Complete_Analysis.ipynb, after cell 36:
import joblib
import os
os.makedirs('../models', exist_ok=True)
joblib.dump(lr_model, '../models/voiceprint_model.pkl')
joblib.dump(scaler, '../models/voiceprint_scaler.pkl')
joblib.dump(label_encoder, '../models/voiceprint_label_encoder.pkl')
if 'feature_cols' not in locals():
    feature_cols = [col for col in df.columns if col not in 
                   ['file_id', 'speaker', 'audio_name', 'augmentation', 'audio_path']]
joblib.dump(feature_cols, '../models/voiceprint_feature_cols.pkl')
print("✅ Voiceprint model saved!")
```

#### Product Model:
```python
# In models/product_recommendation_model.ipynb, after cell 27:
import joblib
import os
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
os.makedirs('../models', exist_ok=True)
joblib.dump(best_model, '../models/product_recommendation_model.pkl')
joblib.dump(label_encoder, '../models/product_label_encoder.pkl')
feature_info = {'numerical_cols': numerical_cols, 'categorical_cols': categorical_cols, 'target': 'product_category'}
joblib.dump(feature_info, '../models/product_feature_info.pkl')
print("✅ Product recommendation model saved!")
```

### Step 2: Install Dependencies (if needed)

```bash
pip install deepface opencv-python librosa soundfile scikit-learn joblib pandas numpy
```

### Step 3: Run the System

```bash
python scripts/system_demonstration.py
```

---

## 📋 System Demonstration Flow

The system implements this flow:

```
1. Input Face Image
   ↓
   Face Recognition (RandomForest + Facenet)
   ↓
   ✅ Recognized? (confidence ≥ 50%)
   ├─ NO → ❌ ACCESS DENIED
   └─ YES → Continue
   
2. Input Voice Sample
   ↓
   Voice Verification (LogisticRegression)
   ↓
   ✅ Verified? (matches identity & confidence ≥ 70%)
   ├─ NO → ❌ ACCESS DENIED
   └─ YES → Continue
   
3. Product Recommendation (RandomForest)
   ↓
   ✅ Display Recommendations → ACCESS GRANTED
```

---

## 🎯 CLI Menu Options

When you run `python scripts/system_demonstration.py`, you'll see:

```
Select an option:
  1. Full Authentication Flow (Authorized User)
  2. Simulate Unauthorized Face Attempt
  3. Simulate Unauthorized Voice Attempt
  4. Test Mode (Uses default test files)
  5. Exit
```

**Recommendation:** Start with **Option 4 (Test Mode)** - it uses default test files automatically.

---

## 📁 Required Files

After saving models, you should have:

```
models/
├── face_recognition_model.pkl
├── face_embeddings_data.pkl
├── voiceprint_model.pkl
├── voiceprint_scaler.pkl
├── voiceprint_label_encoder.pkl
├── voiceprint_feature_cols.pkl
├── product_recommendation_model.pkl
├── product_label_encoder.pkl
└── product_feature_info.pkl
```

---

## 🔍 Test Scenarios

### Test 1: Authorized User (Option 4 - Test Mode)
- Uses default test files
- Expected: ✅ Face recognized → ✅ Voice verified → ✅ Products recommended

### Test 2: Unauthorized Face (Option 2)
- Uses unknown face image
- Expected: ❌ Access denied at Step 1

### Test 3: Unauthorized Voice (Option 3)
- Valid face but wrong voice
- Expected: ✅ Face recognized → ❌ Voice verification failed

---

## 📝 Files Created

All implementation files are in the `scripts/` directory:

- ✅ `scripts/feature_extractors.py` - Feature extraction functions
- ✅ `scripts/system_demonstration.py` - Main CLI application
- ✅ `scripts/save_models.py` - Helper to save models
- ✅ `scripts/README_SAVE_MODELS.md` - Detailed saving instructions

---

## 🎬 Next Steps for Submission

1. ✅ Save all models (Step 1 above)
2. ✅ Test the system (run `python scripts/system_demonstration.py`)
3. ✅ Record video demonstration
4. ✅ Write report describing your approach

---

## ⚡ One-Line Summary

**Save models from notebooks → Run `python scripts/system_demonstration.py` → Test!**

All code is ready. Just save the models and run! 🚀

