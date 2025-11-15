# How to Save Models from Notebooks

This guide explains how to save the trained models from the notebooks so they can be used in the system demonstration.

## Step 1: Save Facial Recognition Model

1. Open `models/Facial_Recognition_Data_Preprocessing.ipynb`
2. Run all cells up to and including **Cell 9** (Random Forest model training)
3. In a **new cell**, copy and paste this code:

```python
import joblib
import os

# Save face recognition model
os.makedirs('../models', exist_ok=True)
joblib.dump(rf_model, '../models/face_recognition_model.pkl')

# Save embeddings data for confidence calculation
face_data = {
    'X': X.values if isinstance(X, pd.DataFrame) else X,
    'y': y.values if isinstance(y, pd.Series) else y
}
joblib.dump(face_data, '../models/face_embeddings_data.pkl')

print("✅ Face recognition model saved!")
```

4. Run the cell. You should see: `✅ Face recognition model saved!`

**Files created:**
- `models/face_recognition_model.pkl`
- `models/face_embeddings_data.pkl`

---

## Step 2: Save Voiceprint Verification Model

1. Open `models/Voiceprint_Complete_Analysis.ipynb`
2. Run all cells up to and including **Cell 36** (model training)
3. In a **new cell**, copy and paste this code:

```python
import joblib
import os
from pathlib import Path

print("="*60)
print("SAVING VOICEPRINT MODEL")
print("="*60)

# Determine correct models directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Try to find or create models directory
if 'models' in current_dir:
    # We're in models/ directory
    model_dir = os.path.abspath('.')
elif os.path.exists('models'):
    model_dir = os.path.abspath('models')
elif os.path.exists('../models'):
    model_dir = os.path.abspath('../models')
else:
    # Create models directory
    model_dir = os.path.abspath('models')
    os.makedirs(model_dir, exist_ok=True)

print(f"Using model directory: {model_dir}")
os.makedirs(model_dir, exist_ok=True)

# Create feature_cols if it doesn't exist
if 'feature_cols' not in locals():
    exclude_cols = ['file_id', 'speaker', 'audio_name', 'augmentation', 'audio_path']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"Created feature_cols: {len(feature_cols)} columns")

# Save model
model_path = os.path.join(model_dir, 'voiceprint_model.pkl')
print(f"\n1. Saving model to: {model_path}")
try:
    joblib.dump(lr_model, model_path)
    if os.path.exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        print(f"   ✅ Saved ({size_kb:.2f} KB)")
    else:
        print(f"   ❌ FAILED - File not created!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Save scaler
scaler_path = os.path.join(model_dir, 'voiceprint_scaler.pkl')
print(f"\n2. Saving scaler to: {scaler_path}")
try:
    joblib.dump(scaler, scaler_path)
    if os.path.exists(scaler_path):
        size_kb = os.path.getsize(scaler_path) / 1024
        print(f"   ✅ Saved ({size_kb:.2f} KB)")
    else:
        print(f"   ❌ FAILED - File not created!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Save label encoder
encoder_path = os.path.join(model_dir, 'voiceprint_label_encoder.pkl')
print(f"\n3. Saving label encoder to: {encoder_path}")
try:
    joblib.dump(label_encoder, encoder_path)
    if os.path.exists(encoder_path):
        size_kb = os.path.getsize(encoder_path) / 1024
        print(f"   ✅ Saved ({size_kb:.2f} KB)")
    else:
        print(f"   ❌ FAILED - File not created!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Save feature columns
feature_cols_path = os.path.join(model_dir, 'voiceprint_feature_cols.pkl')
print(f"\n4. Saving feature columns to: {feature_cols_path}")
try:
    joblib.dump(feature_cols, feature_cols_path)
    if os.path.exists(feature_cols_path):
        size_kb = os.path.getsize(feature_cols_path) / 1024
        print(f"   ✅ Saved {len(feature_cols)} feature columns ({size_kb:.2f} KB)")
    else:
        print(f"   ❌ FAILED - File not created!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Verify all files
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
required_files = [
    'voiceprint_model.pkl',
    'voiceprint_scaler.pkl', 
    'voiceprint_label_encoder.pkl',
    'voiceprint_feature_cols.pkl'
]
all_exist = True
for filename in required_files:
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"✅ {filename} ({size_kb:.2f} KB)")
    else:
        print(f"❌ {filename} - MISSING!")
        all_exist = False

if all_exist:
    print("\n✅ ALL MODEL FILES SAVED SUCCESSFULLY!")
    print(f"\nLocation: {model_dir}")
    print("\nFull paths:")
    for filename in required_files:
        filepath = os.path.join(model_dir, filename)
        print(f"  • {filepath}")
else:
    print("\n❌ SOME FILES ARE MISSING!")
    print(f"Check directory: {model_dir}")
```

4. Run the cell. You should see: `✅ Voiceprint model saved!`

**Files created:**
- `models/voiceprint_model.pkl`
- `models/voiceprint_scaler.pkl`
- `models/voiceprint_label_encoder.pkl`
- `models/voiceprint_feature_cols.pkl` (if feature_cols variable exists)

**Important:** The `feature_cols` should match the features from `Complete_Audio_Processing.ipynb` which created `audio_features.csv`. The code above automatically creates `feature_cols` if it doesn't exist, but make sure `df` is the loaded `audio_features.csv` dataframe.

---

## Step 3: Save Product Recommendation Model

1. Open `models/product_recommendation_model.ipynb`
2. Run all cells up to and including **Cell 27** (save_artifacts)
3. The model should already be saved, but to save with our naming, in a **new cell**, copy and paste:

```python
import joblib
import os

# Get best model
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

# Save with our naming
os.makedirs('../models', exist_ok=True)
joblib.dump(best_model, '../models/product_recommendation_model.pkl')
joblib.dump(label_encoder, '../models/product_label_encoder.pkl')

# Save feature info
feature_info = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'target': 'product_category'
}
joblib.dump(feature_info, '../models/product_feature_info.pkl')

print("✅ Product recommendation model saved!")
```

4. Run the cell.

**Files created:**
- `models/product_recommendation_model.pkl`
- `models/product_label_encoder.pkl`
- `models/product_feature_info.pkl`

**Alternative:** If the notebook already saved files, just rename them:
- `best_model.pkl` → `models/product_recommendation_model.pkl`
- `label_encoder.pkl` → `models/product_label_encoder.pkl`
- `feature_info.pkl` → `models/product_feature_info.pkl`

---

## Step 4: Verify All Models Are Saved

After saving all models, check that these files exist in the `models/` directory:

```
models/
├── face_recognition_model.pkl
├── face_embeddings_data.pkl
├── voiceprint_model.pkl
├── voiceprint_scaler.pkl
├── voiceprint_label_encoder.pkl
├── voiceprint_feature_cols.pkl (optional)
├── product_recommendation_model.pkl
├── product_label_encoder.pkl
└── product_feature_info.pkl
```

---

## Step 5: Run System Demonstration

Once all models are saved, run:

```bash
python scripts/system_demonstration.py
```

Or from the project root:

```bash
cd scripts
python system_demonstration.py
```

---

## Troubleshooting

### Error: "Models not found"
- Make sure you've run all training cells in the notebooks
- Check that variable names match (rf_model, lr_model, etc.)

### Error: "Directory not found"
- Create the `models/` directory manually if needed:
  ```bash
  mkdir models
  ```

### Error: "feature_cols not found"
- For voiceprint model, create feature_cols from the dataframe before saving
- See instructions in Step 2 above

### Models loaded but predictions fail
- Check that feature extraction matches what the model expects
- Verify scalers and encoders are saved and loaded correctly

