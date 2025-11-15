# Downloading Models from Google Colab

If you saved models in Google Colab, you need to download them to your local project.

## Option 1: Download Files from Colab (Easiest)

### After saving the models, run this code in a new Colab cell:

```python
from google.colab import files
import os

# Files to download
files_to_download = [
    '/models/voiceprint_model.pkl',
    '/models/voiceprint_scaler.pkl',
    '/models/voiceprint_label_encoder.pkl',
    '/models/voiceprint_feature_cols.pkl'
]

print("Downloading files...")
for filepath in files_to_download:
    if os.path.exists(filepath):
        files.download(filepath)
        print(f"✅ Downloaded: {os.path.basename(filepath)}")
    else:
        print(f"❌ Not found: {filepath}")
```

This will download each file to your computer. Then:

1. Create a `models/` folder in your project root if it doesn't exist
2. Move the downloaded `.pkl` files into the `models/` folder

## Option 2: Save Directly to Project Directory (Better)

If your project is synced with Google Drive or GitHub, save directly to the project:

### Modified Save Code for Colab:

```python
import joblib
import os
from pathlib import Path

print("="*60)
print("SAVING VOICEPRINT MODEL")
print("="*60)

# Determine project root (adjust based on your setup)
# If using Google Drive:
from google.colab import drive
drive.mount('/content/drive')

# Update this path to your project directory in Drive
project_root = '/content/drive/MyDrive/Formative-2-Multimodal-Data-Preprocessing-1'
model_dir = os.path.join(project_root, 'models')

# OR if using GitHub clone:
# project_root = '/content/Formative-2-Multimodal-Data-Preprocessing-1'
# model_dir = os.path.join(project_root, 'models')

# Create directory
os.makedirs(model_dir, exist_ok=True)
print(f"Using model directory: {model_dir}")

# Create feature_cols if it doesn't exist
if 'feature_cols' not in locals():
    exclude_cols = ['file_id', 'speaker', 'audio_name', 'augmentation', 'audio_path']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"Created feature_cols: {len(feature_cols)} columns")

# Save model
model_path = os.path.join(model_dir, 'voiceprint_model.pkl')
print(f"\n1. Saving model to: {model_path}")
joblib.dump(lr_model, model_path)
if os.path.exists(model_path):
    size_kb = os.path.getsize(model_path) / 1024
    print(f"   ✅ Saved ({size_kb:.2f} KB)")
else:
    print(f"   ❌ FAILED!")

# Save scaler
scaler_path = os.path.join(model_dir, 'voiceprint_scaler.pkl')
print(f"\n2. Saving scaler to: {scaler_path}")
joblib.dump(scaler, scaler_path)
if os.path.exists(scaler_path):
    size_kb = os.path.getsize(scaler_path) / 1024
    print(f"   ✅ Saved ({size_kb:.2f} KB)")
else:
    print(f"   ❌ FAILED!")

# Save label encoder
encoder_path = os.path.join(model_dir, 'voiceprint_label_encoder.pkl')
print(f"\n3. Saving label encoder to: {encoder_path}")
joblib.dump(label_encoder, encoder_path)
if os.path.exists(encoder_path):
    size_kb = os.path.getsize(encoder_path) / 1024
    print(f"   ✅ Saved ({size_kb:.2f} KB)")
else:
    print(f"   ❌ FAILED!")

# Save feature columns
feature_cols_path = os.path.join(model_dir, 'voiceprint_feature_cols.pkl')
print(f"\n4. Saving feature columns to: {feature_cols_path}")
joblib.dump(feature_cols, feature_cols_path)
if os.path.exists(feature_cols_path):
    size_kb = os.path.getsize(feature_cols_path) / 1024
    print(f"   ✅ Saved {len(feature_cols)} feature columns ({size_kb:.2f} KB)")
else:
    print(f"   ❌ FAILED!")

print("\n✅ ALL MODEL FILES SAVED SUCCESSFULLY!")
print(f"\nLocation: {model_dir}")
```

## Option 3: Zip and Download (All at Once)

```python
import zipfile
import os
from google.colab import files

# Create a zip file with all model files
zip_path = '/content/voiceprint_models.zip'
with zipfile.ZipFile(zip_path, 'w') as zipf:
    model_files = [
        '/models/voiceprint_model.pkl',
        '/models/voiceprint_scaler.pkl',
        '/models/voiceprint_label_encoder.pkl',
        '/models/voiceprint_feature_cols.pkl'
    ]
    
    for filepath in model_files:
        if os.path.exists(filepath):
            zipf.write(filepath, os.path.basename(filepath))
            print(f"Added: {os.path.basename(filepath)}")

# Download the zip file
files.download(zip_path)
print("\n✅ Downloaded voiceprint_models.zip")
print("Extract the files and place them in your project's models/ folder")
```

## After Downloading:

1. **Create models folder** in your project root (if it doesn't exist):
   ```
   Formative-2-Multimodal-Data-Preprocessing-1/
   └── models/
       ├── voiceprint_model.pkl
       ├── voiceprint_scaler.pkl
       ├── voiceprint_label_encoder.pkl
       └── voiceprint_feature_cols.pkl
   ```

2. **Verify the files** are in place:
   ```bash
   python scripts/VERIFY_MODELS.py
   ```

3. **Run the system demonstration**:
   ```bash
   python scripts/system_demonstration.py
   ```

---

## Quick Download Code (Copy-Paste Ready)

Run this in a new Colab cell after saving:

```python
from google.colab import files
import os

# Download voiceprint model files
model_files = {
    'voiceprint_model.pkl': '/models/voiceprint_model.pkl',
    'voiceprint_scaler.pkl': '/models/voiceprint_scaler.pkl',
    'voiceprint_label_encoder.pkl': '/models/voiceprint_label_encoder.pkl',
    'voiceprint_feature_cols.pkl': '/models/voiceprint_feature_cols.pkl'
}

print("Downloading model files...")
for local_name, colab_path in model_files.items():
    if os.path.exists(colab_path):
        files.download(colab_path)
        print(f"✅ {local_name}")
    else:
        print(f"❌ {local_name} not found at {colab_path}")

print("\n✅ All files downloaded!")
print("Place them in your project's models/ folder")
```

