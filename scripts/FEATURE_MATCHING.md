# Audio Feature Matching Guide

## Overview

The voiceprint model was trained using features extracted from `Complete_Audio_Processing.ipynb` and saved to `audio_features.csv`.

The model in `Voiceprint_Complete_Analysis.ipynb` uses these features by excluding metadata columns:

```python
feature_cols = [col for col in df.columns if col not in 
               ['file_id', 'speaker', 'audio_name', 'augmentation', 'audio_path']]
```

## Feature Structure

The `audio_features.csv` contains:

### Metadata Columns (EXCLUDED from features):
- `file_id`
- `speaker`
- `audio_name`
- `augmentation`
- `audio_path`

### Feature Columns (INCLUDED in model training):
1. **MFCC Features (52 features)**:
   - `mfcc_mean_0` through `mfcc_mean_12` (13 features)
   - `mfcc_std_0` through `mfcc_std_12` (13 features)
   - `mfcc_max_0` through `mfcc_max_12` (13 features)
   - `mfcc_min_0` through `mfcc_min_12` (13 features)

2. **Spectral Features (6 features)**:
   - `spectral_rolloff_mean`
   - `spectral_rolloff_std`
   - `spectral_centroid_mean`
   - `spectral_centroid_std`
   - `spectral_bandwidth_mean`
   - `spectral_bandwidth_std`

3. **Energy Features (5 features)**:
   - `rms_energy_mean`
   - `rms_energy_std`
   - `rms_energy_max`
   - `zcr_mean`
   - `zcr_std`

4. **Chroma Features (24 features)**:
   - `chroma_mean_0` through `chroma_mean_11` (12 features)
   - `chroma_std_0` through `chroma_std_11` (12 features)

5. **Additional Features (4 features)**:
   - `duration`
   - `sample_rate`
   - `max_amplitude`
   - `mean_amplitude`

**Total: 91 numeric features**

## Ensuring Feature Match

### When Saving the Model:

In `Voiceprint_Complete_Analysis.ipynb`, after training (cell 36), save feature columns:

```python
import joblib
import os

# Load the audio_features.csv to get feature structure
df = pd.read_csv('audio_features.csv')  # Or the path to your CSV

# Get feature columns (same as used in training)
feature_cols = [col for col in df.columns if col not in 
               ['file_id', 'speaker', 'audio_name', 'augmentation', 'audio_path']]

# Save feature columns
os.makedirs('../models', exist_ok=True)
joblib.dump(feature_cols, '../models/voiceprint_feature_cols.pkl')

print(f"✅ Saved {len(feature_cols)} feature columns")
print(f"   First 5: {feature_cols[:5]}")
print(f"   Last 5: {feature_cols[-5:]}")
```

### When Extracting Features for Inference:

The `extract_audio_features_for_model()` function in `scripts/feature_extractors.py` uses the same extraction method as `Complete_Audio_Processing.ipynb`:

```python
def extract_all_audio_features(audio_path):
    """
    Extract ALL audio features by combining all individual functions
    Matches the extraction in Complete_Audio_Processing.ipynb
    """
    # Uses same functions:
    # - extract_mfcc_features()
    # - extract_spectral_features()
    # - extract_energy_features()
    # - extract_chroma_features()
    # Plus: duration, sample_rate, max_amplitude, mean_amplitude
```

### Verification:

To verify features match, compare:

1. **Feature names**: Should match column names in `audio_features.csv` (excluding metadata)
2. **Feature count**: Should be 91 numeric features
3. **Feature order**: Should match the order used during training

### Test Feature Extraction:

```python
# Test feature extraction
from scripts.feature_extractors import extract_all_audio_features
import pandas as pd

# Extract features from a test audio file
audio_path = "data/audio_samples/Pam_Yes approve .wav"
features = extract_all_audio_features(audio_path)

# Compare with CSV structure
df_csv = pd.read_csv("audio_features.csv")
expected_cols = [col for col in df_csv.columns if col not in 
                ['file_id', 'speaker', 'audio_name', 'augmentation', 'audio_path']]

print(f"Extracted features: {len(features)}")
print(f"Expected features: {len(expected_cols)}")

# Check if all expected columns are present
missing = set(expected_cols) - set(features.keys())
if missing:
    print(f"⚠ Missing features: {missing}")
else:
    print("✅ All features match!")
```

## Important Notes

1. **Feature Order**: The order of features matters! Make sure `feature_cols` is saved in the same order used during training.

2. **Feature Extraction**: Always use the same extraction functions from `Complete_Audio_Processing.ipynb` to ensure consistency.

3. **Missing Features**: If some features are missing during inference, the system will fill them with 0, but this may affect model performance.

4. **Feature Scaling**: Features must be scaled using the same `StandardScaler` that was fitted during training.

## Quick Fix if Features Don't Match

If you get feature mismatch errors:

1. **Load the original CSV**:
   ```python
   df = pd.read_csv('audio_features.csv')
   feature_cols = [col for col in df.columns if col not in 
                  ['file_id', 'speaker', 'audio_name', 'augmentation', 'audio_path']]
   ```

2. **Save feature columns**:
   ```python
   joblib.dump(feature_cols, 'models/voiceprint_feature_cols.pkl')
   ```

3. **Verify extraction function** uses the same feature structure as `Complete_Audio_Processing.ipynb`

