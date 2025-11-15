# Models Folder Structure

## Current Structure

Your models are organized in the following structure:

```
Models/
├── face_recognition/
│   └── face_recognition_model.pkl
│
├── voiceprint/
│   ├── voiceprint_model.pkl
│   ├── voiceprint_scaler.pkl
│   ├── voiceprint_label_encoder.pkl
│   └── voiceprint_feature_cols.pkl
│
└── product_recommendation/
    ├── best_product_recommendation_model.pkl
    ├── label_encoder.pkl
    └── feature_info.pkl
```

## File Descriptions

### Face Recognition (`Models/face_recognition/`)
- **face_recognition_model.pkl** - RandomForest classifier for face recognition
- **face_embeddings_data.pkl** - (Optional) Training embeddings for confidence calculation

### Voiceprint Verification (`Models/voiceprint/`)
- **voiceprint_model.pkl** - LogisticRegression classifier (best model)
- **voiceprint_scaler.pkl** - StandardScaler for feature normalization
- **voiceprint_label_encoder.pkl** - LabelEncoder for speaker identities
- **voiceprint_feature_cols.pkl** - List of feature column names (91 features)

### Product Recommendation (`Models/product_recommendation/`)
- **best_product_recommendation_model.pkl** - Best performing model (RandomForest)
- **label_encoder.pkl** - LabelEncoder for product categories
- **feature_info.pkl** - Feature information (numerical & categorical columns)

## Alternative Naming Support

The system supports alternative file naming:
- `product_recommendation_model.pkl` OR `best_product_recommendation_model.pkl`
- `product_label_encoder.pkl` OR `label_encoder.pkl`
- `product_feature_info.pkl` OR `feature_info.pkl`

## Verification

To verify all models are in place:

```bash
python scripts/VERIFY_MODELS.py
```

## System Usage

The system demonstration script automatically detects:
- `Models/` directory (capital M) - **Current structure**
- `models/` directory (lowercase) - Legacy structure

Both naming conventions are supported.

