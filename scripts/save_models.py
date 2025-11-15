"""
Scripts to save trained models from notebooks
Run this after training models in notebooks
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================================
# SAVE FACIAL RECOGNITION MODEL
# ============================================================================

def save_face_recognition_model(notebook_path="models/Facial_Recognition_Data_Preprocessing.ipynb",
                                output_dir="models"):
    """
    Save facial recognition model from notebook
    
    Instructions:
    1. Run the Facial_Recognition_Data_Preprocessing.ipynb notebook
    2. After cell 9 (model training), run this function
    3. It will save:
       - face_recognition_model.pkl (RandomForest model)
       - face_embeddings_data.pkl (training data for confidence calculation)
    """
    
    print("="*60)
    print("SAVING FACIAL RECOGNITION MODEL")
    print("="*60)
    
    # This should be run in the notebook after training
    # The variables rf_model, X, y should be available from the notebook
    
    try:
        # In notebook, after training (cell 9), run:
        # exec(open('scripts/save_models.py').read())
        # save_face_recognition_model()
        
        # Check if variables exist (will be in notebook namespace)
        if 'rf_model' not in globals():
            print("❌ Error: rf_model not found. Please run this in the notebook after training.")
            return
        
        if 'X' not in globals() or 'y' not in globals():
            print("❌ Error: X or y not found. Please load embeddings first.")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, "face_recognition_model.pkl")
        joblib.dump(rf_model, model_path)
        print(f"✅ Saved face recognition model to {model_path}")
        
        # Save embeddings data (for confidence calculation)
        embeddings_data = {
            'X': X.values if isinstance(X, pd.DataFrame) else X,
            'y': y.values if isinstance(y, pd.Series) else y
        }
        data_path = os.path.join(output_dir, "face_embeddings_data.pkl")
        joblib.dump(embeddings_data, data_path)
        print(f"✅ Saved face embeddings data to {data_path}")
        
        print("\n✅ Face recognition model saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving face recognition model: {e}")
        print("\nInstructions:")
        print("1. Open models/Facial_Recognition_Data_Preprocessing.ipynb")
        print("2. Run all cells up to and including cell 9 (model training)")
        print("3. In a new cell, run:")
        print("   from scripts.save_models import save_face_recognition_model")
        print("   save_face_recognition_model()")
        print("   OR copy the save code directly into the notebook")


# ============================================================================
# SAVE VOICEPRINT MODEL
# ============================================================================

def save_voiceprint_model(notebook_path="models/Voiceprint_Complete_Analysis.ipynb",
                          output_dir="models"):
    """
    Save voiceprint verification model from notebook
    
    Instructions:
    1. Run the Voiceprint_Complete_Analysis.ipynb notebook
    2. After cell 36 (model training), run this function
    3. It will save:
       - voiceprint_model.pkl (Logistic Regression model - best performing)
       - voiceprint_scaler.pkl (StandardScaler)
       - voiceprint_label_encoder.pkl (LabelEncoder)
       - voiceprint_feature_cols.pkl (feature column names)
    """
    
    print("="*60)
    print("SAVING VOICEPRINT MODEL")
    print("="*60)
    
    try:
        # Check if variables exist (will be in notebook namespace)
        if 'lr_model' not in globals() and 'models' not in globals():
            print("❌ Error: Models not found. Please run this in the notebook after training.")
            return
        
        if 'X_train_scaled' not in globals() or 'scaler' not in globals():
            print("❌ Error: Scaler not found. Please run preprocessing first.")
            return
        
        if 'label_encoder' not in globals():
            print("❌ Error: LabelEncoder not found. Please run preprocessing first.")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Use Logistic Regression (best model)
        if 'lr_model' in globals():
            best_model = lr_model
        elif 'models' in globals() and 'Logistic Regression' in models:
            best_model = models['Logistic Regression']
        else:
            print("❌ Error: Logistic Regression model not found.")
            return
        
        # Save model
        model_path = os.path.join(output_dir, "voiceprint_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"✅ Saved voiceprint model to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, "voiceprint_scaler.pkl")
        if 'scaler' in globals():
            joblib.dump(scaler, scaler_path)
        else:
            # Recreate scaler from X_train_scaled
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)  # Assuming X_train exists
            joblib.dump(scaler, scaler_path)
        print(f"✅ Saved scaler to {scaler_path}")
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, "voiceprint_label_encoder.pkl")
        joblib.dump(label_encoder, encoder_path)
        print(f"✅ Saved label encoder to {encoder_path}")
        
        # Save feature columns
        if 'feature_cols' in globals():
            feature_cols_path = os.path.join(output_dir, "voiceprint_feature_cols.pkl")
            joblib.dump(feature_cols, feature_cols_path)
            print(f"✅ Saved feature columns to {feature_cols_path}")
        
        print("\n✅ Voiceprint model saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving voiceprint model: {e}")
        import traceback
        traceback.print_exc()
        print("\nInstructions:")
        print("1. Open models/Voiceprint_Complete_Analysis.ipynb")
        print("2. Run all cells up to and including cell 36 (model training)")
        print("3. In a new cell, run:")
        print("   from scripts.save_models import save_voiceprint_model")
        print("   save_voiceprint_model()")


# ============================================================================
# SAVE PRODUCT RECOMMENDATION MODEL
# ============================================================================

def save_product_model(notebook_path="models/product_recommendation_model.ipynb",
                       output_dir="models"):
    """
    Save product recommendation model from notebook
    
    Instructions:
    1. Run the product_recommendation_model.ipynb notebook
    2. After cell 27 (save_artifacts), the model should already be saved
    3. If not, this function will save it again
    """
    
    print("="*60)
    print("SAVING PRODUCT RECOMMENDATION MODEL")
    print("="*60)
    
    try:
        # The notebook already saves the model, but we'll save it again with our naming
        
        if 'results' not in globals() or 'label_encoder' not in globals():
            print("❌ Error: Results or label_encoder not found.")
            print("Please run the notebook up to cell 27 (save_artifacts).")
            return
        
        # Get best model
        if 'best_model' in globals():
            model = best_model
        elif 'results' in globals():
            best_model_name = max(results, key=lambda x: results[x]['f1'])
            model = results[best_model_name]['model']
        else:
            print("❌ Error: Best model not found.")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, "product_recommendation_model.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved product model to {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, "product_label_encoder.pkl")
        joblib.dump(label_encoder, encoder_path)
        print(f"✅ Saved label encoder to {encoder_path}")
        
        # Save feature info
        if 'numerical_cols' in globals() and 'categorical_cols' in globals():
            feature_info = {
                'numerical_cols': numerical_cols,
                'categorical_cols': categorical_cols,
                'target': 'product_category'
            }
            feature_info_path = os.path.join(output_dir, "product_feature_info.pkl")
            joblib.dump(feature_info, feature_info_path)
            print(f"✅ Saved feature info to {feature_info_path}")
        
        print("\n✅ Product recommendation model saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving product model: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: The notebook should have already saved the model.")
        print("Check if 'best_model.pkl' exists and copy it to models/product_recommendation_model.pkl")


# ============================================================================
# NOTEBOOK CODE TO COPY-PASTE
# ============================================================================

NOTEBOOK_CODE_FACE = """
# Copy this code into a new cell in Facial_Recognition_Data_Preprocessing.ipynb
# After running cell 9 (model training)

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
"""

NOTEBOOK_CODE_VOICE = """
# Copy this code into a new cell in Voiceprint_Complete_Analysis.ipynb
# After running cell 36 (model training)

import joblib
import os

# Save voiceprint model (Logistic Regression - best model)
os.makedirs('../models', exist_ok=True)
joblib.dump(lr_model, '../models/voiceprint_model.pkl')

# Save scaler
joblib.dump(scaler, '../models/voiceprint_scaler.pkl')

# Save label encoder
joblib.dump(label_encoder, '../models/voiceprint_label_encoder.pkl')

# Save feature columns
if 'feature_cols' in locals():
    joblib.dump(feature_cols, '../models/voiceprint_feature_cols.pkl')

print("✅ Voiceprint model saved!")
"""

NOTEBOOK_CODE_PRODUCT = """
# Copy this code into a new cell in product_recommendation_model.ipynb
# After running cell 27 (save_artifacts)

import joblib
import os

# The model is already saved, but let's also save with our naming
os.makedirs('../models', exist_ok=True)

# Get best model
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

# Save with our naming
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
"""


if __name__ == "__main__":
    print("="*60)
    print("MODEL SAVING SCRIPTS")
    print("="*60)
    print("\nThese functions should be run from within the notebooks.")
    print("\nAlternatively, copy the code blocks below into your notebooks:\n")
    
    print("━"*60)
    print("FOR FACIAL RECOGNITION NOTEBOOK:")
    print("━"*60)
    print(NOTEBOOK_CODE_FACE)
    
    print("\n" + "━"*60)
    print("FOR VOICEPRINT NOTEBOOK:")
    print("━"*60)
    print(NOTEBOOK_CODE_VOICE)
    
    print("\n" + "━"*60)
    print("FOR PRODUCT RECOMMENDATION NOTEBOOK:")
    print("━"*60)
    print(NOTEBOOK_CODE_PRODUCT)
    
    print("\n" + "="*60)
    print("After saving models, run: python scripts/system_demonstration.py")
    print("="*60)

