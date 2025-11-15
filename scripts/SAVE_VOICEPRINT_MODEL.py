"""
Standalone script to save voiceprint model from notebook
Run this AFTER training in Voiceprint_Complete_Analysis.ipynb
"""

import joblib
import os
import sys
import pandas as pd
from pathlib import Path

def save_voiceprint_model():
    """
    Save voiceprint model and all related artifacts
    This should be run in the notebook after cell 36 (model training)
    """
    
    print("="*60)
    print("SAVING VOICEPRINT MODEL")
    print("="*60)
    
    # Check if we're in notebook or standalone
    try:
        # Check if required variables exist (notebook namespace)
        if 'lr_model' not in globals() and 'lr_model' not in locals():
            print("❌ Error: lr_model not found!")
            print("   Please run this in the notebook AFTER training the model (cell 36)")
            return False
        
        if 'scaler' not in globals() and 'scaler' not in locals():
            print("❌ Error: scaler not found!")
            print("   Please run preprocessing cells first")
            return False
        
        if 'label_encoder' not in globals() and 'label_encoder' not in locals():
            print("❌ Error: label_encoder not found!")
            print("   Please run preprocessing cells first")
            return False
        
        if 'df' not in globals() and 'df' not in locals():
            print("❌ Error: df (dataframe) not found!")
            print("   Please load audio_features.csv first (cell 5)")
            return False
        
        # Get variables
        model = globals().get('lr_model') or locals().get('lr_model')
        scaler_obj = globals().get('scaler') or locals().get('scaler')
        encoder_obj = globals().get('label_encoder') or locals().get('label_encoder')
        df_obj = globals().get('df') or locals().get('df')
        
        # Determine output directory (models folder relative to project root)
        # If running from notebook, we might be in models/ or root/
        current_dir = os.getcwd()
        print(f"Current directory: {current_dir}")
        
        # Try different paths
        possible_model_dirs = [
            'models',           # If in root
            '../models',        # If in models/ or scripts/
            '../../models',     # If deeper
            './models'          # Explicit relative
        ]
        
        model_dir = None
        for path in possible_model_dirs:
            abs_path = os.path.abspath(path)
            if os.path.isdir(abs_path) or path == 'models':  # Create if doesn't exist
                model_dir = abs_path
                break
        
        # Default to 'models' in current directory or parent
        if model_dir is None:
            # Check if we're in notebooks/ or models/ directory
            if 'notebooks' in current_dir.lower() or 'models' in current_dir.lower():
                model_dir = os.path.abspath('../models')
            else:
                model_dir = os.path.abspath('models')
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        print(f"Using model directory: {model_dir}")
        print()
        
        # Create feature_cols from dataframe
        if 'feature_cols' not in globals() and 'feature_cols' not in locals():
            print("Creating feature_cols from dataframe...")
            exclude_cols = ['file_id', 'speaker', 'audio_name', 'augmentation', 'audio_path']
            feature_cols = [col for col in df_obj.columns if col not in exclude_cols]
            print(f"   Found {len(feature_cols)} feature columns")
        else:
            feature_cols = globals().get('feature_cols') or locals().get('feature_cols')
            print(f"Using existing feature_cols: {len(feature_cols)} columns")
        
        # Save all files
        files_saved = []
        
        # 1. Save model
        model_path = os.path.join(model_dir, 'voiceprint_model.pkl')
        print(f"Saving model to: {model_path}")
        joblib.dump(model, model_path)
        if os.path.exists(model_path):
            files_saved.append(model_path)
            print(f"   ✅ Saved ({os.path.getsize(model_path) / 1024:.2f} KB)")
        else:
            print(f"   ❌ FAILED to save!")
            return False
        
        # 2. Save scaler
        scaler_path = os.path.join(model_dir, 'voiceprint_scaler.pkl')
        print(f"Saving scaler to: {scaler_path}")
        joblib.dump(scaler_obj, scaler_path)
        if os.path.exists(scaler_path):
            files_saved.append(scaler_path)
            print(f"   ✅ Saved ({os.path.getsize(scaler_path) / 1024:.2f} KB)")
        else:
            print(f"   ❌ FAILED to save!")
            return False
        
        # 3. Save label encoder
        encoder_path = os.path.join(model_dir, 'voiceprint_label_encoder.pkl')
        print(f"Saving label encoder to: {encoder_path}")
        joblib.dump(encoder_obj, encoder_path)
        if os.path.exists(encoder_path):
            files_saved.append(encoder_path)
            print(f"   ✅ Saved ({os.path.getsize(encoder_path) / 1024:.2f} KB)")
        else:
            print(f"   ❌ FAILED to save!")
            return False
        
        # 4. Save feature columns
        feature_cols_path = os.path.join(model_dir, 'voiceprint_feature_cols.pkl')
        print(f"Saving feature columns to: {feature_cols_path}")
        joblib.dump(feature_cols, feature_cols_path)
        if os.path.exists(feature_cols_path):
            files_saved.append(feature_cols_path)
            print(f"   ✅ Saved {len(feature_cols)} feature columns ({os.path.getsize(feature_cols_path) / 1024:.2f} KB)")
        else:
            print(f"   ❌ FAILED to save!")
            return False
        
        print()
        print("="*60)
        print("✅ ALL FILES SAVED SUCCESSFULLY!")
        print("="*60)
        print(f"\nSaved {len(files_saved)} files to: {model_dir}")
        print("\nFiles saved:")
        for f in files_saved:
            print(f"  • {os.path.basename(f)}")
        
        # Verify all files exist
        print("\nVerifying files...")
        all_exist = True
        required_files = [
            'voiceprint_model.pkl',
            'voiceprint_scaler.pkl',
            'voiceprint_label_encoder.pkl',
            'voiceprint_feature_cols.pkl'
        ]
        for filename in required_files:
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                print(f"  ✅ {filename}")
            else:
                print(f"  ❌ {filename} - MISSING!")
                all_exist = False
        
        if all_exist:
            print("\n✅ All model files verified!")
            return True
        else:
            print("\n❌ Some files are missing!")
            return False
        
    except Exception as e:
        print(f"\n❌ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return False


# Alternative: Code to copy-paste directly into notebook
NOTEBOOK_CODE = """
# ============================================================
# COPY THIS CODE INTO A NEW CELL IN Voiceprint_Complete_Analysis.ipynb
# Run AFTER cell 36 (model training)
# ============================================================

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
print(f"\\n1. Saving model to: {model_path}")
joblib.dump(lr_model, model_path)
if os.path.exists(model_path):
    size_kb = os.path.getsize(model_path) / 1024
    print(f"   ✅ Saved ({size_kb:.2f} KB)")
else:
    print(f"   ❌ FAILED!")

# Save scaler
scaler_path = os.path.join(model_dir, 'voiceprint_scaler.pkl')
print(f"\\n2. Saving scaler to: {scaler_path}")
joblib.dump(scaler, scaler_path)
if os.path.exists(scaler_path):
    size_kb = os.path.getsize(scaler_path) / 1024
    print(f"   ✅ Saved ({size_kb:.2f} KB)")
else:
    print(f"   ❌ FAILED!")

# Save label encoder
encoder_path = os.path.join(model_dir, 'voiceprint_label_encoder.pkl')
print(f"\\n3. Saving label encoder to: {encoder_path}")
joblib.dump(label_encoder, encoder_path)
if os.path.exists(encoder_path):
    size_kb = os.path.getsize(encoder_path) / 1024
    print(f"   ✅ Saved ({size_kb:.2f} KB)")
else:
    print(f"   ❌ FAILED!")

# Save feature columns
feature_cols_path = os.path.join(model_dir, 'voiceprint_feature_cols.pkl')
print(f"\\n4. Saving feature columns to: {feature_cols_path}")
joblib.dump(feature_cols, feature_cols_path)
if os.path.exists(feature_cols_path):
    size_kb = os.path.getsize(feature_cols_path) / 1024
    print(f"   ✅ Saved {len(feature_cols)} feature columns ({size_kb:.2f} KB)")
else:
    print(f"   ❌ FAILED!")

# Verify all files
print("\\n" + "="*60)
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
    print("\\n✅ ALL MODEL FILES SAVED SUCCESSFULLY!")
    print(f"\\nLocation: {model_dir}")
else:
    print("\\n❌ SOME FILES ARE MISSING!")
"""

if __name__ == "__main__":
    print("="*60)
    print("VOICEPRINT MODEL SAVE SCRIPT")
    print("="*60)
    print("\nThis script should be run in the notebook after training.")
    print("\nAlternatively, copy this code into a new cell:\n")
    print(NOTEBOOK_CODE)
    print("\n" + "="*60)

