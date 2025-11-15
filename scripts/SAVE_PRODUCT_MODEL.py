"""
Script to save product recommendation model from notebook
Run this AFTER training in product_recommendation_model.ipynb
"""

import joblib
import os
import sys
from pathlib import Path

def save_product_model():
    """
    Save product recommendation model and all related artifacts
    This should be run in the notebook after cell 27 (save_artifacts)
    """
    
    print("="*60)
    print("SAVING PRODUCT RECOMMENDATION MODEL")
    print("="*60)
    
    try:
        # Check if required variables exist (notebook namespace)
        if 'results' not in globals() and 'results' not in locals():
            print("❌ Error: results not found!")
            print("   Please run this in the notebook AFTER training the models (cell 21)")
            return False
        
        if 'label_encoder' not in globals() and 'label_encoder' not in locals():
            print("❌ Error: label_encoder not found!")
            print("   Please run preprocessing cells first")
            return False
        
        if 'numerical_cols' not in globals() and 'numerical_cols' not in locals():
            print("❌ Error: numerical_cols not found!")
            print("   Please run prepare_data function first (cell 19)")
            return False
        
        if 'categorical_cols' not in globals() and 'categorical_cols' not in locals():
            print("❌ Error: categorical_cols not found!")
            print("   Please run prepare_data function first (cell 19)")
            return False
        
        # Get variables
        results_obj = globals().get('results') or locals().get('results')
        encoder_obj = globals().get('label_encoder') or locals().get('label_encoder')
        num_cols = globals().get('numerical_cols') or locals().get('numerical_cols')
        cat_cols = globals().get('categorical_cols') or locals().get('categorical_cols')
        
        # Get best model (highest F1 score)
        best_model_name = max(results_obj, key=lambda x: results_obj[x]['f1'])
        best_model = results_obj[best_model_name]['model']
        
        print(f"Best model: {best_model_name}")
        print(f"  F1-Score: {results_obj[best_model_name]['f1']:.4f}")
        print(f"  Accuracy: {results_obj[best_model_name]['accuracy']:.4f}")
        print()
        
        # Determine output directory
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
            if 'notebooks' in current_dir.lower() or 'models' in current_dir.lower():
                model_dir = os.path.abspath('../models')
            else:
                model_dir = os.path.abspath('models')
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        print(f"Using model directory: {model_dir}")
        print()
        
        # Save all files
        files_saved = []
        
        # 1. Save model
        model_path = os.path.join(model_dir, 'product_recommendation_model.pkl')
        print(f"1. Saving model to: {model_path}")
        try:
            joblib.dump(best_model, model_path)
            if os.path.exists(model_path):
                size_kb = os.path.getsize(model_path) / 1024
                files_saved.append(model_path)
                print(f"   ✅ Saved ({size_kb:.2f} KB)")
            else:
                print(f"   ❌ FAILED to save!")
                return False
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False
        
        # 2. Save label encoder
        encoder_path = os.path.join(model_dir, 'product_label_encoder.pkl')
        print(f"\n2. Saving label encoder to: {encoder_path}")
        try:
            joblib.dump(encoder_obj, encoder_path)
            if os.path.exists(encoder_path):
                size_kb = os.path.getsize(encoder_path) / 1024
                files_saved.append(encoder_path)
                print(f"   ✅ Saved ({size_kb:.2f} KB)")
            else:
                print(f"   ❌ FAILED to save!")
                return False
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False
        
        # 3. Save feature info
        feature_info = {
            'numerical_cols': num_cols,
            'categorical_cols': cat_cols,
            'target': 'product_category',
            'best_model_name': best_model_name,
            'f1_score': results_obj[best_model_name]['f1'],
            'accuracy': results_obj[best_model_name]['accuracy']
        }
        
        feature_info_path = os.path.join(model_dir, 'product_feature_info.pkl')
        print(f"\n3. Saving feature info to: {feature_info_path}")
        try:
            joblib.dump(feature_info, feature_info_path)
            if os.path.exists(feature_info_path):
                size_kb = os.path.getsize(feature_info_path) / 1024
                files_saved.append(feature_info_path)
                print(f"   ✅ Saved ({size_kb:.2f} KB)")
                print(f"   Numerical columns: {len(num_cols)}")
                print(f"   Categorical columns: {len(cat_cols)}")
            else:
                print(f"   ❌ FAILED to save!")
                return False
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
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
            'product_recommendation_model.pkl',
            'product_label_encoder.pkl',
            'product_feature_info.pkl'
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
            print(f"\nModel Info:")
            print(f"  Best Model: {best_model_name}")
            print(f"  F1-Score: {results_obj[best_model_name]['f1']:.4f}")
            print(f"  Accuracy: {results_obj[best_model_name]['accuracy']:.4f}")
            return True
        else:
            print("\n❌ Some files are missing!")
            return False
        
    except Exception as e:
        print(f"\n❌ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return False


# Code to copy-paste directly into notebook
NOTEBOOK_CODE = """
# ============================================================
# COPY THIS CODE INTO A NEW CELL IN product_recommendation_model.ipynb
# Run AFTER cell 27 (save_artifacts)
# ============================================================

import joblib
import os
from pathlib import Path

print("="*60)
print("SAVING PRODUCT RECOMMENDATION MODEL")
print("="*60)

# Determine correct models directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Try to find or create models directory
if 'models' in current_dir:
    model_dir = os.path.abspath('.')
elif os.path.exists('models'):
    model_dir = os.path.abspath('models')
elif os.path.exists('../models'):
    model_dir = os.path.abspath('../models')
else:
    model_dir = os.path.abspath('models')
    os.makedirs(model_dir, exist_ok=True)

print(f"Using model directory: {model_dir}")
os.makedirs(model_dir, exist_ok=True)

# Get best model (highest F1 score)
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print(f"\\nBest model: {best_model_name}")
print(f"  F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"  Accuracy: {results[best_model_name]['accuracy']:.4f}")
print()

# Save model
model_path = os.path.join(model_dir, 'product_recommendation_model.pkl')
print(f"1. Saving model to: {model_path}")
try:
    joblib.dump(best_model, model_path)
    if os.path.exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        print(f"   ✅ Saved ({size_kb:.2f} KB)")
    else:
        print(f"   ❌ FAILED - File not created!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Save label encoder
encoder_path = os.path.join(model_dir, 'product_label_encoder.pkl')
print(f"\\n2. Saving label encoder to: {encoder_path}")
try:
    joblib.dump(label_encoder, encoder_path)
    if os.path.exists(encoder_path):
        size_kb = os.path.getsize(encoder_path) / 1024
        print(f"   ✅ Saved ({size_kb:.2f} KB)")
    else:
        print(f"   ❌ FAILED - File not created!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Save feature info
feature_info = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'target': 'product_category',
    'best_model_name': best_model_name,
    'f1_score': results[best_model_name]['f1'],
    'accuracy': results[best_model_name]['accuracy']
}

feature_info_path = os.path.join(model_dir, 'product_feature_info.pkl')
print(f"\\n3. Saving feature info to: {feature_info_path}")
try:
    joblib.dump(feature_info, feature_info_path)
    if os.path.exists(feature_info_path):
        size_kb = os.path.getsize(feature_info_path) / 1024
        print(f"   ✅ Saved ({size_kb:.2f} KB)")
        print(f"   Numerical columns: {len(numerical_cols)}")
        print(f"   Categorical columns: {len(categorical_cols)}")
    else:
        print(f"   ❌ FAILED - File not created!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Verify all files
print("\\n" + "="*60)
print("VERIFICATION")
print("="*60)
required_files = [
    'product_recommendation_model.pkl',
    'product_label_encoder.pkl',
    'product_feature_info.pkl'
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
    print(f"\\nModel Info:")
    print(f"  Best Model: {best_model_name}")
    print(f"  F1-Score: {results[best_model_name]['f1']:.4f}")
    print(f"  Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print("\\nFull paths:")
    for filename in required_files:
        filepath = os.path.join(model_dir, filename)
        print(f"  • {filepath}")
else:
    print("\\n❌ SOME FILES ARE MISSING!")
    print(f"Check directory: {model_dir}")
"""

if __name__ == "__main__":
    print("="*60)
    print("PRODUCT RECOMMENDATION MODEL SAVE SCRIPT")
    print("="*60)
    print("\nThis script should be run in the notebook after training.")
    print("\nAlternatively, copy this code into a new cell:\n")
    print(NOTEBOOK_CODE)
    print("\n" + "="*60)

