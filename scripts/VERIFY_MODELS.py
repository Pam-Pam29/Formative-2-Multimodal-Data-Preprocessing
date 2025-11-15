"""
Script to verify all models are saved correctly
Run this from the project root directory
"""

import os
from pathlib import Path

def verify_models(models_dir="Models"):
    """
    Verify all model files exist in organized subdirectories
    
    Args:
        models_dir (str): Path to models directory (should contain subdirectories)
    """
    print("="*60)
    print("MODEL VERIFICATION")
    print("="*60)
    
    # Try both naming conventions
    if not os.path.exists(models_dir):
        if os.path.exists("models"):
            models_dir = "models"
        elif os.path.exists("Models"):
            models_dir = "Models"
        else:
            print(f"\n[ERROR] Models directory not found!")
            print(f"   Tried: 'Models/', 'models/'")
            print(f"   Current directory: {os.getcwd()}")
            print("\nPlease save models first using the instructions in scripts/README_SAVE_MODELS.md")
            return False
    
    # Get absolute path
    models_path = os.path.abspath(models_dir)
    
    print(f"\nChecking directory: {models_path}")
    print()
    
    # Required files organized by subdirectory
    required_files = {
        'Face Recognition': {
            'dir': os.path.join(models_path, 'face_recognition'),
            'files': [
                'face_recognition_model.pkl',
                'face_embeddings_data.pkl'  # Optional
            ],
            'optional': ['face_embeddings_data.pkl']
        },
        'Voiceprint Verification': {
            'dir': os.path.join(models_path, 'voiceprint'),
            'files': [
                'voiceprint_model.pkl',
                'voiceprint_scaler.pkl',
                'voiceprint_label_encoder.pkl',
                'voiceprint_feature_cols.pkl'
            ],
            'optional': []
        },
        'Product Recommendation': {
            'dir': os.path.join(models_path, 'product_recommendation'),
            'files': [
                'product_recommendation_model.pkl',  # Try alternative names
                'best_product_recommendation_model.pkl',
                'product_label_encoder.pkl',
                'label_encoder.pkl',
                'product_feature_info.pkl',
                'feature_info.pkl'
            ],
            'optional': []
        }
    }
    
    all_exist = True
    results = {}
    
    for model_type, config in required_files.items():
        print(f"[{model_type}]:")
        print(f"   Directory: {config['dir']}")
        results[model_type] = {}
        
        model_dir = config['dir']
        if not os.path.exists(model_dir):
            print(f"   [ERROR] Directory not found!")
            for filename in config['files']:
                results[model_type][filename] = {'exists': False, 'size_kb': 0}
            all_exist = False
            print()
            continue
        
        # Check for model files with alternative naming
        found_files = {}  # Track which files we've found to avoid duplicates
        
        for filename in config['files']:
            filepath = os.path.join(model_dir, filename)
            
            # Handle alternative naming for product model
            if 'product_recommendation_model' in filename:
                if os.path.exists(filepath):
                    found_files['model'] = (filepath, filename)
                else:
                    alt_path = os.path.join(model_dir, 'best_product_recommendation_model.pkl')
                    if os.path.exists(alt_path) and 'model' not in found_files:
                        found_files['model'] = (alt_path, 'best_product_recommendation_model.pkl')
                continue
            
            # Handle alternative naming for label encoder
            elif 'label_encoder' in filename:
                if os.path.exists(filepath):
                    found_files['encoder'] = (filepath, filename)
                else:
                    alt_path = os.path.join(model_dir, 'label_encoder.pkl')
                    if os.path.exists(alt_path) and 'encoder' not in found_files and 'product' in model_type.lower():
                        found_files['encoder'] = (alt_path, 'label_encoder.pkl')
                continue
            
            # Handle alternative naming for feature info
            elif 'feature_info' in filename:
                if os.path.exists(filepath):
                    found_files['features'] = (filepath, filename)
                else:
                    alt_path = os.path.join(model_dir, 'feature_info.pkl')
                    if os.path.exists(alt_path) and 'features' not in found_files and 'product' in model_type.lower():
                        found_files['features'] = (alt_path, 'feature_info.pkl')
                continue
            
            # For other files, just check normally
            if os.path.exists(filepath):
                found_files[filename] = (filepath, filename)
        
        # Print all found files
        for key, (filepath, filename) in found_files.items():
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                is_optional = filename in config.get('optional', [])
                marker = "[OK]" if not is_optional else "[OPT]"
                print(f"   {marker} {filename} ({size_kb:.2f} KB)")
                results[model_type][filename] = {'exists': True, 'size_kb': size_kb}
        
        # Check for missing required files
        required_base_files = {
            'face_recognition': ['face_recognition_model.pkl'],
            'voiceprint': ['voiceprint_model.pkl', 'voiceprint_scaler.pkl', 'voiceprint_label_encoder.pkl'],
            'product_recommendation': ['best_product_recommendation_model.pkl', 'label_encoder.pkl', 'feature_info.pkl']
        }
        
        model_key = model_type.lower().replace(' ', '_')
        if model_key in required_base_files:
            for req_file in required_base_files[model_key]:
                if req_file not in [f[1] for f in found_files.values()]:
                    # Check if any similar file exists
                    alt_names = {
                        'best_product_recommendation_model.pkl': ['product_recommendation_model.pkl'],
                        'label_encoder.pkl': ['product_label_encoder.pkl'],
                        'feature_info.pkl': ['product_feature_info.pkl']
                    }
                    found = False
                    if req_file in alt_names:
                        for alt in alt_names[req_file]:
                            if alt in [f[1] for f in found_files.values()]:
                                found = True
                                break
                    if not found and req_file not in config.get('optional', []):
                        print(f"   [MISS] {req_file} - MISSING!")
                        results[model_type][req_file] = {'exists': False, 'size_kb': 0}
                        all_exist = False
        print()
    
    print("="*60)
    if all_exist:
        print("[SUCCESS] ALL MODELS VERIFIED!")
        print(f"\nAll model files exist in: {models_path}")
        return True
    else:
        print("[WARNING] SOME MODELS ARE MISSING!")
        print(f"\nMissing files:")
        for model_type, files in results.items():
            for filename, status in files.items():
                if not status['exists']:
                    print(f"  - {model_type}: {filename}")
        print(f"\nPlease save missing models using instructions in scripts/README_SAVE_MODELS.md")
        return False


if __name__ == "__main__":
    verify_models()

