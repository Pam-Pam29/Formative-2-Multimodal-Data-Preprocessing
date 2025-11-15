"""
Script to verify all models are saved correctly
Run this from the project root directory
"""

import os
from pathlib import Path

def verify_models(models_dir="models"):
    """
    Verify all model files exist
    
    Args:
        models_dir (str): Path to models directory
    """
    print("="*60)
    print("MODEL VERIFICATION")
    print("="*60)
    
    # Get absolute path
    models_path = os.path.abspath(models_dir)
    
    if not os.path.exists(models_path):
        print(f"\n❌ Models directory not found: {models_path}")
        print("\nPlease save models first using the instructions in scripts/README_SAVE_MODELS.md")
        return False
    
    print(f"\nChecking directory: {models_path}")
    print()
    
    # Required files
    required_files = {
        'Face Recognition': [
            'face_recognition_model.pkl',
            'face_embeddings_data.pkl'
        ],
        'Voiceprint Verification': [
            'voiceprint_model.pkl',
            'voiceprint_scaler.pkl',
            'voiceprint_label_encoder.pkl',
            'voiceprint_feature_cols.pkl'
        ],
        'Product Recommendation': [
            'product_recommendation_model.pkl',
            'product_label_encoder.pkl',
            'product_feature_info.pkl'
        ]
    }
    
    all_exist = True
    results = {}
    
    for model_type, files in required_files.items():
        print(f"📦 {model_type}:")
        results[model_type] = {}
        
        for filename in files:
            filepath = os.path.join(models_path, filename)
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                print(f"   ✅ {filename} ({size_kb:.2f} KB)")
                results[model_type][filename] = {'exists': True, 'size_kb': size_kb}
            else:
                print(f"   ❌ {filename} - MISSING!")
                results[model_type][filename] = {'exists': False, 'size_kb': 0}
                all_exist = False
        print()
    
    print("="*60)
    if all_exist:
        print("✅ ALL MODELS VERIFIED!")
        print(f"\nAll model files exist in: {models_path}")
        return True
    else:
        print("❌ SOME MODELS ARE MISSING!")
        print(f"\nMissing files:")
        for model_type, files in results.items():
            for filename, status in files.items():
                if not status['exists']:
                    print(f"  • {model_type}: {filename}")
        print(f"\nPlease save missing models using instructions in scripts/README_SAVE_MODELS.md")
        return False


if __name__ == "__main__":
    verify_models()

