"""
Complete script to save and download ALL models from Google Colab
This combines saving and downloading for all 3 models
"""

from google.colab import files
import joblib
import os
import zipfile

def save_and_download_all_models():
    """
    Save all models (face, voice, product) and prepare for download
    """
    
    print("="*70)
    print("SAVING AND PREPARING ALL MODELS FOR DOWNLOAD")
    print("="*70)
    
    # Determine models directory
    current_dir = os.getcwd()
    if 'models' in current_dir:
        model_dir = os.path.abspath('.')
    elif os.path.exists('models'):
        model_dir = os.path.abspath('models')
    elif os.path.exists('../models'):
        model_dir = os.path.abspath('../models')
    else:
        model_dir = os.path.abspath('models')
        os.makedirs(model_dir, exist_ok=True)
    
    print(f"Using model directory: {model_dir}\n")
    
    all_files = []
    
    # ============================================================
    # 1. FACE RECOGNITION MODEL
    # ============================================================
    print("="*70)
    print("1. FACE RECOGNITION MODEL")
    print("="*70)
    
    if 'rf_model' in globals() and 'X' in globals() and 'y' in globals():
        try:
            # Save model
            face_model_path = os.path.join(model_dir, 'face_recognition_model.pkl')
            joblib.dump(rf_model, face_model_path)
            all_files.append(face_model_path)
            print(f"✅ Saved face_recognition_model.pkl")
            
            # Save embeddings data
            face_data = {
                'X': X.values if isinstance(X, pd.DataFrame) else X,
                'y': y.values if isinstance(y, pd.Series) else y
            }
            face_data_path = os.path.join(model_dir, 'face_embeddings_data.pkl')
            joblib.dump(face_data, face_data_path)
            all_files.append(face_data_path)
            print(f"✅ Saved face_embeddings_data.pkl")
        except Exception as e:
            print(f"❌ Error saving face model: {e}")
    else:
        print("⚠ Skipping: Face model variables not found")
    
    # ============================================================
    # 2. VOICEPRINT MODEL
    # ============================================================
    print("\n" + "="*70)
    print("2. VOICEPRINT VERIFICATION MODEL")
    print("="*70)
    
    if 'lr_model' in globals() and 'scaler' in globals() and 'label_encoder' in globals() and 'df' in globals():
        try:
            # Save model
            voice_model_path = os.path.join(model_dir, 'voiceprint_model.pkl')
            joblib.dump(lr_model, voice_model_path)
            all_files.append(voice_model_path)
            print(f"✅ Saved voiceprint_model.pkl")
            
            # Save scaler
            voice_scaler_path = os.path.join(model_dir, 'voiceprint_scaler.pkl')
            joblib.dump(scaler, voice_scaler_path)
            all_files.append(voice_scaler_path)
            print(f"✅ Saved voiceprint_scaler.pkl")
            
            # Save label encoder
            voice_encoder_path = os.path.join(model_dir, 'voiceprint_label_encoder.pkl')
            joblib.dump(label_encoder, voice_encoder_path)
            all_files.append(voice_encoder_path)
            print(f"✅ Saved voiceprint_label_encoder.pkl")
            
            # Save feature columns
            if 'feature_cols' not in locals():
                exclude_cols = ['file_id', 'speaker', 'audio_name', 'augmentation', 'audio_path']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
            voice_feature_cols_path = os.path.join(model_dir, 'voiceprint_feature_cols.pkl')
            joblib.dump(feature_cols, voice_feature_cols_path)
            all_files.append(voice_feature_cols_path)
            print(f"✅ Saved voiceprint_feature_cols.pkl")
        except Exception as e:
            print(f"❌ Error saving voice model: {e}")
    else:
        print("⚠ Skipping: Voice model variables not found")
    
    # ============================================================
    # 3. PRODUCT RECOMMENDATION MODEL
    # ============================================================
    print("\n" + "="*70)
    print("3. PRODUCT RECOMMENDATION MODEL")
    print("="*70)
    
    if 'results' in globals() and 'label_encoder' in globals() and 'numerical_cols' in globals() and 'categorical_cols' in globals():
        try:
            # Get best model
            best_model_name = max(results, key=lambda x: results[x]['f1'])
            best_model = results[best_model_name]['model']
            
            # Save model
            product_model_path = os.path.join(model_dir, 'product_recommendation_model.pkl')
            joblib.dump(best_model, product_model_path)
            all_files.append(product_model_path)
            print(f"✅ Saved product_recommendation_model.pkl")
            print(f"   Best model: {best_model_name}")
            
            # Save label encoder (different from voice one, so rename if needed)
            product_encoder_path = os.path.join(model_dir, 'product_label_encoder.pkl')
            joblib.dump(label_encoder, product_encoder_path)
            all_files.append(product_encoder_path)
            print(f"✅ Saved product_label_encoder.pkl")
            
            # Save feature info
            feature_info = {
                'numerical_cols': numerical_cols,
                'categorical_cols': categorical_cols,
                'target': 'product_category',
                'best_model_name': best_model_name,
                'f1_score': results[best_model_name]['f1'],
                'accuracy': results[best_model_name]['accuracy']
            }
            product_feature_info_path = os.path.join(model_dir, 'product_feature_info.pkl')
            joblib.dump(feature_info, product_feature_info_path)
            all_files.append(product_feature_info_path)
            print(f"✅ Saved product_feature_info.pkl")
        except Exception as e:
            print(f"❌ Error saving product model: {e}")
    else:
        print("⚠ Skipping: Product model variables not found")
    
    # ============================================================
    # VERIFY AND DOWNLOAD
    # ============================================================
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    existing_files = []
    for filepath in all_files:
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"✅ {os.path.basename(filepath)} ({size_kb:.2f} KB)")
            existing_files.append(filepath)
        else:
            print(f"❌ {os.path.basename(filepath)} - MISSING!")
    
    print("\n" + "="*70)
    print("DOWNLOAD OPTIONS")
    print("="*70)
    
    if existing_files:
        print(f"\nFound {len(existing_files)} model files to download.\n")
        print("Option 1: Download individual files")
        print("-" * 70)
        print("Run this code to download each file:")
        print()
        for filepath in existing_files:
            print(f"files.download('{filepath}')")
        
        print("\nOption 2: Download as ZIP (Recommended)")
        print("-" * 70)
        print("Run this code to create and download a ZIP file:")
        print()
        zip_code = f"""
zip_path = '/content/all_models.zip'
with zipfile.ZipFile(zip_path, 'w') as zipf:
"""
        for filepath in existing_files:
            filename = os.path.basename(filepath)
            zip_code += f"    zipf.write('{filepath}', '{filename}')\n"
        zip_code += "    print('Created ZIP file')\n\nfiles.download(zip_path)\nprint('✅ Downloaded all_models.zip')"
        
        print(zip_code)
        
        # Auto-create ZIP
        try:
            zip_path = '/content/all_models.zip'
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for filepath in existing_files:
                    filename = os.path.basename(filepath)
                    zipf.write(filepath, filename)
                    print(f"Added to ZIP: {filename}")
            
            print(f"\n✅ Created ZIP file: {zip_path}")
            print("Ready to download!")
            print("\nTo download, run:")
            print("files.download('/content/all_models.zip')")
        except Exception as e:
            print(f"⚠ Could not create ZIP: {e}")
    else:
        print("\n❌ No files found to download!")
        print("Make sure you've saved the models first.")


if __name__ == "__main__":
    save_and_download_all_models()

