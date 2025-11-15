"""
Quick script to download voiceprint model files from Google Colab
Run this in a new Colab cell after saving the models
"""

from google.colab import files
import os

# Model files to download
model_files = {
    'voiceprint_model.pkl': '/models/voiceprint_model.pkl',
    'voiceprint_scaler.pkl': '/models/voiceprint_scaler.pkl',
    'voiceprint_label_encoder.pkl': '/models/voiceprint_label_encoder.pkl',
    'voiceprint_feature_cols.pkl': '/models/voiceprint_feature_cols.pkl'
}

print("="*60)
print("DOWNLOADING VOICEPRINT MODEL FILES")
print("="*60)
print("\nFiles to download:\n")

for local_name, colab_path in model_files.items():
    print(f"  • {local_name}")
    print(f"    From: {colab_path}")

print("\n" + "="*60)
print("Starting downloads...")
print("="*60)

downloaded = []
missing = []

for local_name, colab_path in model_files.items():
    if os.path.exists(colab_path):
        try:
            files.download(colab_path)
            downloaded.append(local_name)
            print(f"✅ Downloaded: {local_name}")
        except Exception as e:
            print(f"❌ Error downloading {local_name}: {e}")
            missing.append(local_name)
    else:
        print(f"❌ Not found: {colab_path}")
        missing.append(local_name)

print("\n" + "="*60)
print("DOWNLOAD SUMMARY")
print("="*60)

if downloaded:
    print(f"\n✅ Successfully downloaded {len(downloaded)} files:")
    for name in downloaded:
        print(f"   • {name}")

if missing:
    print(f"\n❌ {len(missing)} files were missing or failed:")
    for name in missing:
        print(f"   • {name}")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("""
1. The files have been downloaded to your computer's Downloads folder
2. Create a 'models/' folder in your project root if it doesn't exist
3. Move the downloaded .pkl files to: Formative-2-Multimodal-Data-Preprocessing-1/models/
4. Verify with: python scripts/VERIFY_MODELS.py
5. Run system: python scripts/system_demonstration.py
""")

