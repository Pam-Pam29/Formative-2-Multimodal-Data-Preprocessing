"""
Quick script to download product recommendation model files from Google Colab
Run this in a new Colab cell after saving the models
"""

from google.colab import files
import os

# Model files to download
model_files = {
    'product_recommendation_model.pkl': '/models/product_recommendation_model.pkl',
    'product_label_encoder.pkl': '/models/product_label_encoder.pkl',
    'product_feature_info.pkl': '/models/product_feature_info.pkl'
}

print("="*60)
print("DOWNLOADING PRODUCT RECOMMENDATION MODEL FILES")
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

print("\n" + "="*60)
print("ALTERNATIVE: Download as ZIP")
print("="*60)
print("""
If you prefer to download all files at once as a ZIP, run:

import zipfile
from google.colab import files

zip_path = '/content/product_models.zip'
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for filepath in [
        '/models/product_recommendation_model.pkl',
        '/models/product_label_encoder.pkl',
        '/models/product_feature_info.pkl'
    ]:
        if os.path.exists(filepath):
            zipf.write(filepath, os.path.basename(filepath))
            print(f"Added: {os.path.basename(filepath)}")

files.download(zip_path)
print("✅ Downloaded product_models.zip")
""")

