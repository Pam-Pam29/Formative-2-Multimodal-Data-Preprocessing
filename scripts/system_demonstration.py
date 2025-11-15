"""
Multimodal Authentication & Product Recommendation System
Command-line application demonstrating the complete authentication flow
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.feature_extractors import extract_face_embedding, extract_audio_features_for_model


# ============================================================================
# USER IDENTITY MAPPING
# ============================================================================

# Map identities between models
IDENTITY_MAPPING = {
    # Face recognition uses: "Victoria", "Relebohile", "Denis"
    # Voice recognition uses: "Pam", "Rele", "dennis"
    "Victoria": "Pam",
    "Relebohile": "Rele", 
    "Denis": "dennis",
    # Reverse mapping
    "Pam": "Victoria",
    "Rele": "Relebohile",
    "dennis": "Denis"
}

# Standardized identity for product lookup (use voice names)
STANDARD_IDENTITIES = ["Pam", "Rele", "dennis"]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models(models_dir="Models"):
    """
    Load all trained models and related artifacts from organized subdirectories
    
    Args:
        models_dir (str): Directory containing model subdirectories
        
    Returns:
        dict: Dictionary containing all loaded models and artifacts
    """
    models = {}
    
    try:
        # ============================================================
        # FACE RECOGNITION MODEL
        # ============================================================
        face_dir = os.path.join(models_dir, "face_recognition")
        face_model_path = os.path.join(face_dir, "face_recognition_model.pkl")
        
        if os.path.exists(face_model_path):
            models['face_model'] = joblib.load(face_model_path)
            print(f"[OK] Loaded face recognition model from {face_model_path}")
        else:
            print(f"[WARNING] Warning: Face model not found at {face_model_path}")
            models['face_model'] = None
        
        # Load face embeddings data (for confidence calculation) - optional
        face_data_path = os.path.join(face_dir, "face_embeddings_data.pkl")
        if os.path.exists(face_data_path):
            face_data = joblib.load(face_data_path)
            models['face_embeddings_X'] = face_data['X']
            models['face_embeddings_y'] = face_data['y']
            print(f"[OK] Loaded face embeddings data from {face_data_path}")
        else:
            models['face_embeddings_X'] = None
            models['face_embeddings_y'] = None
            print(f"[NOTE] Note: Face embeddings data not found (optional for confidence calculation)")
        
        # ============================================================
        # VOICEPRINT MODEL
        # ============================================================
        voice_dir = os.path.join(models_dir, "voiceprint")
        voice_model_path = os.path.join(voice_dir, "voiceprint_model.pkl")
        voice_scaler_path = os.path.join(voice_dir, "voiceprint_scaler.pkl")
        voice_encoder_path = os.path.join(voice_dir, "voiceprint_label_encoder.pkl")
        voice_features_path = os.path.join(voice_dir, "voiceprint_feature_cols.pkl")
        
        if all(os.path.exists(p) for p in [voice_model_path, voice_scaler_path, voice_encoder_path]):
            models['voice_model'] = joblib.load(voice_model_path)
            models['voice_scaler'] = joblib.load(voice_scaler_path)
            models['voice_encoder'] = joblib.load(voice_encoder_path)
            print(f"[OK] Loaded voiceprint model from {voice_model_path}")
            
            if os.path.exists(voice_features_path):
                models['voice_feature_cols'] = joblib.load(voice_features_path)
                print(f"[OK] Loaded voice feature columns ({len(models['voice_feature_cols'])} features)")
            else:
                models['voice_feature_cols'] = None
                print(f"[WARNING] Warning: Voice feature columns not found")
        else:
            missing = [p for p in [voice_model_path, voice_scaler_path, voice_encoder_path] if not os.path.exists(p)]
            print(f"[WARNING] Warning: Voice model files not found:")
            for p in missing:
                print(f"     Missing: {os.path.basename(p)}")
            models['voice_model'] = None
            models['voice_scaler'] = None
            models['voice_encoder'] = None
            models['voice_feature_cols'] = None
        
        # ============================================================
        # PRODUCT RECOMMENDATION MODEL
        # ============================================================
        product_dir = os.path.join(models_dir, "product_recommendation")
        
        # Try both naming conventions
        product_model_path = os.path.join(product_dir, "product_recommendation_model.pkl")
        if not os.path.exists(product_model_path):
            # Try alternative naming
            product_model_path = os.path.join(product_dir, "best_product_recommendation_model.pkl")
        
        product_encoder_path = os.path.join(product_dir, "product_label_encoder.pkl")
        if not os.path.exists(product_encoder_path):
            # Try alternative naming
            product_encoder_path = os.path.join(product_dir, "label_encoder.pkl")
        
        product_features_path = os.path.join(product_dir, "product_feature_info.pkl")
        if not os.path.exists(product_features_path):
            # Try alternative naming
            product_features_path = os.path.join(product_dir, "feature_info.pkl")
        
        if os.path.exists(product_model_path):
            models['product_model'] = joblib.load(product_model_path)
            print(f"[OK] Loaded product recommendation model from {product_model_path}")
            
            if os.path.exists(product_encoder_path):
                models['product_encoder'] = joblib.load(product_encoder_path)
                print(f"[OK] Loaded product label encoder from {product_encoder_path}")
            else:
                models['product_encoder'] = None
                print(f"[WARNING] Warning: Product label encoder not found")
            
            if os.path.exists(product_features_path):
                models['product_feature_info'] = joblib.load(product_features_path)
                print(f"[OK] Loaded product feature info from {product_features_path}")
            else:
                models['product_feature_info'] = None
                print(f"[WARNING] Warning: Product feature info not found")
        else:
            print(f"[WARNING] Warning: Product model not found in {product_dir}")
            models['product_model'] = None
            models['product_encoder'] = None
            models['product_feature_info'] = None
        
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        raise
    
    return models


# ============================================================================
# FACE RECOGNITION
# ============================================================================

def recognize_face(image_path, face_model, face_embeddings_X, face_embeddings_y, threshold=0.5):
    """
    Recognize face from image
    
    Args:
        image_path (str): Path to face image
        face_model: Trained RandomForest face recognition model
        face_embeddings_X: Training embeddings for confidence calculation
        face_embeddings_y: Training labels for confidence calculation
        threshold (float): Minimum confidence threshold (0-1)
        
    Returns:
        tuple: (is_recognized: bool, identity: str or None, confidence: float)
    """
    if face_model is None:
        return False, None, 0.0
    
    try:
        # Extract embedding
        embedding = extract_face_embedding(image_path)
        if embedding is None:
            return False, None, 0.0
        
        # Predict identity
        pred_name = face_model.predict([embedding])[0]
        
        # Calculate confidence using cosine similarity or prediction probability
        if face_embeddings_X is not None and face_embeddings_y is not None:
            try:
                # Get embeddings for predicted class
                class_mask = face_embeddings_y == pred_name
                if class_mask.sum() > 0:
                    class_embeddings = face_embeddings_X[class_mask]
                    similarities = cosine_similarity([embedding], class_embeddings)[0]
                    confidence = float(np.max(similarities))
                else:
                    # Fallback: use prediction probability
                    proba = face_model.predict_proba([embedding])[0]
                    confidence = float(np.max(proba))
            except Exception:
                # Fallback: use prediction probability if cosine similarity fails
                proba = face_model.predict_proba([embedding])[0]
                confidence = float(np.max(proba))
        else:
            # Fallback: use prediction probability
            proba = face_model.predict_proba([embedding])[0]
            confidence = float(np.max(proba))
        
        # Check threshold
        is_recognized = confidence >= threshold
        
        return is_recognized, pred_name, confidence
        
    except Exception as e:
        print(f"[ERROR] Error in face recognition: {e}")
        return False, None, 0.0


# ============================================================================
# VOICEPRINT VERIFICATION
# ============================================================================

def verify_voice(audio_path, voice_model, voice_scaler, voice_encoder, 
                voice_feature_cols, expected_identity=None, threshold=0.7):
    """
    Verify voice sample matches expected identity
    
    Args:
        audio_path (str): Path to audio file
        voice_model: Trained voiceprint verification model
        voice_scaler: StandardScaler for audio features
        voice_encoder: LabelEncoder for speaker identities
        voice_feature_cols: List of feature column names
        expected_identity (str): Expected speaker identity (for verification)
        threshold (float): Minimum confidence threshold (0-1)
        
    Returns:
        tuple: (is_verified: bool, speaker_identity: str or None, confidence: float)
    """
    if voice_model is None or voice_scaler is None or voice_encoder is None:
        return False, None, 0.0
    
    try:
        # Extract audio features using the same method as Complete_Audio_Processing.ipynb
        # The features should match what was used in Voiceprint_Complete_Analysis.ipynb
        audio_features = extract_audio_features_for_model(audio_path, voice_feature_cols)
        
        # Scale features
        audio_features_scaled = voice_scaler.transform(audio_features)
        
        # Predict speaker
        pred_encoded = voice_model.predict(audio_features_scaled)[0]
        speaker_identity = voice_encoder.inverse_transform([pred_encoded])[0]
        
        # Get confidence
        proba = voice_model.predict_proba(audio_features_scaled)[0]
        confidence = float(np.max(proba))
        
        # Verify identity matches (if expected_identity provided)
        if expected_identity is not None:
            # Map identity if needed
            expected_mapped = IDENTITY_MAPPING.get(expected_identity, expected_identity)
            is_verified = (speaker_identity == expected_mapped) and (confidence >= threshold)
        else:
            # Just check confidence
            is_verified = confidence >= threshold
        
        return is_verified, speaker_identity, confidence
        
    except Exception as e:
        print(f"[ERROR] Error in voice verification: {e}")
        import traceback
        traceback.print_exc()
        return False, None, 0.0


# ============================================================================
# PRODUCT RECOMMENDATION
# ============================================================================

def recommend_products(user_identity, product_model, product_encoder, 
                      merged_customer_data_path="data/merged_customer_data.csv"):
    """
    Get product recommendations for user
    
    Args:
        user_identity (str): User identity (voice format: Pam/Rele/dennis)
        product_model: Trained product recommendation model
        product_encoder: LabelEncoder for product categories
        merged_customer_data_path (str): Path to merged customer dataset
        
    Returns:
        dict: Product recommendations with probabilities
    """
    if product_model is None:
        return {
            'status': 'error',
            'message': 'Product model not loaded'
        }
    
    try:
        # Load customer data
        if not os.path.exists(merged_customer_data_path):
            # Return default recommendations if data not found
            return {
                'status': 'success',
                'prediction': 'Electronics',
                'confidence': 0.3,
                'top_predictions': {
                    'Electronics': 0.3,
                    'Clothing': 0.25,
                    'Books': 0.2
                },
                'message': f'Recommendations for {user_identity} (using default)'
            }
        
        # Load merged data and find user's record
        merged_data = pd.read_csv(merged_customer_data_path)
        
        # Try to find user's data (this depends on how customer_id maps to identity)
        # For now, use the first record as a sample
        if len(merged_data) > 0:
            customer_record = merged_data.iloc[0].to_dict()
            
            # Prepare features (remove target and non-feature columns)
            exclude_cols = ['product_category_<lambda>', 'social_media_platform_<lambda>',
                          'customer_id_new', 'customer_id_legacy', 'customer_id_legacy_mapped']
            features = {k: v for k, v in customer_record.items() 
                       if k not in exclude_cols}
            
            # Convert to DataFrame
            input_df = pd.DataFrame([features])
            
            # Predict
            prediction = product_model.predict(input_df)[0]
            probabilities = product_model.predict_proba(input_df)[0]
            
            # Get top 3 predictions
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            if product_encoder is not None:
                top3_categories = product_encoder.inverse_transform(top3_indices)
                prediction_name = product_encoder.inverse_transform([prediction])[0]
            else:
                top3_categories = [f"Category_{i}" for i in top3_indices]
                prediction_name = f"Category_{prediction}"
            
            top3_probs = probabilities[top3_indices]
            
            return {
                'status': 'success',
                'prediction': prediction_name,
                'confidence': float(np.max(probabilities)),
                'top_predictions': dict(zip(top3_categories, top3_probs)),
                'user': user_identity
            }
        else:
            return {
                'status': 'error',
                'message': 'No customer data found'
            }
            
    except Exception as e:
        print(f"[ERROR] Error in product recommendation: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }


# ============================================================================
# MAIN AUTHENTICATION FLOW
# ============================================================================

def main_authentication_flow(models, test_mode=False):
    """
    Main authentication flow:
    1. Face recognition
    2. Voice verification  
    3. Product recommendation
    
    Args:
        models (dict): Dictionary of loaded models
        test_mode (bool): If True, uses predefined test files
    """
    print("\n" + "="*60)
    print("MULTIMODAL AUTHENTICATION & PRODUCT RECOMMENDATION SYSTEM")
    print("="*60)
    print("\n[LOCK] Starting authentication process...\n")
    
    # Step 1: Face Recognition
    print("-" * 60)
    print("STEP 1: FACE RECOGNITION")
    print("-" * 60)
    
    if test_mode:
        # Use a test image
        test_images = {
            "Pam": "data/images/Neutral-Pam.jpg",
            "Rele": "data/images/neutral-Rele.jpg",
            "dennis": "data/images/neutral.jpg"
        }
        image_path = list(test_images.values())[0]  # Use first available
        print(f"Using test image: {image_path}")
    else:
        image_path = input("Enter face image path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(image_path):
        print(f"[ERROR] Error: Image file not found: {image_path}")
        return
    
    print(f"Processing face image: {image_path}")
    is_recognized, face_identity, face_confidence = recognize_face(
        image_path, 
        models.get('face_model'),
        models.get('face_embeddings_X'),
        models.get('face_embeddings_y'),
        threshold=0.5
    )
    
    if not is_recognized:
        print(f"\n[ERROR] ACCESS DENIED: Face not recognized")
        print(f"   Confidence: {face_confidence*100:.2f}% (threshold: 50%)")
        return
    
    # Map face identity to voice identity
    voice_identity = IDENTITY_MAPPING.get(face_identity, face_identity)
    
    print(f"\n[SUCCESS] Face recognized!")
    print(f"   Identity: {face_identity} (voice: {voice_identity})")
    print(f"   Confidence: {face_confidence*100:.2f}%")
    
    # Step 2: Voice Verification
    print("\n" + "-" * 60)
    print("STEP 2: VOICEPRINT VERIFICATION")
    print("-" * 60)
    
    if test_mode:
        # Use a test audio file
        test_audio = {
            "Pam": "data/audio_samples/Pam_Yes approve .wav",
            "Rele": "data/audio_samples/Rele_Recording_1.m4a",
            "dennis": "data/audio_samples/dennis_approve.wav"
        }
        audio_path = test_audio.get(voice_identity, list(test_audio.values())[0])
        print(f"Using test audio: {audio_path}")
    else:
        print(f"Expected speaker: {voice_identity}")
        audio_path = input("Enter audio file path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(audio_path):
        print(f"[ERROR] Error: Audio file not found: {audio_path}")
        return
    
    print(f"Processing voice sample: {audio_path}")
    is_verified, speaker_identity, voice_confidence = verify_voice(
        audio_path,
        models.get('voice_model'),
        models.get('voice_scaler'),
        models.get('voice_encoder'),
        models.get('voice_feature_cols'),
        expected_identity=voice_identity,
        threshold=0.7
    )
    
    if not is_verified:
        print(f"\n[ERROR] ACCESS DENIED: Voice verification failed")
        print(f"   Expected: {voice_identity}")
        print(f"   Detected: {speaker_identity}")
        print(f"   Confidence: {voice_confidence*100:.2f}% (threshold: 70%)")
        return
    
    print(f"\n[SUCCESS] Voice verified!")
    print(f"   Speaker: {speaker_identity}")
    print(f"   Confidence: {voice_confidence*100:.2f}%")
    
    # Step 3: Product Recommendation
    print("\n" + "-" * 60)
    print("STEP 3: PRODUCT RECOMMENDATION")
    print("-" * 60)
    
    print(f"Fetching product recommendations for {speaker_identity}...")
    
    recommendations = recommend_products(
        speaker_identity,
        models.get('product_model'),
        models.get('product_encoder'),
        merged_customer_data_path="data/merged_customer_data.csv"
    )
    
    if recommendations['status'] == 'success':
        print(f"\n[SUCCESS] Product Recommendations:")
        print(f"   Recommended Category: {recommendations['prediction']}")
        print(f"   Confidence: {recommendations['confidence']*100:.2f}%")
        print(f"\n   Top 3 Categories:")
        for category, prob in recommendations['top_predictions'].items():
            print(f"     • {category}: {prob*100:.2f}%")
    else:
        print(f"\n[WARNING] Warning: {recommendations.get('message', 'Could not generate recommendations')}")
    
    print("\n" + "="*60)
    print("[SUCCESS] AUTHENTICATION COMPLETE - ACCESS GRANTED")
    print("="*60)


# ============================================================================
# UNAUTHORIZED ATTEMPT SIMULATIONS
# ============================================================================

def simulate_unauthorized_face(models):
    """Simulate unauthorized face attempt"""
    print("\n" + "="*60)
    print("UNAUTHORIZED FACE ATTEMPT SIMULATION")
    print("="*60)
    
    print("\nUsing unauthorized/unknown face image...")
    
    # You can use any image that doesn't match registered users
    # For demo, we'll try with an existing image but expect low confidence
    unauthorized_image = input("Enter path to unauthorized face image (or press Enter for default): ").strip()
    
    if not unauthorized_image:
        # Use a different registered user's image as "unauthorized" for demo
        unauthorized_image = "data/images/Neutral-Pam.jpg"  # Replace with actual unauthorized image
    
    if not os.path.exists(unauthorized_image):
        print(f"[ERROR] Error: Image file not found: {unauthorized_image}")
        return
    
    is_recognized, identity, confidence = recognize_face(
        unauthorized_image,
        models.get('face_model'),
        models.get('face_embeddings_X'),
        models.get('face_embeddings_y'),
        threshold=0.5
    )
    
    print(f"\nResults:")
    print(f"  Detected Identity: {identity}")
    print(f"  Confidence: {confidence*100:.2f}%")
    print(f"  Threshold: 50%")
    
    if not is_recognized or confidence < 0.3:
        print(f"\n[SUCCESS] ACCESS DENIED: Face not recognized")
        print(f"   Reason: Confidence below threshold")
    else:
        print(f"\n[WARNING] Note: This demonstrates the security check")
        print(f"   (In real scenario, would check against unauthorized database)")


def simulate_unauthorized_voice(models):
    """Simulate unauthorized voice attempt"""
    print("\n" + "="*60)
    print("UNAUTHORIZED VOICE ATTEMPT SIMULATION")
    print("="*60)
    
    print("\nScenario: Face recognized, but wrong voice sample provided")
    
    # Step 1: Valid face
    print("\nStep 1: Face recognized (simulated)")
    face_identity = "Victoria"  # Simulated
    voice_identity = IDENTITY_MAPPING.get(face_identity)
    print(f"   Face Identity: {face_identity}")
    print(f"   Expected Voice: {voice_identity}")
    
    # Step 2: Wrong voice
    print("\nStep 2: Voice sample provided")
    wrong_audio = input("Enter path to wrong voice sample (or press Enter for demo): ").strip()
    
    if not wrong_audio:
        # Use different user's audio as "wrong" for demo
        wrong_audio = "data/audio_samples/Rele_Recording_1.m4a"  # Wrong user
    
    if not os.path.exists(wrong_audio):
        print(f"[ERROR] Error: Audio file not found: {wrong_audio}")
        return
    
    is_verified, speaker_identity, confidence = verify_voice(
        wrong_audio,
        models.get('voice_model'),
        models.get('voice_scaler'),
        models.get('voice_encoder'),
        models.get('voice_feature_cols'),
        expected_identity=voice_identity,  # Expecting Pam but got Rele
        threshold=0.7
    )
    
    print(f"\nResults:")
    print(f"  Expected Speaker: {voice_identity}")
    print(f"  Detected Speaker: {speaker_identity}")
    print(f"  Confidence: {confidence*100:.2f}%")
    
    if not is_verified:
        print(f"\n[ERROR] ACCESS DENIED: Voice verification failed")
        print(f"   Reason: Speaker identity mismatch or low confidence")
    else:
        print(f"\n[WARNING] Warning: Unexpected verification (check thresholds)")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI interface"""
    print("\n" + "="*60)
    print("MULTIMODAL AUTHENTICATION & PRODUCT RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
    try:
        # Try Models directory first (new structure), then models (old structure)
        if os.path.exists("Models"):
            models = load_models(models_dir="Models")
        elif os.path.exists("models"):
            models = load_models(models_dir="models")
        else:
            print("[ERROR] Models directory not found!")
            print("   Expected: 'Models/' or 'models/' directory")
            print("   Current directory:", os.getcwd())
            return
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure models are saved in the 'Models' or 'models' directory.")
        return
    
    # Check if models loaded successfully
    if not any([models.get('face_model'), models.get('voice_model'), models.get('product_model')]):
        print("\n[ERROR] No models loaded. Please train and save models first.")
        return
    
    while True:
        print("\n" + "-"*60)
        print("Select an option:")
        print("  1. Full Authentication Flow (Authorized User)")
        print("  2. Simulate Unauthorized Face Attempt")
        print("  3. Simulate Unauthorized Voice Attempt")
        print("  4. Test Mode (Uses default test files)")
        print("  5. Exit")
        print("-"*60)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            main_authentication_flow(models, test_mode=False)
        elif choice == "2":
            simulate_unauthorized_face(models)
        elif choice == "3":
            simulate_unauthorized_voice(models)
        elif choice == "4":
            print("\n[TEST] Test Mode: Using default test files")
            main_authentication_flow(models, test_mode=True)
        elif choice == "5":
            print("\n[BYE] Exiting...")
            break
        else:
            print("\n[ERROR] Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()

