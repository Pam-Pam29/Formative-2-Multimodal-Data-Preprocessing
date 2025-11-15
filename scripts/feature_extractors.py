"""
Feature Extraction Functions for Multimodal Authentication System
Extracts features from face images and audio files
"""

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import librosa
import soundfile as sf
import os


# ============================================================================
# FACE FEATURE EXTRACTION
# ============================================================================

def extract_face_embedding(image_path):
    """
    Extract face embedding using DeepFace Facenet model
    
    Args:
        image_path (str): Path to face image
        
    Returns:
        np.array: Face embedding vector (128 dimensions) or None if error
    """
    try:
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None


# ============================================================================
# AUDIO FEATURE EXTRACTION
# ============================================================================

def load_audio(file_path, target_sr=22050):
    """Load and resample audio file"""
    y, sr = librosa.load(file_path, sr=target_sr)
    return y, sr


def extract_mfcc_features(audio, sr, n_mfcc=13):
    """Extract MFCC features"""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return {
        'mfcc_mean': np.mean(mfccs, axis=1),
        'mfcc_std': np.std(mfccs, axis=1),
        'mfcc_max': np.max(mfccs, axis=1),
        'mfcc_min': np.min(mfccs, axis=1)
    }


def extract_spectral_features(audio, sr):
    """Extract spectral features including roll-off"""
    features = {}
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(rolloff)
    features['spectral_rolloff_std'] = np.std(rolloff)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features['spectral_centroid_mean'] = np.mean(centroid)
    features['spectral_centroid_std'] = np.std(centroid)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(bandwidth)
    features['spectral_bandwidth_std'] = np.std(bandwidth)
    return features


def extract_energy_features(audio):
    """Extract energy features"""
    features = {}
    rms = librosa.feature.rms(y=audio)
    features['rms_energy_mean'] = np.mean(rms)
    features['rms_energy_std'] = np.std(rms)
    features['rms_energy_max'] = np.max(rms)
    zcr = librosa.feature.zero_crossing_rate(audio)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    return features


def extract_chroma_features(audio, sr):
    """Extract chroma features"""
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    return {
        'chroma_mean': np.mean(chroma, axis=1),
        'chroma_std': np.std(chroma, axis=1)
    }


def extract_all_audio_features(audio_path):
    """
    Extract ALL audio features by combining all individual functions
    
    Args:
        audio_path (str): Path to audio file
        
    Returns:
        dict: Dictionary containing all audio features
    """
    audio, sr = librosa.load(audio_path, sr=None)
    features = {}

    # MFCC features
    mfcc_data = extract_mfcc_features(audio, sr)
    for i in range(len(mfcc_data['mfcc_mean'])):
        features[f'mfcc_mean_{i}'] = mfcc_data['mfcc_mean'][i]
        features[f'mfcc_std_{i}'] = mfcc_data['mfcc_std'][i]
        features[f'mfcc_max_{i}'] = mfcc_data['mfcc_max'][i]
        features[f'mfcc_min_{i}'] = mfcc_data['mfcc_min'][i]

    # Spectral features
    spectral_features = extract_spectral_features(audio, sr)
    features.update(spectral_features)

    # Energy features
    energy_features = extract_energy_features(audio)
    features.update(energy_features)

    # Chroma features
    chroma_data = extract_chroma_features(audio, sr)
    for i in range(len(chroma_data['chroma_mean'])):
        features[f'chroma_mean_{i}'] = chroma_data['chroma_mean'][i]
        features[f'chroma_std_{i}'] = chroma_data['chroma_std'][i]

    # Additional features
    features['duration'] = len(audio) / sr
    features['sample_rate'] = sr
    features['max_amplitude'] = np.max(np.abs(audio))
    features['mean_amplitude'] = np.mean(np.abs(audio))

    return features


def extract_audio_features_for_model(audio_path, feature_cols=None):
    """
    Extract audio features formatted for voiceprint model
    Uses the same feature extraction as Complete_Audio_Processing.ipynb
    
    Args:
        audio_path (str): Path to audio file
        feature_cols (list): List of feature column names expected by model
                            If None, will infer from audio_features.csv structure
                            (all numeric columns except metadata)
        
    Returns:
        np.array: Features array matching feature_cols (shape: [1, n_features])
    """
    # Extract all features using the same method as Complete_Audio_Processing.ipynb
    features_dict = extract_all_audio_features(audio_path)
    
    # Create DataFrame with single row
    features_df = pd.DataFrame([features_dict])
    
    # If feature_cols not provided, infer from audio_features.csv structure
    if feature_cols is None:
        # These are the metadata columns that should be excluded
        exclude_cols = ['file_id', 'speaker', 'audio_name', 'augmentation', 'audio_path']
        # Get all numeric columns (these are the feature columns used in training)
        feature_cols = [col for col in features_df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
    
    # Select only the columns that the model expects
    available_cols = [col for col in feature_cols if col in features_df.columns]
    
    if len(available_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        print(f"⚠ Warning: Missing features: {len(missing)} features")
        if len(missing) <= 10:
            print(f"   Missing: {list(missing)}")
        else:
            print(f"   Missing: {list(missing)[:5]}... and {len(missing)-5} more")
        # Fill missing features with 0
        for col in missing:
            features_df[col] = 0
    
    # Reorder to match feature_cols order (important for model prediction)
    try:
        features_array = features_df[feature_cols].values.reshape(1, -1)
    except KeyError as e:
        print(f"❌ Error: Feature column mismatch: {e}")
        print(f"   Expected {len(feature_cols)} features, got {len(features_df.columns)}")
        print(f"   Sample expected features: {feature_cols[:5]}")
        print(f"   Sample available features: {list(features_df.columns)[:5]}")
        raise
    
    return features_array

