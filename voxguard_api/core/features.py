"""
VoxGuard Audio Feature Extraction Module
Handles audio decoding and feature extraction for deepfake detection
"""

import base64
import io
import numpy as np
from typing import Tuple, Dict, Any
import librosa
import warnings

from voxguard_api.core.config import (
    settings,
    MFCC_N_COEFFS,
    N_MELS,
    HOP_LENGTH,
    N_FFT
)

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)


def decode_base64_mp3_to_array(audio_base64: str) -> Tuple[np.ndarray, int]:
    """
    Decode Base64 encoded MP3 audio to numpy array.
    
    Args:
        audio_base64: Base64 encoded MP3 audio string
        
    Returns:
        Tuple of (waveform numpy array, sample rate)
        
    Raises:
        ValueError: If decoding fails or audio is invalid
    """
    try:
        # Decode Base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        raise ValueError(f"Failed to decode Base64 audio: {str(e)}")
    
    try:
        # Create a file-like object from bytes
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load audio using librosa
        # mono=True to get single channel, sr=None to preserve original sample rate
        y, sr = librosa.load(audio_buffer, sr=settings.sample_rate, mono=True)
        
        # Validate audio
        if len(y) == 0:
            raise ValueError("Audio file is empty or contains no audio data")
        
        duration = len(y) / sr
        if duration < settings.min_audio_duration:
            raise ValueError(
                f"Audio too short ({duration:.2f}s). "
                f"Minimum duration is {settings.min_audio_duration}s"
            )
        
        # Truncate if too long (only for analysis, not modifying stored audio)
        if duration > settings.max_audio_duration:
            max_samples = int(settings.max_audio_duration * sr)
            y = y[:max_samples]
        
        # Check for silence
        rms = np.sqrt(np.mean(y**2))
        if rms < 1e-6:
            raise ValueError("Audio appears to be silent or contains only noise")
        
        return y, sr
        
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {str(e)}")


def extract_audio_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Extract audio features for deepfake detection.
    
    Computes:
    - MFCCs (Mel-frequency cepstral coefficients)
    - Log-mel spectrogram
    - Prosodic features (pitch, energy, speaking rate)
    
    Args:
        y: Audio waveform as numpy array
        sr: Sample rate
        
    Returns:
        Dictionary containing extracted features
    """
    features = {}
    
    # 1. MFCC Features
    mfccs = librosa.feature.mfcc(
        y=y, 
        sr=sr, 
        n_mfcc=MFCC_N_COEFFS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    
    # MFCC statistics
    features["mfcc_mean"] = np.mean(mfccs, axis=1)
    features["mfcc_std"] = np.std(mfccs, axis=1)
    features["mfcc_min"] = np.min(mfccs, axis=1)
    features["mfcc_max"] = np.max(mfccs, axis=1)
    features["mfcc_delta"] = np.mean(np.abs(np.diff(mfccs, axis=1)), axis=1)
    
    # 2. Mel Spectrogram Features
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Mel spectrogram statistics (reduced)
    features["mel_mean"] = np.mean(log_mel_spec, axis=1)[::4]  # Subsample
    features["mel_std"] = np.std(log_mel_spec, axis=1)[::4]
    
    # 3. Spectral Features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    
    features["spectral_centroid_mean"] = np.mean(spectral_centroid)
    features["spectral_centroid_std"] = np.std(spectral_centroid)
    features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
    features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)
    features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
    features["spectral_rolloff_std"] = np.std(spectral_rolloff)
    features["spectral_flatness_mean"] = np.mean(spectral_flatness)
    features["spectral_flatness_std"] = np.std(spectral_flatness)
    features["zcr_mean"] = np.mean(zero_crossing_rate)
    features["zcr_std"] = np.std(zero_crossing_rate)
    
    # 4. Pitch/F0 Features (using pyin for better robustness)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=HOP_LENGTH
        )
        
        # Filter out unvoiced frames
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            features["pitch_mean"] = np.mean(f0_voiced)
            features["pitch_std"] = np.std(f0_voiced)
            features["pitch_min"] = np.min(f0_voiced)
            features["pitch_max"] = np.max(f0_voiced)
            features["pitch_range"] = features["pitch_max"] - features["pitch_min"]
            features["voiced_ratio"] = len(f0_voiced) / len(f0)
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0
            features["pitch_min"] = 0.0
            features["pitch_max"] = 0.0
            features["pitch_range"] = 0.0
            features["voiced_ratio"] = 0.0
    except Exception:
        # Fallback if pitch detection fails
        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_min"] = 0.0
        features["pitch_max"] = 0.0
        features["pitch_range"] = 0.0
        features["voiced_ratio"] = 0.0
    
    # 5. Energy Features
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    features["energy_mean"] = np.mean(rms)
    features["energy_std"] = np.std(rms)
    features["energy_min"] = np.min(rms)
    features["energy_max"] = np.max(rms)
    features["energy_range"] = features["energy_max"] - features["energy_min"]
    
    # Dynamic range
    features["dynamic_range"] = 20 * np.log10(
        features["energy_max"] / (features["energy_min"] + 1e-10)
    )
    
    # 6. Tempo/Rhythm Features
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])
    except Exception:
        features["tempo"] = 0.0
    
    # 7. Harmonic/Percussive Separation Features
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.sum(y_harmonic**2) / (np.sum(y**2) + 1e-10)
    percussive_ratio = np.sum(y_percussive**2) / (np.sum(y**2) + 1e-10)
    
    features["harmonic_ratio"] = harmonic_ratio
    features["percussive_ratio"] = percussive_ratio
    
    # 8. Chroma Features (tonal content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    features["chroma_mean"] = np.mean(chroma, axis=1)
    features["chroma_std"] = np.mean(np.std(chroma, axis=1))
    
    return features


def build_feature_vector(features: Dict[str, Any]) -> np.ndarray:
    """
    Convert extracted features into a flat feature vector for ML model.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        1D numpy array suitable for ML model input
    """
    vector_parts = []
    
    # MFCC features (40 * 5 = 200 features)
    vector_parts.append(features["mfcc_mean"])
    vector_parts.append(features["mfcc_std"])
    vector_parts.append(features["mfcc_min"])
    vector_parts.append(features["mfcc_max"])
    vector_parts.append(features["mfcc_delta"])
    
    # Mel features (32 * 2 = 64 features)
    vector_parts.append(features["mel_mean"])
    vector_parts.append(features["mel_std"])
    
    # Spectral features (10 features)
    spectral = np.array([
        features["spectral_centroid_mean"],
        features["spectral_centroid_std"],
        features["spectral_bandwidth_mean"],
        features["spectral_bandwidth_std"],
        features["spectral_rolloff_mean"],
        features["spectral_rolloff_std"],
        features["spectral_flatness_mean"],
        features["spectral_flatness_std"],
        features["zcr_mean"],
        features["zcr_std"]
    ])
    vector_parts.append(spectral)
    
    # Pitch features (6 features)
    pitch = np.array([
        features["pitch_mean"],
        features["pitch_std"],
        features["pitch_min"],
        features["pitch_max"],
        features["pitch_range"],
        features["voiced_ratio"]
    ])
    vector_parts.append(pitch)
    
    # Energy features (6 features)
    energy = np.array([
        features["energy_mean"],
        features["energy_std"],
        features["energy_min"],
        features["energy_max"],
        features["energy_range"],
        features["dynamic_range"]
    ])
    vector_parts.append(energy)
    
    # Additional features (4 features)
    additional = np.array([
        features["tempo"],
        features["harmonic_ratio"],
        features["percussive_ratio"],
        features["chroma_std"]
    ])
    vector_parts.append(additional)
    
    # Chroma mean (12 features)
    vector_parts.append(features["chroma_mean"])
    
    # Concatenate all features
    feature_vector = np.concatenate(vector_parts)
    
    # Handle NaN and Inf values
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_vector.astype(np.float32)


def get_feature_names() -> list:
    """Get the names of all features in the feature vector."""
    names = []
    
    # MFCC names
    for stat in ["mean", "std", "min", "max", "delta"]:
        for i in range(MFCC_N_COEFFS):
            names.append(f"mfcc_{i}_{stat}")
    
    # Mel names
    for stat in ["mean", "std"]:
        for i in range(32):
            names.append(f"mel_{i}_{stat}")
    
    # Spectral names
    names.extend([
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_flatness_mean", "spectral_flatness_std",
        "zcr_mean", "zcr_std"
    ])
    
    # Pitch names
    names.extend([
        "pitch_mean", "pitch_std", "pitch_min", 
        "pitch_max", "pitch_range", "voiced_ratio"
    ])
    
    # Energy names
    names.extend([
        "energy_mean", "energy_std", "energy_min",
        "energy_max", "energy_range", "dynamic_range"
    ])
    
    # Additional names
    names.extend([
        "tempo", "harmonic_ratio", "percussive_ratio", "chroma_std"
    ])
    
    # Chroma names
    for i in range(12):
        names.append(f"chroma_mean_{i}")
    
    return names
