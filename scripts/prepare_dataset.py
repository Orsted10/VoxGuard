"""
VoxGuard Dataset Preparation Script
Prepares audio dataset for training the deepfake detector
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from voxguard_api.core.features import extract_audio_features, build_feature_vector

# Try to import librosa for audio loading
try:
    import librosa
except ImportError:
    print("Error: librosa not installed. Run: pip install librosa")
    sys.exit(1)


def load_audio_file(file_path: str, sample_rate: int = 22050) -> Tuple[np.ndarray, int]:
    """Load an audio file and return waveform and sample rate."""
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def process_audio_files(
    audio_dir: Path,
    label: int,
    sample_rate: int = 22050
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Process all audio files in a directory.
    
    Args:
        audio_dir: Path to directory containing audio files
        label: Label for all files (0=HUMAN, 1=AI_GENERATED)
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (feature_vectors, labels)
    """
    features = []
    labels = []
    
    # Supported audio extensions
    extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    if not audio_dir.exists():
        print(f"Warning: Directory not found: {audio_dir}")
        return features, labels
    
    audio_files = [
        f for f in audio_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in extensions
    ]
    
    print(f"Found {len(audio_files)} audio files in {audio_dir}")
    
    for i, audio_file in enumerate(audio_files):
        try:
            # Load audio
            y, sr = load_audio_file(str(audio_file), sample_rate)
            
            if y is None or len(y) == 0:
                continue
            
            # Skip very short files
            duration = len(y) / sr
            if duration < 0.5:
                print(f"  Skipping {audio_file.name}: too short ({duration:.2f}s)")
                continue
            
            # Truncate very long files
            if duration > 30:
                y = y[:sr * 30]
            
            # Extract features
            audio_features = extract_audio_features(y, sr)
            feature_vector = build_feature_vector(audio_features)
            
            features.append(feature_vector)
            labels.append(label)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files...")
                
        except Exception as e:
            print(f"  Error processing {audio_file.name}: {e}")
            continue
    
    return features, labels


def create_synthetic_data(num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic training data when no real dataset is available.
    
    This generates feature vectors with characteristics of AI vs Human audio.
    For production, use real data!
    """
    print(f"Generating {num_samples} synthetic samples...")
    
    features = []
    labels = []
    
    for i in range(num_samples):
        # Determine class
        is_ai = i < num_samples // 2
        label = 1 if is_ai else 0
        
        # Generate base feature vector
        feature_vector = np.random.randn(302).astype(np.float32)
        
        if is_ai:
            # AI characteristics
            # Low pitch variation (indices 274-279)
            feature_vector[274] = np.random.uniform(50, 150)    # pitch_mean
            feature_vector[275] = np.random.uniform(5, 20)      # pitch_std (low)
            feature_vector[279] = np.random.uniform(0.9, 0.99)  # voiced_ratio (high)
            
            # Low dynamic range (indices 280-285)
            feature_vector[284] = np.random.uniform(5, 15)      # dynamic_range (low)
            
            # High harmonic ratio (index 287)
            feature_vector[287] = np.random.uniform(0.85, 0.98) # harmonic_ratio (high)
            
            # Low spectral flatness (index 270)
            feature_vector[270] = np.random.uniform(0.001, 0.02)
            
        else:
            # Human characteristics
            # Higher pitch variation
            feature_vector[274] = np.random.uniform(80, 200)    # pitch_mean
            feature_vector[275] = np.random.uniform(30, 80)     # pitch_std (higher)
            feature_vector[279] = np.random.uniform(0.6, 0.85)  # voiced_ratio (natural)
            
            # Higher dynamic range
            feature_vector[284] = np.random.uniform(25, 60)     # dynamic_range
            
            # More varied harmonic ratio
            feature_vector[287] = np.random.uniform(0.5, 0.8)   # harmonic_ratio
            
            # Higher spectral flatness
            feature_vector[270] = np.random.uniform(0.05, 0.15)
        
        features.append(feature_vector)
        labels.append(label)
    
    return np.array(features), np.array(labels)


def main():
    """Main entry point for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare VoxGuard training dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw audio (with ai_generated/ and human/ subdirs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed features"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate for audio"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data if no real data available"
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=500,
        help="Number of synthetic samples to generate"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    all_labels = []
    
    # Check for real data
    ai_dir = data_dir / "ai_generated"
    human_dir = data_dir / "human"
    
    has_real_data = ai_dir.exists() or human_dir.exists()
    
    if has_real_data:
        print("Processing real audio data...")
        
        # Process AI-generated audio
        if ai_dir.exists():
            print(f"\nProcessing AI-generated audio from {ai_dir}...")
            ai_features, ai_labels = process_audio_files(ai_dir, label=1, sample_rate=args.sample_rate)
            all_features.extend(ai_features)
            all_labels.extend(ai_labels)
            print(f"  Extracted {len(ai_labels)} AI samples")
        
        # Process human audio
        if human_dir.exists():
            print(f"\nProcessing human audio from {human_dir}...")
            human_features, human_labels = process_audio_files(human_dir, label=0, sample_rate=args.sample_rate)
            all_features.extend(human_features)
            all_labels.extend(human_labels)
            print(f"  Extracted {len(human_labels)} human samples")
    
    # Generate synthetic data if needed
    if args.synthetic or len(all_features) < 50:
        if len(all_features) < 50:
            print("\nNot enough real data found. Generating synthetic data...")
        else:
            print("\nAdding synthetic data as requested...")
        
        synth_features, synth_labels = create_synthetic_data(args.num_synthetic)
        all_features.extend(synth_features)
        all_labels.extend(synth_labels)
    
    if len(all_features) == 0:
        print("\nError: No data available!")
        print("Please either:")
        print(f"  1. Add audio files to {data_dir}/ai_generated/ and {data_dir}/human/")
        print("  2. Run with --synthetic flag to generate synthetic data")
        sys.exit(1)
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Save processed data
    features_path = output_dir / "features.npy"
    labels_path = output_dir / "labels.npy"
    
    np.save(features_path, X)
    np.save(labels_path, y)
    
    # Save metadata
    metadata = {
        "num_samples": len(y),
        "num_ai_generated": int(np.sum(y == 1)),
        "num_human": int(np.sum(y == 0)),
        "feature_dim": X.shape[1],
        "sample_rate": args.sample_rate,
        "has_real_data": has_real_data
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Dataset preparation complete!")
    print(f"{'='*50}")
    print(f"Total samples: {len(y)}")
    print(f"  AI-generated: {np.sum(y == 1)}")
    print(f"  Human: {np.sum(y == 0)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"\nSaved to:")
    print(f"  {features_path}")
    print(f"  {labels_path}")


if __name__ == "__main__":
    main()
