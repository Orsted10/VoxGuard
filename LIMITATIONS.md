# VoxGuard API - Known Limitations

## Model Limitations

### Dataset Coverage
- **Limited Training Data**: The model works best with clear, studio-quality audio. Training data coverage for all 5 supported languages (Tamil, English, Hindi, Malayalam, Telugu) may vary.
- **Accent Variations**: Performance may vary across different regional accents within each language.
- **Speaker Diversity**: Model accuracy depends on the diversity of speakers in the training data.

### Audio Quality Requirements
- **Minimum Duration**: Audio clips shorter than 0.5 seconds cannot be reliably classified.
- **Maximum Duration**: Only the first 30 seconds are analyzed for very long clips.
- **Background Noise**: Heavy background noise may reduce classification accuracy.
- **Compression Artifacts**: Highly compressed audio may affect feature extraction quality.

### Detection Capabilities
- **Evolving TTS Technology**: Modern text-to-speech systems are constantly improving. This model may not detect the most advanced deepfake voices.
- **Voice Cloning**: Sophisticated voice cloning that captures micro-expressions may be harder to detect.
- **Not Foolproof**: This is a detection aid, not a definitive verification system.

## Technical Limitations

### Performance
- **Latency**: Audio processing takes 1-3 seconds depending on file size and server load.
- **Concurrent Requests**: High concurrent load may increase response times.
- **Memory Usage**: Feature extraction is memory-intensive for long audio files.

### Audio Format
- **MP3 Only**: Currently only MP3 format is supported (Base64 encoded).
- **Mono Conversion**: Stereo audio is automatically converted to mono.
- **Sample Rate**: Audio is resampled to 22.05kHz for analysis.

## Language Support

### Supported Languages
✓ Tamil  
✓ English  
✓ Hindi  
✓ Malayalam  
✓ Telugu  

### Not Supported
- Other Indian languages (Kannada, Bengali, Marathi, etc.)
- Non-Indian languages
- Mixed-language audio (code-switching)

## Ethical Considerations

### Intended Use
- This tool is designed for **detection and verification purposes only**.
- Should be used as one component of a broader verification process.
- Not intended for mass surveillance or unauthorized monitoring.

### False Positives/Negatives
- **False Positives**: Some human voices with unusual characteristics may be flagged as AI.
- **False Negatives**: Sophisticated deepfakes may evade detection.
- Always consider results as probabilistic, not deterministic.

## Future Improvements

1. **Transformer-based Models**: Moving from MFCC+SVM to transformer architectures
2. **Real-time Streaming**: Support for live audio analysis
3. **More Languages**: Expanding to all major Indian languages
4. **Watermark Detection**: Identifying audio watermarks from major TTS providers
5. **Ensemble Methods**: Combining multiple detection approaches
6. **Continuous Learning**: Regular model updates with new deepfake samples

## Reporting Issues

If you encounter:
- Consistent misclassifications
- Audio processing errors
- Performance issues

Please report to the development team with:
- Sample audio (if permitted)
- Expected vs actual classification
- Confidence score received
- Any error messages

---

*VoxGuard v1.0.0 - Built for responsible AI use*
