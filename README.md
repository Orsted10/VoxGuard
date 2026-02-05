# 🎙️ VoxGuard - Multilingual AI Voice Deepfake Detector

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-grade REST API that detects AI-generated voice deepfakes across multiple Indian languages.**

VoxGuard leverages advanced audio feature extraction (MFCC, spectral analysis, prosodic features) and machine learning to classify voice audio as **AI_GENERATED** or **HUMAN** with confidence scores and human-readable explanations.

---

## 🌟 Features

- **🌐 Multilingual Support**: Tamil, English, Hindi, Malayalam, Telugu
- **⚡ Fast Analysis**: Real-time audio processing (< 2 seconds)
- **📊 Explainable AI**: Human-readable explanations for every classification
- **🔐 Secure API**: API key authentication
- **📈 Production Ready**: Docker deployment, comprehensive error handling
- **📖 Auto-Generated Docs**: Swagger UI at `/docs`

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client                               │
│              (cURL / Web App / Mobile App)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ POST /api/voice-detection
                              │ x-api-key: sk_xxx
                              │ {language, audioFormat, audioBase64}
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
├──────────────┬──────────────────────┬───────────────────────┤
│  Auth Layer  │   Voice Detection    │   Response Handler    │
│  (API Key)   │      Router          │   (JSON + Explain)    │
├──────────────┴──────────────────────┴───────────────────────┤
│                     Core Module                              │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │ features.py│→ │  model.py   │→ │  explanations.py     │  │
│  │ MFCC/Pitch │  │  ML Predict │  │  Human-Readable      │  │
│  └────────────┘  └─────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd voxguard-api

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set environment variable
set VOXGUARD_API_KEY=sk_test_123456789  # Windows
# export VOXGUARD_API_KEY=sk_test_123456789  # Linux/Mac

# Run the server
uvicorn voxguard_api.api.main:app --reload

# Server runs at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Option 2: Docker

```bash
# Build image
docker build -t voxguard-api .

# Run container
docker run -p 8000:8000 -e VOXGUARD_API_KEY=sk_test_123456789 voxguard-api
```

### Option 3: Docker Compose

```bash
docker-compose up -d
```

### Option 4: Vercel Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Set environment variable in Vercel dashboard:
# VOXGUARD_API_KEY = your_secret_key

# Deploy
vercel
```

---

## 📡 API Reference

### Authentication

All endpoints (except `/health`) require the `x-api-key` header:

```
x-api-key: sk_test_123456789
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (no auth) |
| `GET` | `/info` | API information |
| `GET` | `/metrics` | Usage statistics |
| `POST` | `/api/voice-detection` | Detect AI voice |

### Voice Detection Request

**POST** `/api/voice-detection`

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAAD..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `language` | string | One of: Tamil, English, Hindi, Malayalam, Telugu |
| `audioFormat` | string | Must be "mp3" |
| `audioBase64` | string | Base64-encoded MP3 audio |

### Voice Detection Response

**Success (200)**

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Analysis strongly indicates AI-generated audio. Detected: unnaturally consistent pitch, unnaturally uniform volume levels and overly clean spectral characteristics. Voice patterns analyzed for Tamil language characteristics."
}
```

**Error (4xx/5xx)**

```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=voxguard_api
```

### Test with cURL

```bash
# Health check
curl http://localhost:8000/health

# Voice detection
curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_AUDIO_HERE"
  }'
```

### Generate Base64 Audio (Python)

```python
import base64

with open("audio.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    print(audio_base64)
```

---

## 🧠 Model Training

### Prepare Dataset

Place audio files in the following structure:

```
data/raw/
├── ai_generated/
│   ├── sample1.mp3
│   ├── sample2.wav
│   └── ...
└── human/
    ├── real1.mp3
    ├── real2.wav
    └── ...
```

### Train Model

```bash
# Prepare features (or use --synthetic for demo)
python scripts/prepare_dataset.py --synthetic --num-synthetic 500

# Train model
python scripts/train_model.py --model-type gradient_boosting

# Model saved to models/ai_detector.pkl
```

### Model Types

- `gradient_boosting` (default) - Best accuracy
- `random_forest` - Faster training
- `svm` - Good for small datasets

---

## 📁 Project Structure

```
voxguard-api/
├── api/                      # Vercel entry point
│   └── index.py
├── voxguard_api/
│   ├── api/
│   │   ├── main.py           # FastAPI app
│   │   ├── schemas.py        # Pydantic models
│   │   └── routers/
│   │       ├── auth.py       # API key auth
│   │       └── voice_detection.py
│   └── core/
│       ├── config.py         # Settings
│       ├── features.py       # Audio extraction
│       ├── model.py          # ML prediction
│       ├── language_id.py    # Language validation
│       └── explanations.py   # Result explanations
├── models/                   # Trained models
├── scripts/
│   ├── prepare_dataset.py    # Data prep
│   └── train_model.py        # Training
├── tests/                    # Pytest tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── vercel.json               # Vercel config
└── README.md
```

---

## ⚙️ Configuration

Environment variables (set in `.env` or deployment platform):

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXGUARD_API_KEY` | `sk_test_123456789` | API key for authentication |
| `MODEL_PATH` | `models/ai_detector.pkl` | Path to model file |
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 🔬 Technical Details

### Features Extracted

| Category | Features |
|----------|----------|
| **MFCC** | 40 coefficients × 5 stats (mean, std, min, max, delta) |
| **Mel Spectrogram** | 32 bands × 2 stats |
| **Spectral** | Centroid, bandwidth, rolloff, flatness, ZCR |
| **Pitch** | Mean, std, min, max, range, voiced ratio |
| **Energy** | Mean, std, min, max, range, dynamic range |
| **Additional** | Tempo, harmonic ratio, percussive ratio, chroma |

**Total: 302 features per audio sample**

### Classification Logic

1. Audio decoded from Base64 MP3
2. Features extracted using librosa
3. Features scaled using StandardScaler
4. Classification by GradientBoosting model
5. Explanation generated from feature analysis

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Latency | < 2 seconds |
| Max audio duration | 30 seconds |
| Min audio duration | 0.5 seconds |
| Supported sample rates | Any (resampled to 22.05kHz) |

---

## 🛡️ Error Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `400` | Bad request (invalid input) |
| `401` | Invalid API key |
| `422` | Validation error |
| `500` | Internal server error |

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📬 Contact

**VoxGuard Team**  
Built with ❤️ for ethical AI voice detection

---

*Protecting authenticity in the age of AI-generated content.*
