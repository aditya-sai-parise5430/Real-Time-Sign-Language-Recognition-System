# 🤟 Real-Time Sign Language Recognition System

A deep learning-powered system for real-time American Sign Language (ASL) alphabet recognition using computer vision and LSTM networks.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- ✅ **Real-time Recognition** - Recognizes ASL alphabet (A-Z) at 30-60 FPS
- ✅ **Bidirectional System** - Sign-to-Text AND Text-to-Sign conversion
- ✅ **High Accuracy** - 90%+ validation accuracy
- ✅ **Modern UI** - Netflix-themed full-screen interface
- ✅ **Word Builder** - Compose words from recognized letters
- ✅ **Comprehensive Evaluation** - Multiple metrics and visualizations
- ✅ **Data Augmentation** - Robust to lighting/speed variations

## 🎬 Demo

### Real-Time Recognition
![Demo GIF](demo/recognition_demo.gif)

### Text-to-Sign Conversion
![Text to Sign](demo/text_to_sign_demo.gif)

## 🏗️ System Architecture

```
Input Video → MediaPipe Hand Detection → Keypoint Extraction → 
LSTM Sequence Model → Letter Recognition → Word Builder → Output
```

### Model Architecture
- **3x Bidirectional LSTM layers** (128→160→128 units)
- **Attention mechanism** for temporal focus
- **Batch Normalization** after each layer
- **Dropout (0.4-0.5)** for regularization
- **L2 Regularization** to prevent overfitting

### Key Specifications
| Parameter | Value |
|-----------|-------|
| Sequence Length | 30 frames |
| Features per Frame | 63 (21 landmarks × 3D) |
| Classes | 26 (A-Z) |
| Confidence Threshold | 90% |
| Model Size | ~8-10 MB |

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time recognition)
- GPU (optional, for faster training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

2. **Create virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models**
```bash
# Download from release page or Google Drive
# Place model files in project root:
# - model(0.2).json
# - best_model(0.2).h5
```

## 🚀 Usage

### 1. Real-Time Recognition
```bash
python app.py
```
**Controls:**
- `SPACE` - Add current letter to word
- `BACKSPACE` - Delete last letter
- `ENTER` - Clear word
- `Q` - Quit

### 2. Collect Training Data
```bash
python collectdata.py
```
**Controls:**
- `0` - Start recording (300 images per session)
- `1` - Stop recording
- `-/+` - Navigate between letters
- `A-Z` - Jump to specific letter

### 3. Preprocess Data
```bash
python data.py
```
Converts images to MediaPipe keypoint sequences.

### 4. Train Model
```bash
# Basic model
python trainmodel.py

# Improved model with augmentation
python newtrainmodel.py
```

### 5. Evaluate Model
```bash
python evaluate_model.py --model_json model(0.35).json --weights newmodel(0.35).h5 --out evaluation_results
```

### 6. Text-to-Sign Conversion
```bash
python text_to_sign.py
```
Type any word to see corresponding sign language images.

## 📊 Model Performance

### Training Results
- **Validation Accuracy:** 92.3%
- **Top-3 Accuracy:** 97.8%
- **Top-5 Accuracy:** 99.1%
- **Average ROC-AUC:** 0.984

### Confusion Matrix
![Confusion Matrix](evaluation_results/confusion_matrix_normalized.png)

### Per-Class Performance
![Per Class Metrics](evaluation_results/per_class_metrics.png)

## 📁 Project Structure

```
sign-language-recognition/
│
├── app.py                      # Main real-time recognition app
├── collectdata.py              # Data collection tool
├── data.py                     # Data preprocessing
├── trainmodel.py               # Basic model training
├── newtrainmodel.py            # Advanced model with augmentation
├── evaluate_model.py           # Comprehensive evaluation
├── text_to_sign.py             # Text-to-sign converter
├── function.py                 # Utility functions
│
├── model(0.2).json             # Model architecture (example)
├── best_model(0.2).h5          # Pre-trained weights (download separately)
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── LICENSE                     # MIT License
│
├── Image/                      # Training images (A-Z folders) - NOT in repo
├── MP_Data/                    # Preprocessed keypoints - NOT in repo
├── Output/                     # Text-to-sign outputs
└── evaluation_results/         # Evaluation metrics and graphs
```

## 🔬 Technical Details

### Data Augmentation Techniques
1. **Gaussian Noise** (2% noise factor)
2. **Time Warping** (temporal distortion)
3. **Magnitude Warping** (scale variation)

### Training Configuration
- **Optimizer:** Adam with cosine annealing
- **Learning Rate:** 0.001 → 1e-6 (with warmup)
- **Batch Size:** 32
- **Epochs:** 300 (with early stopping)
- **Loss:** Categorical Crossentropy
- **Class Weights:** Balanced

### Preprocessing Pipeline
```
Raw Image → MediaPipe Detection → 21 Hand Landmarks → 
(x,y,z) × 21 = 63 features → Sequence of 30 frames → 
Normalized [0,1] → Model Input
```

## 🎯 Use Cases

1. **Communication Aid** - Help deaf/mute individuals communicate
2. **Education** - Teaching ASL to students
3. **Accessibility** - Voice-to-sign and sign-to-voice systems
4. **Gaming** - Gesture-based game controls
5. **Medical** - Patient communication in hospitals

## 🚧 Limitations & Future Work

### Current Limitations
- ✗ Static alphabet only (no words/sentences)
- ✗ Single hand detection
- ✗ Requires good lighting
- ✗ Limited to 26 ASL letters

### Future Enhancements
- [ ] Full sentence recognition
- [ ] Two-handed signs
- [ ] Dynamic gestures (J, Z with motion)
- [ ] Mobile app deployment
- [ ] Real-time translation with TTS
- [ ] Multi-language support

## 📄 Dataset

Due to GitHub size limits, the dataset is not included. 

### Create Your Own Dataset:
1. Run `python collectdata.py`
2. Collect 300+ images per letter (A-Z)
3. Vary lighting, angles, backgrounds
4. Run `python data.py` to preprocess

### Or Download Pre-collected Dataset:
[Download from Google Drive](https://drive.google.com/your-link-here) (Example link)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- MediaPipe team for hand tracking library
- TensorFlow/Keras for deep learning framework
- ASL community for inspiration
- [Add your college/institution name]

## 📧 Contact

For questions or collaboration:
- Email: adityasaiparise@gmail.com

---

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ for accessibility and inclusion
