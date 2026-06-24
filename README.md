# 🎙️ Speech Emotion Recognition (SER)

A machine learning system that detects human emotions from speech audio using RAVDESS and TESS datasets.

## 🚀 Live Demo

**Streamlit App:** [Click here to try the app](<APP_LINK_HERE>)

---

## 📊 Results

| Model | Accuracy |
|---|---|
| MLP | **89.62%** 🏆 |
| CNN | 87.62% |
| Random Forest | 84.91% |
| SVM | 84.43% |

---

## 🎭 Detectable Emotions

😐 Neutral &nbsp; 😄 Happy &nbsp; 😢 Sad &nbsp; 😠 Angry &nbsp; 😨 Fear &nbsp; 🤢 Disgust &nbsp; 😲 Surprise

---

## 📁 Project Structure

```
├── 01_data_loading.ipynb           # Load and merge RAVDESS + TESS
├── 02_EDA.ipynb                    # Exploratory data analysis
├── 03_feature_extraction.ipynb     # Extract MFCC features
├── 04_model_training.ipynb         # Train and compare 4 models
├── 05_evaluation.ipynb             # Confusion matrix and report
├── 06_demo.ipynb                   # Live microphone demo
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── best_model.pkl                  # Saved MLP model
├── scaler.pkl                      # Saved StandardScaler
└── best_model_name.txt             # Best model name
```

---

## 🗂️ Datasets

| Dataset | Speakers | Files | Emotions |
|---|---|---|---|
| RAVDESS | 24 actors (12M, 12F) | 1,440 | 8 (calm merged into neutral) |
| TESS | 2 female speakers | 2,800 | 7 |
| **Total** | | **4,240** | **7** |

---

## ⚙️ How It Works

```
Audio Input (.wav)
      ↓
Preprocessing (normalize, trim silence, pad to 3s)
      ↓
Feature Extraction (40 MFCC coefficients)
      ↓
Normalization (StandardScaler)
      ↓
MLP Classifier
      ↓
Predicted Emotion + Confidence %
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **librosa** — audio processing and MFCC extraction
- **scikit-learn** — ML models, scaler, metrics
- **TensorFlow/Keras** — CNN model
- **Streamlit** — web application
- **matplotlib / seaborn** — visualizations

---

## 🏃 Run Locally

```bash
# Clone the repository
git clone https://github.com/samiul602/speech_emotion_recognation.git
cd speech_emotion_recognation

# Create virtual environment (Python 3.11)
py -3.11 -m venv ser_env
ser_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📓 Run Notebooks

Run notebooks in this order:

```
01_data_loading.ipynb        → creates dataset.csv
02_EDA.ipynb                 → creates EDA plots
03_feature_extraction.ipynb  → creates features.csv
04_model_training.ipynb      → trains models, saves best_model.pkl
05_evaluation.ipynb          → evaluation plots
06_demo.ipynb                → live mic demo
```

> ⚠️ Notebooks require RAVDESS and TESS datasets placed in `data/RAVDESS/` and `data/TESS/` folders.

---

## 📚 Course

**CSE 445 — Machine Learning**
North South University