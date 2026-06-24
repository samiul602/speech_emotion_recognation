# Speech Emotion Recognition (SER) — Full Project Guide

## Project Overview
A machine learning project that recognizes human emotions from speech audio.
Trained on RAVDESS and TESS datasets, achieving **90% accuracy** using MLP classifier.

---

## Final Results

| Model | Accuracy |
|---|---|
| MLP | **89.62%** 🏆 |
| CNN | 87.62% |
| Random Forest | 84.91% |
| SVM | 84.43% |

### Per Emotion Accuracy
| Emotion | Accuracy |
|---|---|
| Neutral | 94.93% |
| Angry | 94.07% |
| Fear | 90.76% |
| Surprise | 88.14% |
| Happy | 87.29% |
| Disgust | 86.55% |
| Sad | 84.75% |

---

## Project Structure

```
speech_emotion_recognition/
│
├── data/
│   ├── RAVDESS/                    # RAVDESS dataset (Actor_01 ... Actor_24)
│   └── TESS/                       # TESS dataset (flat structure, 2800 .wav files)
│
├── recordings/                     # Live demo recordings (timestamped .wav files)
│
├── ser_env/                        # Virtual environment (Python 3.11.9)
│
├── 01_data_loading.ipynb           # Load and merge datasets
├── 02_EDA.ipynb                    # Exploratory data analysis
├── 03_feature_extraction.ipynb     # Extract MFCC features
├── 04_model_training.ipynb         # Train 4 models and compare
├── 05_evaluation.ipynb             # Confusion matrix and classification report
├── 06_demo.ipynb                   # Live microphone demo
│
├── dataset.csv                     # Output of notebook 01 (4240 rows)
├── features.csv                    # Output of notebook 03 (4240 x 41 columns)
├── best_model.pkl                  # Saved MLP model
├── best_model_name.txt             # Contains "MLP"
├── scaler.pkl                      # StandardScaler fitted on training data
│
├── eda_emotion_distribution.png    # Plot from notebook 02
├── eda_waveforms.png               # Plot from notebook 02
├── eda_spectrograms.png            # Plot from notebook 02
├── eda_mfcc_heatmaps.png           # Plot from notebook 02
├── eda_durations.png               # Plot from notebook 02
├── training_curves.png             # CNN training curves from notebook 04
├── model_comparison.png            # Model accuracy comparison from notebook 04
├── confusion_matrix.png            # From notebook 05
└── per_emotion_accuracy.png        # From notebook 05
```

---

## Step 1 — Environment Setup

### Problem
Default Python version was **3.14** which is NOT supported by TensorFlow.
TensorFlow only supports up to Python 3.11.

### Solution
Installed **Python 3.11.9** separately and created a virtual environment.

### Commands
```bash
# Navigate to project folder
cd D:\Projects\CSE445\speech_emotion_recognition

# Create virtual environment using Python 3.11
py -3.11 -m venv ser_env

# Activate virtual environment
ser_env\Scripts\activate
# You should see (ser_env) at the start of the command line

# Install all required packages
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn librosa tqdm joblib jupyter sounddevice soundfile
```

### To activate the environment next time
```bash
cd D:\Projects\CSE445\speech_emotion_recognition
ser_env\Scripts\activate
jupyter notebook
```

---

## Step 2 — Datasets

### RAVDESS
- **Full name:** Ryerson Audio-Visual Database of Emotional Speech and Song
- **Speakers:** 24 actors (12 male, 12 female)
- **Files:** 1440 .wav files
- **Structure:** Actor_01/, Actor_02/, ... subfolders
- **Emotion encoding:** 3rd segment of filename
  - e.g. `03-01-06-01-02-01-12.wav` → `06` = fear
- **Emotions:** neutral, calm, happy, sad, angry, fear, disgust, surprise

### TESS
- **Full name:** Toronto Emotional Speech Set
- **Speakers:** 2 older female speakers (OAF, YAF)
- **Files:** 2800 .wav files
- **Structure:** Flat — all files in one folder
- **Emotion encoding:** Last segment of filename
  - e.g. `OAF_back_angry.wav` → `angry`
- **Emotions:** neutral, happy, sad, angry, fear, ps (pleasant surprise), disgust

### Emotion Merging Decision
- RAVDESS `calm` → merged into `neutral`
- TESS `ps` (pleasant surprise) → mapped to `surprise`
- **Final 7 classes:** neutral, happy, sad, angry, fear, disgust, surprise

### Combined Dataset
| Source | Files |
|---|---|
| RAVDESS | 1440 |
| TESS | 2800 |
| **Total** | **4240** |

---

## Step 3 — Notebook 01: Data Loading

**File:** `01_data_loading.ipynb`
**Output:** `dataset.csv`

### What it does
- Loads RAVDESS files by parsing filename (3rd segment = emotion code)
- Loads TESS files by parsing filename (last segment = emotion key)
- Merges calm → neutral
- Validates all file paths exist on disk
- Saves `dataset.csv` with columns: `path`, `emotion`, `source`

### Key Code — RAVDESS Loader
```python
# Filename: 03-01-06-01-02-01-12.wav
parts = fname.replace(".wav", "").split("-")
emotion_code = parts[2]   # "06" = fear
```

### Key Code — TESS Loader
```python
# Filename: OAF_back_angry.wav
parts = fname.replace(".wav", "").split("_")
emotion_key = parts[-1].lower()   # "angry"
```

### Output — dataset.csv
```
path                                    emotion   source
data/RAVDESS/Actor_01/03-01-...wav     happy     RAVDESS
data/TESS/OAF_back_angry.wav           angry     TESS
...
```

---

## Step 4 — Notebook 02: EDA

**File:** `02_EDA.ipynb`
**Output:** 4 PNG plot files

### What it does
- Plots emotion distribution (overall + by source)
- Plots waveforms for each emotion
- Plots spectrograms for each emotion
- Plots MFCC heatmaps for each emotion
- Plots audio duration distribution

### Libraries Used
```python
import librosa          # audio loading and feature extraction
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Step 5 — Notebook 03: Feature Extraction

**File:** `03_feature_extraction.ipynb`
**Output:** `features.csv`
**Time:** ~10-20 minutes

### What is MFCC?
MFCC (Mel Frequency Cepstral Coefficients) are numerical representations
of audio that capture how the sound spectrum changes over time.
We extract 40 MFCC coefficients per audio file and take their mean.
Result: each audio file becomes a vector of 40 numbers.

### Settings
```python
SAMPLE_RATE = 22050   # Hz
DURATION    = 3       # seconds (crop or pad all audio to 3s)
N_MFCC      = 40      # number of coefficients
```

### Feature Extraction Function
```python
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    # Pad if shorter than 3 seconds
    if len(y) < SAMPLE_RATE * DURATION:
        y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)))
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1)   # shape: (40,)
```

### Output — features.csv
- 4240 rows (one per audio file)
- 41 columns (mfcc_0 to mfcc_39 + label)
- label is encoded as integer: neutral=0, happy=1, sad=2, angry=3, fear=4, disgust=5, surprise=6

---

## Step 6 — Notebook 04: Model Training

**File:** `04_model_training.ipynb`
**Output:** `best_model.pkl`, `scaler.pkl`, `best_model_name.txt`

### Data Split — 70/20/10
```
70% → Training   (model learns from this)
10% → Validation (model checks itself during training — automatic)
20% → Testing    (final exam — touched only once)
```

### Why StandardScaler?
Raw MFCC values vary wildly (e.g. -500 to +30).
StandardScaler normalizes all features to mean=0, std=1.
This helps all models learn better and faster.
**Important:** Scaler is fit on training data only, then applied to val and test.

### 4 Models Trained

**SVM (Support Vector Machine)**
```python
SVC(kernel='rbf', C=1.0, random_state=42)
```

**Random Forest**
```python
RandomForestClassifier(n_estimators=200, random_state=42)
```

**MLP (Multi-Layer Perceptron)**
```python
MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', max_iter=300)
```

**CNN (Convolutional Neural Network)**
```python
# Architecture:
Conv1D(64) → BatchNorm → MaxPool → Dropout(0.3)
Conv1D(128) → BatchNorm → MaxPool → Dropout(0.3)
Flatten → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3)
Dense(7, softmax)
# Early stopping: patience=10, monitors val_loss
```

### Why MLP beat CNN?
MFCC features are already averaged into flat 1D vectors (40 numbers).
MLP handles flat vectors naturally.
CNN works better with full 2D spectrograms, not averaged MFCCs.
This is a great talking point for viva!

### Best Model Selection
```python
# Automatically finds and saves the best model
best_model_name = max(results, key=results.get)
joblib.dump(mlp_model, 'best_model.pkl')   # since MLP won
```

---

## Step 7 — Notebook 05: Evaluation

**File:** `05_evaluation.ipynb`
**Output:** `confusion_matrix.png`, `per_emotion_accuracy.png`

### What it does
- Loads `best_model.pkl` and `scaler.pkl`
- Recreates the exact same test split using `random_state=42`
- Prints classification report (precision, recall, F1 per emotion)
- Plots confusion matrix heatmap
- Plots per emotion accuracy bar chart

### Key Metrics
- **Precision:** Of all predicted "happy", how many were actually happy?
- **Recall:** Of all actual "happy", how many did we correctly predict?
- **F1-score:** Harmonic mean of precision and recall

---

## Step 8 — Notebook 06: Live Demo

**File:** `06_demo.ipynb`

### What it does
- Records 3 seconds of audio from microphone
- Saves recording with timestamp to `recordings/` folder
- Extracts MFCC features from recording
- Applies same scaler normalization
- Predicts emotion with confidence %
- Shows probability bar chart for all 7 emotions

### Known Issue — Domain Mismatch
Model trained on studio-quality acted audio (RAVDESS/TESS).
Live microphone audio is slightly different (room noise, mic quality).
This causes occasional wrong predictions.

### Fix Applied
```python
# Normalize audio amplitude
y = y / np.max(np.abs(y))
# Trim leading/trailing silence
y, _ = librosa.effects.trim(y, top_db=20)
```

### Tips for Better Live Predictions
- Speak expressively and clearly
- Happy → laugh, speak fast "hahaha this is amazing!"
- Angry → speak loudly "I am very ANGRY right now!"
- Sad → speak slowly and quietly "I feel so sad today..."
- Surprised → gasp "Oh WOW! I can't believe this!"

---

## Libraries Summary

| Library | Version | Purpose |
|---|---|---|
| tensorflow | 2.x | CNN and deep learning |
| scikit-learn | latest | SVM, RF, MLP, scaler, metrics |
| librosa | latest | Audio loading and MFCC extraction |
| numpy | latest | Numerical operations |
| pandas | latest | DataFrame and CSV handling |
| matplotlib | latest | Plotting |
| seaborn | latest | Heatmaps and styled plots |
| tqdm | latest | Progress bar during feature extraction |
| joblib | latest | Saving/loading sklearn models |
| sounddevice | latest | Recording microphone audio |
| soundfile | latest | Saving recorded audio as .wav |

---

## Important Files

| File | Created by | Used by |
|---|---|---|
| `dataset.csv` | Notebook 01 | Notebooks 02, 03 |
| `features.csv` | Notebook 03 | Notebooks 04, 05 |
| `scaler.pkl` | Notebook 04 | Notebooks 05, 06 |
| `best_model.pkl` | Notebook 04 | Notebooks 05, 06 |
| `best_model_name.txt` | Notebook 04 | Notebooks 05, 06 |

---

## How to Run the Project Again

```bash
# 1. Navigate to project folder
cd D:\Projects\CSE445\speech_emotion_recognition

# 2. Activate virtual environment
ser_env\Scripts\activate

# 3. Launch Jupyter
jupyter notebook

# 4. Run notebooks in order:
#    01_data_loading.ipynb       → creates dataset.csv
#    02_EDA.ipynb                → creates EDA plots
#    03_feature_extraction.ipynb → creates features.csv (takes 10-20 min)
#    04_model_training.ipynb     → trains models, saves best_model.pkl
#    05_evaluation.ipynb         → evaluation plots
#    06_demo.ipynb               → live mic demo

# 5. For demo only (if models already trained):
#    Just run 06_demo.ipynb directly
```

---

## Viva Key Points

1. **Why RAVDESS + TESS?** — Combined dataset gives more data (4240 files), better generalization, matches recent paper approaches

2. **Why merge calm → neutral?** — Calm and neutral are acoustically very similar, reduces class confusion, standard practice in literature

3. **Why MFCC?** — Most widely used feature for speech emotion recognition, captures spectral characteristics of speech efficiently

4. **Why 70/20/10 split?** — Separate validation set for tuning (early stopping), test set touched only once for honest final evaluation

5. **Why MLP beat CNN?** — MFCC features are averaged flat vectors, MLP handles them naturally, CNN needs 2D spectrograms to show its advantage

6. **Why 90% accuracy?** — Strong feature extraction (MFCC), proper normalization (StandardScaler), sufficient data (4240 samples), tuned model architecture

7. **Live demo issue?** — Domain mismatch between studio-recorded training data and real microphone audio, solved partially with amplitude normalization and silence trimming
```



##Honestly no need for your current project because:

we already have 4 models covering all categories — classical ML (SVM, RF) and neural networks (MLP, CNN)
Adding weak models like Naive Bayes or Decision Tree will lower your comparison chart without adding value
90% accuracy with MLP is already excellent for a course project
More models = more training time + more complexity

Instead of adding more models, focus on explaining why each model performed the way it did:

Why SVM got 84%? — Good with high dimensional data but struggles with complex emotion boundaries
Why Random Forest got 84%? — Ensemble of trees, reduces overfitting but limited by flat MFCC features
Why MLP got 90%? — Best suited for flat 1D feature vectors like averaged MFCCs
Why CNN got 87%? — Needs 2D input to shine, averaged MFCCs don't give it enough spatial structure