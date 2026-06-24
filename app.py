import os
import io
import numpy as np
import librosa
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SAMPLE_RATE    = 22050
DURATION       = 3
N_MFCC         = 40
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
EMOTION_EMOJIS = {
    'neutral':  '😐', 'happy':   '😄', 'sad':     '😢',
    'angry':    '😠', 'fear':    '😨', 'disgust': '🤢', 'surprise': '😲'
}
EMOTION_COLORS = {
    'neutral': '#7f8c8d', 'happy': '#f1c40f', 'sad': '#3498db',
    'angry': '#e74c3c', 'fear': '#9b59b6', 'disgust': '#27ae60', 'surprise': '#e67e22'
}

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
                
        .home-title {
        text-align: center;
        padding: 30px 0;
    }

    # .home-title h1 {
    #     color: #e0e0e0;
    #     font-size: 48px;
    #     margin-bottom: 10px;
          text-align: center;
    # }

    # .home-title h4 {
    #     color: #8892a4;
    #     font-size: 16px;
    #      font-weight: 10;
    # }


    /* Main background */
    .stApp { background: #707070;
     background: linear-gradient(90deg,rgba(112, 112, 112, 1) 0%, rgba(0, 0, 0, 1) 100%); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        
        border-right: 1px solid #2a2d3e;
    }

            

                /* Upload area */
    [data-testid="stFileUploaderDropzone"] {
        background: #1a1d2e;
        border: 2px dashed #4a90d9;
        border-radius: 12px;
        padding: 20px;
    }

    /* Drag & drop text */
    [data-testid="stFileUploaderDropzone"] p {
        color: #c0c8d8 !important;
        font-size: 16px !important;
    }

    /* Browse files button */
    [data-testid="stFileUploaderDropzone"] button {
       
           
        background: #4a90d9 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    } 
 

    /* Cards */
    .card {
        background: #1a1d2e;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        border: 1px solid #2a2d3e;
    }

    /* Emotion result box */
    .emotion-result {
        background: linear-gradient(135deg, #1a1d2e, #12151f);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        border: 2px solid #4a90d9;
        margin: 16px 0;
    }

    .emotion-emoji { font-size: 72px; }
    .emotion-name  { font-size: 36px; font-weight: 700; color: #4a90d9; margin: 8px 0; }
    .emotion-conf  { font-size: 18px; color: #8892a4; }

    /* Stat cards */
    .stat-card {
        background: #1a1d2e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2a2d3e;
    }
    .stat-number { font-size: 32px; font-weight: 700; color: #4a90d9; }
    .stat-label  { font-size: 13px; color: #8892a4; margin-top: 4px; }

    /* Section title */
    .section-title {
        font-size: 22px;
        font-weight: 600;
        color: #e0e0e0;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #4a90d9;
    }

    /* Pipeline step */
    .pipeline-step {
        background: #1a1d2e;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 4px solid #4a90d9;
        font-size: 15px;
    }

    /* Hide streamlit branding */
    # MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    # header  { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('best_model_name.txt', 'r') as f:
        model_name = f.read().strip()
    scaler = joblib.load('scaler.pkl')
    if model_name == 'CNN':
        from tensorflow.keras.models import load_model as keras_load
        model = keras_load('best_model.h5')
    else:
        model = joblib.load('best_model.pkl')
    return model, scaler, model_name

model, scaler, model_name = load_model()

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, duration=DURATION, mono=True)
    target_length = SAMPLE_RATE * DURATION
    # Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)
    # Pad or crop
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1), y

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict_emotion(features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    if model_name == 'CNN':
        features_cnn = features_scaled.reshape(1, features_scaled.shape[1], 1)
        probabilities = model.predict(features_cnn)[0]
    else:
        probabilities = model.predict_proba(features_scaled)[0]
    predicted_idx     = np.argmax(probabilities)
    predicted_emotion = EMOTION_LABELS[predicted_idx]
    confidence        = probabilities[predicted_idx] * 100
    return predicted_emotion, confidence, probabilities

# ─────────────────────────────────────────────
# PLOT WAVEFORM
# ─────────────────────────────────────────────
def plot_waveform(y):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor('#1a1d2e')
    ax.set_facecolor('#1a1d2e')
    ax.plot(np.linspace(0, DURATION, len(y)), y, color='#4a90d9', linewidth=0.8)
    ax.set_xlabel('Time (seconds)', color='#8892a4')
    ax.set_ylabel('Amplitude', color='#8892a4')
    ax.tick_params(colors='#8892a4')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2d3e')
    ax.set_title('Audio Waveform', color='#e0e0e0', pad=10)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# PLOT PROBABILITIES
# ─────────────────────────────────────────────
def plot_probabilities(probabilities, predicted_emotion):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1a1d2e')
    ax.set_facecolor('#1a1d2e')

    colors = [EMOTION_COLORS[e] for e in EMOTION_LABELS]
    labels = [f"{EMOTION_EMOJIS[e]} {e}" for e in EMOTION_LABELS]
    bars   = ax.bar(labels, probabilities * 100, color=colors,
                    edgecolor=['white' if e == predicted_emotion else 'none' for e in EMOTION_LABELS],
                    linewidth=2)

    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{prob*100:.1f}%', ha='center', fontsize=9,
                fontweight='bold', color='#e0e0e0')

    ax.set_ylabel('Probability (%)', color='#8892a4')
    ax.set_ylim([0, 115])
    ax.tick_params(colors='#8892a4', axis='both')
    ax.tick_params(axis='x', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2d3e')
    ax.set_title('Emotion Probabilities', color='#e0e0e0', pad=10)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ SER System")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🎙️ Predict", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<div style='color:#8892a4; font-size:12px;'>"
        "Speech Emotion Recognition<br>"
        "RAVDESS + TESS Dataset<br>"
        "CSE 445 — Machine Learning"
        "</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="home-title">
        <h1 style="color:#e0e0e0; font-size: 48px; margin-bottom: 10px text-align: center;">🎙️ Speech Emotion Recognition</h1>
    </div>
    """, unsafe_allow_html=True)
    # st.markdown("# 🎙️ Speech Emotion Recognition")
    # st.markdown("#### Detect human emotions from speech audio using Machine Learning")
    # st.markdown("---")
    
   
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>4,240</div>
            <div class='stat-label'>Audio Files</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>90%</div>
            <div class='stat-label'>Best Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>7</div>
            <div class='stat-label'>Emotions</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>4</div>
            <div class='stat-label'>Models Trained</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Emotion classes
    st.markdown("<div class='section-title'>Detectable Emotions</div>", unsafe_allow_html=True)
    cols = st.columns(7)
    for col, emotion in zip(cols, EMOTION_LABELS):
        with col:
            st.markdown(
                f"<div class='stat-card'>"
                f"<div style='font-size:32px'>{EMOTION_EMOJIS[emotion]}</div>"
                f"<div style='font-size:13px; color:#8892a4; margin-top:6px'>{emotion.capitalize()}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Datasets
    st.markdown("<div class='section-title'>Datasets Used</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='card'>
            <h4 style='color:#4a90d9'>RAVDESS</h4>
            <p style='color:#8892a4'>Ryerson Audio-Visual Database of Emotional Speech and Song</p>
            <ul style='color:#c0c8d8'>
                <li>24 professional actors</li>
                <li>1,440 audio files</li>
                <li>8 emotions (calm merged into neutral)</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card'>
            <h4 style='color:#4a90d9'>TESS</h4>
            <p style='color:#8892a4'>Toronto Emotional Speech Set</p>
            <ul style='color:#c0c8d8'>
                <li>2 female speakers (OAF, YAF)</li>
                <li>2,800 audio files</li>
                <li>7 emotions</li>
            </ul>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 2 — PREDICT
# ─────────────────────────────────────────────
elif page == "🎙️ Predict":
    # st.markdown("# 🎙️ Predict Emotion")
    # st.markdown("Upload a `.wav` audio file to detect the emotion.")
    # st.markdown("---")
    
    st.markdown("""
    <div class="home-title">
        <h1 style="color:#e0e0e0; font-size: 48px margin-bottom: 10px;"> 🎙️ Predict Emotion</h1>
        <h4 style="color:#8892a4; font-size: 16px; margin-bottom: 10px;">Upload a `.wav` audio file to detect the emotion.</h4>
     </div>
    """, unsafe_allow_html=True)

    # Mic recording note
    # st.info("💡 **Local users:** You can also record directly from your microphone using `06_demo.ipynb` in the project folder.")
    st.markdown("""
    <div class="custom-info">
        <div>
         <h2 style="color: #8892a4; font-size: 16px; margin-bottom: 10px;">💡 Local users: You can also record directly from your microphone using `06_demo.ipynb` in the project folder.</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)




    # uploaded_file = st.file_uploader(
    #     "Upload a WAV audio file",
    #     type=["wav"],
    #     help="Upload any .wav file — from the dataset or your own recording"
    # )

    st.markdown(
        "<div class='section-title'>Upload Audio File</div>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "",
        type=["wav"],
        help="Upload any .wav file — from the dataset or your own recording",
        label_visibility="collapsed"
    )
   




    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()

        st.markdown("<div class='section-title'>Uploaded Audio</div>", unsafe_allow_html=True)
        st.audio(audio_bytes, format='audio/wav')

        with st.spinner("Analyzing emotion..."):
            try:
                features, y = extract_features(audio_bytes)
                predicted_emotion, confidence, probabilities = predict_emotion(features)

                # Waveform
                st.markdown("<div class='section-title'>Waveform</div>", unsafe_allow_html=True)
                st.pyplot(plot_waveform(y))

                # Result
                st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='emotion-result'>"
                    f"<div class='emotion-emoji'>{EMOTION_EMOJIS[predicted_emotion]}</div>"
                    f"<div class='emotion-name'>{predicted_emotion.upper()}</div>"
                    f"<div class='emotion-conf'>Confidence: {confidence:.2f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Probabilities
                st.markdown("<div class='section-title'>All Emotion Probabilities</div>", unsafe_allow_html=True)
                st.pyplot(plot_probabilities(probabilities, predicted_emotion))

            except Exception as e:
                st.error(f"Error processing audio: {e}")
    else:
        st.markdown("""
        <div class='card' style='text-align:center; padding:48px'>
            <div style='font-size:48px'>📁</div>
            <div style='color:#8892a4; margin-top:12px'>Upload a .wav file to get started</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "📊 Model Performance":
    # st.markdown("# 📊 Model Performance")
    # st.markdown("Comparison of all 4 trained models on the test set.")
    # st.markdown("---")

    st.markdown("""
      <div>
            <h1 style="color:#e0e0e0; font-size: 48px; margin-bottom: 10px; text-align: center;">📊 Model Performance</h1>
             <h4 style="color: #8892a4; font-size: 16px; margin-bottom: 10px;">Comparison of all 4 trained models on the test set.</h4>
      </div>
     """, unsafe_allow_html=True)

    # Model comparison
    st.markdown("<div class='section-title'>Model Accuracy Comparison</div>", unsafe_allow_html=True)

    model_results = {
        'SVM':           84.43,
        'Random Forest': 84.91,
        'MLP':           89.62,
        'CNN':           87.62,
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1d2e')
    ax.set_facecolor('#1a1d2e')
    colors = ['#4a90d9' if m == 'MLP' else '#2a2d3e' for m in model_results.keys()]
    bars = ax.bar(model_results.keys(), model_results.values(), color=colors, edgecolor='#4a90d9', linewidth=0.5)
    for bar, acc in zip(bars, model_results.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc:.2f}%', ha='center', fontweight='bold', color='#e0e0e0')
    ax.set_ylabel('Accuracy (%)', color='#8892a4')
    ax.set_ylim([75, 95])
    ax.tick_params(colors='#8892a4')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2d3e')
    ax.set_title('Model Accuracy Comparison', color='#e0e0e0', pad=10)
    legend = [mpatches.Patch(color='#4a90d9', label='Best Model (MLP)')]
    ax.legend(handles=legend, facecolor='#1a1d2e', labelcolor='#e0e0e0')
    plt.tight_layout()
    st.pyplot(fig)

    # Per emotion accuracy
    st.markdown("<div class='section-title'>Per Emotion Accuracy (MLP)</div>", unsafe_allow_html=True)

    per_emotion = {
        'neutral': 94.93, 'happy': 87.29, 'sad': 84.75,
        'angry': 94.07, 'fear': 90.76, 'disgust': 86.55, 'surprise': 88.14
    }

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor('#1a1d2e')
    ax2.set_facecolor('#1a1d2e')
    bar_colors = [EMOTION_COLORS[e] for e in per_emotion.keys()]
    labels2 = [f"{EMOTION_EMOJIS[e]} {e}" for e in per_emotion.keys()]
    bars2 = ax2.bar(labels2, per_emotion.values(), color=bar_colors)
    for bar, acc in zip(bars2, per_emotion.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold', color='#e0e0e0')
    ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax2.set_ylabel('Accuracy (%)', color='#8892a4')
    ax2.set_ylim([75, 102])
    ax2.tick_params(colors='#8892a4')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#2a2d3e')
    ax2.set_title('Per Emotion Accuracy', color='#e0e0e0', pad=10)
    ax2.legend(facecolor='#1a1d2e', labelcolor='#e0e0e0')
    plt.tight_layout()
    st.pyplot(fig2)

    # Classification report table
    st.markdown("<div class='section-title'>Classification Report</div>", unsafe_allow_html=True)
    import pandas as pd
    report_data = {
        'Emotion':   ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'],
        'Precision': [0.86, 0.91, 0.84, 0.91, 0.91, 0.91, 0.95],
        'Recall':    [0.95, 0.87, 0.85, 0.94, 0.91, 0.87, 0.88],
        'F1-Score':  [0.90, 0.89, 0.84, 0.93, 0.91, 0.89, 0.91],
        'Support':   [138,  118,  118,  118,  119,  119,  118],
    }
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df.set_index('Emotion'), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 4 — ABOUT
# ─────────────────────────────────────────────
elif page == "ℹ️ About":
    # st.markdown("# ℹ️ About This Project")
    # st.markdown("---")

    st.markdown("""
      <div>
            <h1 style="color:#e0e0e0; font-size: 48px; margin-bottom: 10px; text-align: center;">ℹ️ About This Project</h1>
      </div>
     """, unsafe_allow_html=True)
    

    # Project info
    st.markdown("""
    <div class='card'>
        <h4 style='color:#4a90d9'>Speech Emotion Recognition System</h4>
        <p style='color:#c0c8d8'>
        This project builds a machine learning system that detects human emotions from speech audio.
        It was developed as part of CSE 445 — Machine Learning course using the RAVDESS and TESS datasets,
        achieving 90% accuracy with an MLP classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline
    st.markdown("<div class='section-title'>How It Works</div>", unsafe_allow_html=True)
    steps = [
        ("1", "Audio Input", "Raw .wav audio file uploaded or recorded from microphone"),
        ("2", "Preprocessing", "Audio loaded at 22050 Hz, normalized, silence trimmed, padded to 3 seconds"),
        ("3", "Feature Extraction", "40 MFCC coefficients extracted using librosa, averaged over time axis"),
        ("4", "Normalization", "Features scaled using StandardScaler (mean=0, std=1)"),
        ("5", "Prediction", "MLP classifier predicts emotion from scaled features"),
        ("6", "Output", "Predicted emotion displayed with confidence % and probability chart"),
    ]
    for num, title, desc in steps:
        st.markdown(
            f"<div class='pipeline-step'>"
            f"<span style='color:#4a90d9; font-weight:700'>Step {num} — {title}</span><br>"
            f"<span style='color:#8892a4'>{desc}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tech stack
    st.markdown("<div class='section-title'>Tech Stack</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='card'>
            <h4 style='color:#4a90d9'>Libraries</h4>
            <ul style='color:#c0c8d8'>
                <li>librosa — audio processing and MFCC extraction</li>
                <li>scikit-learn — ML models, scaler, metrics</li>
                <li>TensorFlow/Keras — CNN model</li>
                <li>Streamlit — web application</li>
                <li>matplotlib — visualizations</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card'>
            <h4 style='color:#4a90d9'>Models Trained</h4>
            <ul style='color:#c0c8d8'>
                <li>SVM — 84.43%</li>
                <li>Random Forest — 84.91%</li>
                <li>CNN — 87.62%</li>
                <li>MLP — 89.62% 🏆 Best</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # Why MLP beat CNN
    st.markdown("<div class='section-title'>Why MLP Outperformed CNN?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='color:#c0c8d8'>
        MFCC features are averaged into flat 1D vectors (40 numbers per audio file).
        MLP handles flat 1D vectors naturally and efficiently.
        CNN works better with full 2D spectrograms where spatial structure matters.
        Since we fed averaged MFCCs rather than full spectrograms, MLP had the advantage.
        </p>
    </div>
    """, unsafe_allow_html=True)