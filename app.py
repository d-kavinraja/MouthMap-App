import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from typing import List
import tempfile
import os
import logging
import time
from datetime import datetime

# Configure the page (must be the first Streamlit command)
st.set_page_config(
    page_title="MouthMap - 2.0",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_WEIGHTS_PATH = "Model/40th-epoch-model-checkpoint-keras-default-v1/checkpoint.weights.h5"
TEST_DATASET_FOLDER = "test_video_dataset"  # contains videos and .txt alignments

# App configuration
APP_TITLE = "MouthMap"
APP_SUBTITLE = "Advanced Lip Reading Technology"
APP_VERSION = "2.0"

# Define color palettes for themes
THEMES = {
    "Dark": {
        "bg_color": "#1e1e1e",
        "text_color": "#ffffff",
        "accent_color": "#00c3ff",
        "secondary_color": "#7d3ac1",
        "card_bg": "#2c2c2c",
        "gradient": "linear-gradient(to right, #434343, #000000)",
    },
    "Ocean": {
        "bg_color": "#e0f7fa",
        "text_color": "#01579b",
        "accent_color": "#00838f",
        "secondary_color": "#0277bd",
        "card_bg": "#e1f5fe",
        "gradient": "linear-gradient(to right, #4facfe, #00f2fe)",
    },
    "Forest": {
        "bg_color": "#e8f5e9",
        "text_color": "#1b5e20",
        "accent_color": "#2e7d32",
        "secondary_color": "#558b2f",
        "card_bg": "#f1f8e9",
        "gradient": "linear-gradient(to right, #72c975, #2c9033)",
    },
    "Sunset": {
        "bg_color": "#fce4ec",
        "text_color": "#880e4f",
        "accent_color": "#c2185b",
        "secondary_color": "#f50057",
        "card_bg": "#ffebee",
        "gradient": "linear-gradient(to right, #ff8177, #ff867a, #ff8c7f, #f99185, #cf556c)",
    }
}

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789'?! "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# --- Functions ---
def load_video(path: str, target_frames: int = 75) -> tf.Tensor:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = tf.image.rgb_to_grayscale(frame)
            frame = frame[190:236, 80:220, :]
            frames.append(frame)

    cap.release()
    while len(frames) < target_frames:
        frames.append(frames[-1] if frames else tf.zeros((46, 140, 1), dtype=tf.uint8))

    frames = tf.stack(frames[:target_frames])
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std
    return tf.expand_dims(frames, axis=0)

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred,
                                           input_length * tf.ones((batch_len, 1), dtype="int64"),
                                           label_length * tf.ones((batch_len, 1), dtype="int64"))

def build_model():
    inputs = tf.keras.layers.Input(shape=(75, 46, 140, 1), name="input")
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    x = tf.keras.layers.Conv3D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    x = tf.keras.layers.Conv3D(75, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    x = tf.keras.layers.Reshape((75, -1))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(char_to_num.get_vocabulary()), activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

def clean_prediction(predicted_text: str) -> str:
    return ' '.join(predicted_text.replace("!", " ").split())

def predict_video(video_path: str, target_frames: int = 75):
    st.session_state['progress_bar'].progress(10)
    time.sleep(0.5)  # Simulate loading time
    
    model = build_model()
    st.session_state['progress_bar'].progress(30)
    time.sleep(0.5)  # Simulate loading time
    
    model.load_weights(MODEL_WEIGHTS_PATH)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=CTCLoss)
    st.session_state['progress_bar'].progress(50)
    time.sleep(0.5)  # Simulate loading time

    video_input = load_video(video_path, target_frames)
    st.session_state['progress_bar'].progress(70)
    time.sleep(0.5)  # Simulate loading time
    
    yhat = model.predict(video_input)
    st.session_state['progress_bar'].progress(90)
    time.sleep(0.5)  # Simulate loading time
    
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    predicted_text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')
    st.session_state['progress_bar'].progress(100)
    time.sleep(0.3)  # Simulate loading time
    
    return clean_prediction(predicted_text)

def parse_alignment(file_path: str) -> str:
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        words = [line.strip().split()[-1] for line in lines]
        while words and words[0] == "sil":
            words.pop(0)
        while words and words[-1] == "sil":
            words.pop()
        return " ".join(words)
    except Exception as e:
        logger.error(f"Error parsing alignment file: {e}")
        return f"Error: {e}"

def save_and_convert_video(uploaded_file, temp_dir):
    with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Convert to MP4 if needed (fallback for .mpg)
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.unlink(tmp_path)
        raise ValueError("Invalid video file.")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_path.replace('.mpg', '.mp4'), fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    os.unlink(tmp_path)
    return tmp_path.replace('.mpg', '.mp4')

def load_animate_css():
    return """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    """

def render_lottie_animation():
    return """
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <div class="animation-container">
        <lottie-player src="https://assets3.lottiefiles.com/packages/lf20_bXJJ9J4H2o.json" background="transparent" speed="1" style="width: 100%; height: 300px;" loop autoplay></lottie-player>
    </div>
    """

def render_stats_cards(target_frames):
    # Mock stats for demonstration (removed accuracy rate)
    videos_processed = st.session_state.get('videos_processed', 0)
    
    return f"""
    <div class="stats-container">
        <div class="stat-card animate__animated animate__fadeInUp">
            <div class="stat-icon">🎥</div>
            <div class="stat-value">{videos_processed}</div>
            <div class="stat-label">Videos Processed</div>
        </div>
        <div class="stat-card animate__animated animate__fadeInUp" style="animation-delay: 0.2s">
            <div class="stat-icon">⚡</div>
            <div class="stat-value">{target_frames}</div>
            <div class="stat-label">Frames Analyzed</div>
        </div>
    </div>
    """

def render_features_section():
    return """
    <div class="features-section animate__animated animate__fadeIn">
        <h2>✨ Key Features</h2>
        <div class="features-container">
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">AI-Powered Recognition</div>
                <div class="feature-desc">State-of-the-art deep learning models for precise lip movement tracking</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">High Precision</div>
                <div class="feature-desc">Advanced algorithms for reliable text prediction from visual input</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Real-time Processing</div>
                <div class="feature-desc">Fast analysis with optimized neural networks for quick results</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Detailed Insights</div>
                <div class="feature-desc">Comprehensive performance metrics for better understanding</div>
            </div>
        </div>
    </div>
    """

def render_how_it_works():
    return """
    <div class="how-it-works animate__animated animate__fadeIn">
        <h2>🔄 How It Works</h2>
        <div class="steps-container">
            <div class="step">
                <div class="step-number">1</div>
                <div class="step-content">
                    <div class="step-title">Upload Video</div>
                    <div class="step-desc">Select or upload a video file containing lip movements</div>
                </div>
            </div>
            <div class="step">
                <div class="step-number">2</div>
                <div class="step-content">
                    <div class="step-title">AI Processing</div>
                    <div class="step-desc">Our neural network analyzes the lip movements frame by frame</div>
                </div>
            </div>
            <div class="step">
                <div class="step-number">3</div>
                <div class="step-content">
                    <div class="step-title">Text Generation</div>
                    <div class="step-desc">The model translates visual patterns into predicted speech</div>
                </div>
            </div>
            <div class="step">
                <div class="step-number">4</div>
                <div class="step-content">
                    <div class="step-title">View Results</div>
                    <div class="step-desc">See the predicted text from the video</div>
                </div>
            </div>
        </div>
    </div>
    """

def render_applications():
    return """
    <div class="applications-section animate__animated animate__fadeIn">
        <h2>🌟 Applications</h2>
        <div class="applications-container">
            <div class="application-card">
                <div class="application-icon">👂</div>
                <div class="application-title">Accessibility</div>
                <div class="application-desc">Assist deaf and hard-of-hearing individuals</div>
            </div>
            <div class="application-card">
                <div class="application-icon">🎬</div>
                <div class="application-title">Media Production</div>
                <div class="application-desc">Automated dubbing and subtitle generation</div>
            </div>
            <div class="application-card">
                <div class="application-icon">🔒</div>
                <div class="application-title">Security</div>
                <div class="application-desc">Remote surveillance and communication monitoring</div>
            </div>
            <div class="application-card">
                <div class="application-icon">📱</div>
                <div class="application-title">Silent Communication</div>
                <div class="application-desc">Understand speech in noisy environments</div>
            </div>
        </div>
    </div>
    """

def apply_custom_css(theme):
    selected_theme = THEMES[theme]
    
    # Base CSS with professional fonts (Inter and Montserrat)
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Montserrat:wght@600;700&display=swap');

        body {{
            font-family: 'Inter', sans-serif;
            background-color: {selected_theme["bg_color"]};
            color: {selected_theme["text_color"]};
            transition: all 0.3s ease;
            margin: 0;
        }}

        .main {{
            background-color: {selected_theme["bg_color"]};
            color: {selected_theme["text_color"]};
            padding: 20px;
            border-radius: 10px;
            font-size: 18px;
        }}

        h1, h2, h3 {{
            font-family: 'Montserrat', sans-serif;
            color: {selected_theme["text_color"]};
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        h1 {{
            font-size: 48px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 10px;
        }}

        h2 {{
            font-size: 32px;
            font-weight: 600;
            margin-top: 30px;
        }}

        .subtitle {{
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
            opacity: 0.85;
            font-family: 'Inter', sans-serif;
        }}

        .stButton>button {{
            background-color: {selected_theme["accent_color"]};
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            transition: all 0.3s;
            font-size: 16px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-family: 'Inter', sans-serif;
        }}
        
        .stButton>button:hover {{
            background-color: {selected_theme["secondary_color"]};
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }}

        /* Video Container */
        .video-container {{
            border: 2px solid rgba(0,0,0,0.1);
            border-radius: 10px;
            padding: 20px;
            background: {selected_theme["card_bg"]};
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            margin: 20px 0;
            transition: all 0.3s ease;
            color: {selected_theme["text_color"]};
        }}

        .video-container:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 28px rgba(0,0,0,0.15);
        }}

        /* Result Card */
        .result-card {{
            background: {selected_theme["card_bg"]};
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            margin-top: 30px;
            transition: all 0.3s ease;
            border-left: 5px solid {selected_theme["accent_color"]};
            color: {selected_theme["text_color"]};
        }}

        .result-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 28px rgba(0,0,0,0.15);
        }}

        .result-title {{
            font-weight: 600;
            font-size: 18px;
            margin-bottom: 8px;
            color: {selected_theme["accent_color"]};
            font-family: 'Montserrat', sans-serif;
        }}

        .result-content {{
            font-size: 20px;
            line-height: 1.6;
            font-family: 'Inter', sans-serif;
        }}

        /* Stats Cards */
        .stats-container {{
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            margin: 30px 0;
            gap: 15px;
        }}

        .stat-card {{
            background: {selected_theme["card_bg"]};
            border-radius: 10px;
            padding: 20px;
            flex: 1;
            min-width: 180px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            color: {selected_theme["text_color"]};
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.15);
        }}

        .stat-icon {{
            font-size: 32px;
            margin-bottom: 10px;
        }}

        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: {selected_theme["accent_color"]};
            font-family: 'Montserrat', sans-serif;
        }}

        .stat-label {{
            font-size: 16px;
            opacity: 0.8;
            font-family: 'Inter', sans-serif;
        }}

        /* Features Section */
        .features-section {{
            margin: 40px 0;
        }}

        .features-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: space-between;
        }}

        .feature-card {{
            background: {selected_theme["card_bg"]};
            border-radius: 10px;
            padding: 25px;
            flex: 1 1 200px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            color: {selected_theme["text_color"]};
        }}

        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.15);
        }}

        .feature-icon {{
            font-size: 32px;
            margin-bottom: 15px;
            color: {selected_theme["accent_color"]};
        }}

        .feature-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
            font-family: 'Montserrat', sans-serif;
        }}

        .feature-desc {{
            font-size: 14px;
            opacity: 0.8;
            line-height: 1.5;
            font-family: 'Inter', sans-serif;
        }}

        /* How it Works */
        .how-it-works {{
            margin: 40px 0;
        }}

        .steps-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        .step {{
            display: flex;
            align-items: center;
            background: {selected_theme["card_bg"]};
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            color: {selected_theme["text_color"]};
        }}

        .step:hover {{
            transform: translateX(5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}

        .step-number {{
            background: {selected_theme["accent_color"]};
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-weight: bold;
            margin-right: 20px;
            flex-shrink: 0;
            font-family: 'Inter', sans-serif;
        }}

        .step-content {{
            flex-grow: 1;
        }}

        .step-title {{
            font-weight: 600;
            font-size: 18px;
            margin-bottom: 5px;
            font-family: 'Montserrat', sans-serif;
        }}

        .step-desc {{
            font-size: 14px;
            opacity: 0.8;
            font-family: 'Inter', sans-serif;
        }}

        /* Applications */
        .applications-section {{
            margin: 40px 0;
        }}

        .applications-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: space-between;
        }}

        .application-card {{
            background: {selected_theme["card_bg"]};
            border-radius: 10px;
            padding: 20px;
            flex: 1 1 150px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            color: {selected_theme["text_color"]};
        }}

        .application-card:hover {{
            transform: scale(1.05);
            box-shadow: 0 8px 15px rgba(0,0,0,0.15);
        }}

        .application-icon {{
            font-size: 32px;
            margin-bottom: 15px;
            color: {selected_theme["accent_color"]};
        }}

        .application-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
            font-family: 'Montserrat', sans-serif;
        }}

        .application-desc {{
            font-size: 14px;
            opacity: 0.8;
            font-family: 'Inter', sans-serif;
        }}

        /* Header Animation */
        .animate-header {{
            animation: fadeIn 1.5s ease-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Upload Area */
        .upload-area {{
            border: 2px dashed rgba(0,0,0,0.2);
            border-radius: 10px;
            padding: 40px 20px;
            text-align: center;
            background: {selected_theme["card_bg"]};
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
            color: {selected_theme["text_color"]};
        }}

        .upload-area:hover {{
            border-color: {selected_theme["accent_color"]};
            background: rgba(0,0,0,0.02);
        }}

        .upload-icon {{
            font-size: 48px;
            margin-bottom: 15px;
            color: {selected_theme["accent_color"]};
        }}

        .upload-text {{
            font-size: 18px;
            font-family: 'Inter', sans-serif;
        }}

        /* Progress Bar */
        .progress-container {{
            margin: 20px 0;
        }}

        .custom-progress {{
            height: 10px;
            background-color: #e9ecef;
            border-radius: 5px;
            margin-bottom: 10px;
            overflow: hidden;
        }}

        .progress-bar {{
            height: 100%;
            background: {selected_theme["gradient"]};
            width: 0%;
            transition: width 0.5s ease;
            border-radius: 5px;
        }}

        .progress-status {{
            font-size: 14px;
            text-align: right;
            font-family: 'Inter', sans-serif;
            color: {selected_theme["text_color"]};
        }}

        /* Mode Selector */
        .mode-selector {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}

        .mode-button {{
            background: {selected_theme["card_bg"]};
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 8px;
            padding: 12px 24px;
            flex: 1;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            color: {selected_theme["text_color"]};
            font-family: 'Inter', sans-serif;
        }}

        .mode-button.active {{
            background: {selected_theme["accent_color"]};
            color: white;
            border-color: {selected_theme["accent_color"]};
            font-weight: bold;
        }}

        .mode-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}

        /* Footer */
        .footer {{
            margin-top: 50px;
            padding-top: 30px;
            border-top: 1px solid rgba(0,0,0,0.1);
            text-align: center;
        }}

        .footer-content {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .footer-links {{
            display: flex;
            gap: 20px;
        }}

        .footer-link {{
            color: {selected_theme["accent_color"]};
            text-decoration: none;
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
        }}

        .footer-link:hover {{
            color: {selected_theme["secondary_color"]};
            text-decoration: underline;
        }}

        .footer-copyright {{
            opacity: 0.7;
            font-size: 14px;
            font-family: 'Inter', sans-serif;
            color: {selected_theme["text_color"]};
        }}

        /* Animation Container */
        .animation-container {{
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            .stats-container, 
            .features-container, 
            .applications-container {{
                flex-direction: column;
            }}

            .stat-card, 
            .feature-card, 
            .application-card {{
                width: 100%;
            }}

            h1 {{
                font-size: 32px;
            }}

            .subtitle {{
                font-size: 16px;
            }}
        }}

        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
        }}

        ::-webkit-scrollbar-track {{
            background: {selected_theme["bg_color"]}; 
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {selected_theme["accent_color"]}; 
            border-radius: 5px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {selected_theme["secondary_color"]}; 
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {selected_theme["card_bg"]};
            border-right: 1px solid rgba(0,0,0,0.1);
            color: {selected_theme["text_color"]};
        }}

        /* Navigation Pills */
        .nav-pills {{
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}

        .nav-pill {{
            padding: 8px 16px;
            background: {selected_theme["card_bg"]};
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid rgba(0,0,0,0.1);
            color: {selected_theme["text_color"]};
            font-family: 'Inter', sans-serif;
        }}

        .nav-pill.active {{
            background: {selected_theme["accent_color"]};
            color: white;
            border-color: {selected_theme["accent_color"]};
        }}

        .nav-pill:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
    </style>
    """
    return css

# --- UI ---
def main():
    # Initialize session state values if they don't exist
    if 'videos_processed' not in st.session_state:
        st.session_state['videos_processed'] = 0
    if 'current_mode' not in st.session_state:
        st.session_state['current_mode'] = "📤 Upload Video"
    if 'current_tab' not in st.session_state:
        st.session_state['current_tab'] = "Home"
    if 'progress_status' not in st.session_state:
        st.session_state['progress_status'] = ""
    if 'progress_bar' not in st.session_state:
        st.session_state['progress_bar'] = st.progress(0)

    # Sidebar
    with st.sidebar:
        st.image("https://i.ibb.co/jhVJP1G/lipsync-logo.png", width=250)
        st.markdown("### ⚙️ Settings & Controls")
        
        theme = st.selectbox(
            "🎨 Select Theme",
            list(THEMES.keys()),
            index=0
        )
        
        st.markdown("#### 📊 Display Options")
        show_stats = st.checkbox("Show Statistics", value=True)
        show_features = st.checkbox("Show Features Section", value=True)
        show_how_it_works = st.checkbox("Show How It Works", value=True)
        
        st.markdown("#### 🔧 Advanced Settings")
        target_frames = st.slider("🎞️ Target Frames", 50, 100, 75, help="Number of frames to process")
        
        st.markdown("#### 📱 App Info")
        st.markdown(f"**Version:** {APP_VERSION}")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")
        
        st.markdown("---")
        
        st.markdown("""
        ### 🌟 Quick Links
        - [Documentation](#)
        - [Report Issues](#)
        - [API Access](#)
        """)

    # Apply custom CSS based on selected theme
    st.markdown(load_animate_css(), unsafe_allow_html=True)
    st.markdown(apply_custom_css(theme), unsafe_allow_html=True)

    # Header Section
    st.markdown(f"""
    <div class="animate-header">
        <h1>🎬 {APP_TITLE}</h1>
        <p class="subtitle">{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats Cards
    if show_stats:
        st.markdown(render_stats_cards(target_frames), unsafe_allow_html=True)
    
    # Main Tabs Navigation
    tabs = ["🏠 Home", "💬 Recognition", "📊 Analytics", "ℹ️ About"]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        tab1 = st.button("🏠 Home", key="tab1", use_container_width=True)
    with col2:
        tab2 = st.button("💬 Recognition", key="tab2", use_container_width=True)
    with col3:
        tab3 = st.button("📊 Analytics", key="tab3", use_container_width=True)
    with col4:
        tab4 = st.button("ℹ️ About", key="tab4", use_container_width=True)

    # Tab Logic
    if tab1:
        st.session_state['current_tab'] = "Home"
    elif tab2:
        st.session_state['current_tab'] = "Recognition"
    elif tab3:
        st.session_state['current_tab'] = "Analytics"
    elif tab4:
        st.session_state['current_tab'] = "About"

    if st.session_state['current_tab'] == "Home":
        st.markdown(render_lottie_animation(), unsafe_allow_html=True)
        if show_features:
            st.markdown(render_features_section(), unsafe_allow_html=True)
        if show_how_it_works:
            st.markdown(render_how_it_works(), unsafe_allow_html=True)
        st.markdown(render_applications(), unsafe_allow_html=True)

    elif st.session_state['current_tab'] == "Recognition":
        st.markdown("<h2>💬 Video Recognition</h2>", unsafe_allow_html=True)

        # Mode Selection
        mode = st.radio(
            "🛠️ Select Mode",
            ["📤 Upload Video", "📂 Use Test Dataset"],
            index=["📤 Upload Video", "📂 Use Test Dataset"].index(st.session_state['current_mode']),
            key="mode_radio",
            horizontal=True
        )
        st.session_state['current_mode'] = mode

        if mode == "📤 Upload Video":
            uploaded_video = st.file_uploader("📽️ Upload a video (.mp4, .webm, .mpg)", type=["mp4", "webm", "mpg"])
            if uploaded_video:
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_video_path = save_and_convert_video(uploaded_video, temp_dir)
                        st.markdown("<div class='video-container'>", unsafe_allow_html=True)
                        st.video(temp_video_path)
                        st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"⚠️ Video preview failed. Ensure the file is a valid video format. Error: {str(e)}")
                    logger.error(f"Video preview error: {str(e)}")
                    try:
                        st.markdown("<div class='video-container'>", unsafe_allow_html=True)
                        st.video(uploaded_video)
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e2:
                        st.warning(f"⚠️ Original file preview also failed. Error: {str(e2)}")
                        logger.error(f"Original file preview error: {str(e2)}")

                if st.button("🚀 Predict Sentence"):
                    with st.spinner("🔄 Processing video..."):
                        st.session_state['progress_bar'] = st.progress(0)
                        st.session_state['progress_status'] = "Starting prediction..."
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            tmp.write(uploaded_video.read())
                            tmp_path = tmp.name
                        try:
                            result = predict_video(tmp_path, target_frames)
                            st.session_state['videos_processed'] += 1
                            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                            st.success(f"🎉 Predicted Sentence: {result}")
                            st.markdown("</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                        finally:
                            os.unlink(tmp_path)

        elif mode == "📂 Use Test Dataset":
            test_files = [f for f in os.listdir(TEST_DATASET_FOLDER) if f.endswith((".mp4", ".webm", ".mpg"))]
            if not test_files:
                st.warning("⚠️ No video files found in test dataset folder.")
            else:
                selected_file = st.selectbox("🎥 Select a test video", test_files)
                if selected_file:
                    video_path = os.path.join(TEST_DATASET_FOLDER, selected_file)
                    align_path = os.path.splitext(video_path)[0] + ".align"

                    try:
                        if video_path.endswith(".mpg"):
                            with tempfile.TemporaryDirectory() as temp_dir:
                                with open(video_path, 'rb') as f:
                                    temp_video_path = save_and_convert_video(f, temp_dir)
                                st.markdown("<div class='video-container'>", unsafe_allow_html=True)
                                st.video(temp_video_path)
                                st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='video-container'>", unsafe_allow_html=True)
                            st.video(video_path)
                            st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"⚠️ Video preview failed. Ensure the file is a valid video format. Error: {str(e)}")
                        logger.error(f"Video preview error: {str(e)}")

                    if st.button("🚀 Predict & Compare"):
                        with st.spinner("🔄 Processing video..."):
                            st.session_state['progress_bar'] = st.progress(0)
                            st.session_state['progress_status'] = "Starting prediction..."
                            try:
                                pred_sentence = predict_video(video_path, target_frames)
                                actual_sentence = parse_alignment(align_path)
                                st.session_state['videos_processed'] += 1
                                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                                st.success("✅ Prediction Complete")
                                st.markdown(f"**Predicted Sentence:** {pred_sentence}")
                                st.markdown(f"**Actual Sentence:** {actual_sentence}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

    elif st.session_state['current_tab'] == "Analytics":
        st.markdown("<h2>📊 Analytics Dashboard</h2>")
        # Placeholder for analytics content (removed accuracy)
        st.markdown("<div class='dashboard-cards'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='dashboard-card'>
            <div class='dashboard-card-header'>
                <div class='dashboard-card-title'>Total Videos</div>
                <div class='dashboard-card-icon'>🎥</div>
            </div>
            <div class='dashboard-card-content'>0</div>
            <div class='dashboard-card-footer'>Updated today</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state['current_tab'] == "About":
        st.markdown("<h2>ℹ️ About MouthMap</h2>")
        st.markdown("""
        <p>MouthMap is an innovative application powered by advanced deep learning models to interpret lip movements and convert them into text. Developed with cutting-edge technology, it aims to enhance accessibility, media production, and silent communication.</p>
        <p><strong>Version:</strong> 2.0 | <strong>Release Date:</strong> 2025-04-29</p>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <div class="footer-links">
                <a href="#" class="footer-link">Privacy Policy</a>
                <a href="#" class="footer-link">Terms of Service</a>
                <a href="#" class="footer-link">Contact Us</a>
            </div>
            <div class="footer-copyright">© 2025 MouthMap by xAI. All rights reserved.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()