# --- Project Name: MouthMap ---

# Developed by:
# Kavin Raja D (212222240047)
# Karnala Santhan Kumar (212223240065)
# Thiyagarajan A (212222240110)

# --- Import Packages ---
import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from typing import List
import tempfile
import os
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_WEIGHTS_PATH = "Model/40th-epoch-model-checkpoint-keras-default-v1/checkpoint.weights.h5"
TEST_DATASET_FOLDER = "test_video_dataset"

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789'?! "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&family=Montserrat:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

body {
    background-color: #0f1117;
    color: white;
}

h1 {
    color: white;
    text-align: center;
    font-weight: 700;
    font-size: 42px;
    animation: fadeIn 2s ease-in-out;
}

h2 {
    color: white;
    font-weight: 500;
    font-size: 30px;
    animation: fadeIn 2.5s ease-in-out;
}

h3 {
    color: white;
    text-align: center;
    font-weight: 500;
    font-size: 28px;
    animation: fadeIn 2.5s ease-in-out;
}

p, label, .stRadio > div {
    color: white !important;
    font-size: 20px;
    animation: fadeIn 3s ease-in-out;
}

section[data-testid="stFileUploader"] div {
    color: white !important;
    font-size: 20px;
}

.st-emotion-cache-1cypcdb {
    background-color: #1e1f26 !important;
    border-radius: 10px;
    padding: 20px;
}

.stButton > button {
    background-color: #007bff;
    color: white;
    border-radius: 10px;
    padding: 14px 28px;
    font-size: 20px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    background-color: #0056b3;
    box-shadow: 0 0 10px rgba(0, 123, 255, 0.5);
}

.stButton > button::after {
    content: '|';
    position: absolute;
    right: 10px;
    font-size: 20px;
    animation: blinkCursor 1s step-end infinite;
    transition: transform 0.5s ease;
}

.stButton > button:hover::after {
    transform: translateX(-10px);
}

.stImage > img {
    border-radius: 10px;
    animation: fadeIn 2s ease-in-out;
    margin-top: 20px; /* Shift GIF downward to align with text */
}

@media screen and (max-width: 768px) {
    h1 { font-size: 32px; }
    h2 { font-size: 24px; }
    h3 { font-size: 22px; }
    p, label, .stButton > button { font-size: 18px; }
    .stImage > img { margin-top: 10px; } /* Smaller adjustment for mobile */
}

/* Contact section styles */
.contact-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 20px;
    margin-top: 20px;
}

.contact-box {
    background: linear-gradient(145deg, #1e1f26, #2a2b33);
    border: 2px solid transparent;
    border-image: linear-gradient(to right, #007bff, #00d4ff) 1;
    border-radius: 15px;
    padding: 20px;
    width: 280px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 123, 255, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    font-family: 'Montserrat', sans-serif;
}

.contact-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 20px rgba(0, 123, 255, 0.4);
}

.contact-box h4 {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
    color: white;
}

.contact-box p {
    font-size: 16px;
    margin: 5px 0;
}

.contact-link {
    color: #00d4ff;
    text-decoration: none;
    transition: color 0.3s ease;
}

.contact-link:hover {
    color: #007bff;
    text-decoration: underline;
}

@media screen and (max-width: 600px) {
    .contact-box {
        width: 100%;
        max-width: 300px;
    }
}

/* Fade-in animation for text */
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Blinking and sliding cursor animation */
@keyframes blinkCursor {
    0% { opacity: 1; }
    50% { opacity: 0; }
    100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def load_video(path: str, target_frames: int = 75) -> tf.Tensor:
    cap = cv2.VideoCapture(path)
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

def predict_video(video_path: str):
    model = build_model()
    model.load_weights(MODEL_WEIGHTS_PATH)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=CTCLoss)
    video_input = load_video(video_path)
    yhat = model.predict(video_input)
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    predicted_text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')
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
st.markdown("<h1>📹 MouthMap: Lip Reading App</h1>", unsafe_allow_html=True)
# --- UI Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.image("Img-Source/Lip Movement.gif", caption="Lip Reading Icon", use_container_width=True)

with col2:
    st.markdown("""
    <p>
        <b>MouthMap</b> is a deep learning-powered lip reading application that converts silent video inputs into spoken sentences. It utilizes 3D Convolutional Neural Networks and BiLSTMs for sequence modeling and is trained using the CTC loss function.
    </p>
    """, unsafe_allow_html=True)

st.markdown("<h3>Upload a video or select from the test dataset to predict spoken sentences</h3>", unsafe_allow_html=True)

with st.container():
    mode = st.radio("Select Mode", ["Upload Video", "Use Test Dataset"], horizontal=True)

# --- Sidebar with About Us and Project Info ---
st.sidebar.markdown("<h2>About Us</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
<p>
    We are a team of passionate students from Saveetha Engineering College, dedicated to advancing AI-driven solutions. Our expertise lies in deep learning, computer vision, and innovative application development.
</p>
<ul>
    <li><b>Kavinraja D</b>: Specializes in neural network architectures.</li>
    <li><b>Karnala Santhan Kumar</b>: Expert in model optimization and deployment.</li>
    <li><b>Thiyagarajan A</b>: Skilled in Preprocessing the data and Data Science</li>
</ul>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h2>About the Project</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
<p>
    <b>MouthMap</b> is a cutting-edge lip-reading application designed to transcribe spoken words from silent videos. It leverages advanced deep learning techniques, including 3D CNNs and BiLSTMs, to achieve high accuracy. The project aims to assist in accessibility, communication, and real-time transcription scenarios.
</p>
<p>
    <b>Key Features:</b>
    <ul>
        <li>Processes videos in .mp4, .webm, and .mpg formats.</li>
        <li>Supports both uploaded videos and a test dataset.</li>
        <li>Utilizes CTC loss for robust sequence modeling.</li>
    </ul>
</p>
""", unsafe_allow_html=True)

if mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video (.mp4, .webm, .mpg)", type=["mp4", "webm", "mpg"])
    if uploaded_video:
        st.video(uploaded_video)
        st.markdown(f"<p><strong>Selected Video:</strong> {uploaded_video.name}</p>", unsafe_allow_html=True)
        if st.button("🚀 Predict Sentence"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Predicting..."):
                    result = predict_video(tmp_path)
                st.success(f"Predicted Sentence: {result}")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)

elif mode == "Use Test Dataset":
    test_files = [f for f in os.listdir(TEST_DATASET_FOLDER) if f.endswith((".mp4", ".webm", ".mpg"))]
    selected_file = st.selectbox("Select a test video", test_files)
    if selected_file:
        video_path = os.path.join(TEST_DATASET_FOLDER, selected_file)
        align_path = os.path.splitext(video_path)[0] + ".align"
        st.video(video_path)
        st.markdown(f"<p><strong>Selected Video:</strong> {selected_file}</p>", unsafe_allow_html=True)
        if st.button("🚀 Predict & Compare"):
            try:
                with st.spinner("Predicting..."):
                    pred_sentence = predict_video(video_path)
                    actual_sentence = parse_alignment(align_path)
                st.success("✅ Prediction Complete")
                st.markdown(f"**Predicted Sentence:** {pred_sentence}")
                st.markdown(f"**Actual Sentence:** {actual_sentence}")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Footer with Contact Information ---
st.markdown("""
<hr>
<div style='text-align: center;'>
    <h2>Contact Us</h2>
    <div class="contact-container">
        <div class="contact-box">
            <h4>Kavinraja D</h4>
            <p>LinkedIn: <a href="https://www.linkedin.com/in/kavinraja" class="contact-link">kavinraja</a></p>
            <p>GitHub: <a href="https://github.com/kavinraja" class="contact-link">kavinraja</a></p>
        </div>
        <div class="contact-box">
            <h4>Karnala Santhan Kumar</h4>
            <p>LinkedIn: <a href="https://www.linkedin.com/in/karnalasanthankumar" class="contact-link">karnalasanthankumar</a></p>
            <p>GitHub: <a href="https://github.com/karnalasanthankumar" class="contact-link">karnalasanthankumar</a></p>
        </div>
        <div class="contact-box">
            <h4>Thiyagarajan A</h4>
            <p>LinkedIn: <a href="https://www.linkedin.com/in/thiyagarajana" class="contact-link">thiyagarajana</a></p>
            <p>GitHub: <a href="https://github.com/thiyagarajana" class="contact-link">thiyagarajana</a></p>
        </div>
    </div>
    <p style='font-size: 18px; color: gray; margin-top: 20px;'>
        Developed by <b>Kavinraja D</b>, <b>Karnala Santhan Kumar</b>, and <b>Thiyagarajan A</b> • Saveetha Engineering College • 2025
    </p>
</div>
""", unsafe_allow_html=True)