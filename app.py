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
TEST_DATASET_FOLDER = "test_video_dataset"  # contains videos and .txt alignments

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789'?! "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# --- Custom CSS for styling ---
st.markdown("""
<style>
    body {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
    }
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        font-size: 20px;
    }

    h1, h2, h3 {
        font-size: 36px;
        color: #2c3e50;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s;
        font-size: 16px;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    .video-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        background: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .result-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    }

    .stVideo>video {
        width: 100%;
    }

    .sidebar .sidebar-content {
        background-color: #e8ecef;
        border-radius: 10px;
        padding: 10px;
    }

    /* Dark theme */
    .dark .main {
        background-color: #2c2c2c;
        color: #ffffff;
    }

    .dark h1, .dark h2, .dark h3 {
        color: #ffffff;
    }

    .dark .stButton>button {
        background-color: #555;
    }

    /* Custom theme */
    .custom .main {
        background: linear-gradient(to right, #a1c4fd, #c2e9fb);
    }

</style>
""", unsafe_allow_html=True)

# --- Theme Selector ---
theme = st.sidebar.selectbox("🎨 Select Theme", ["Light", "Dark", "Custom"])

if theme == "Dark":
    st.markdown("<style>body { background-color: #1e1e1e; color: #ffffff; } .main { background-color: #2c2c2c; color: #ffffff; }</style>", unsafe_allow_html=True)
    st.markdown("<script>document.body.classList.add('dark');</script>", unsafe_allow_html=True)
elif theme == "Custom":
    st.markdown("<style>body { background: linear-gradient(to right, #a1c4fd, #c2e9fb); } .main { background-color: #ffffff; }</style>", unsafe_allow_html=True)
    st.markdown("<script>document.body.classList.add('custom');</script>", unsafe_allow_html=True)

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

# --- UI ---
st.markdown("<h1 style='text-align:center;'>📹 Lip Reading App</h1>", unsafe_allow_html=True)


# Sidebar controls
st.sidebar.header("⚙️ Settings")
target_frames = st.sidebar.slider("🎞️ Target Frames", 50, 100, 75, help="Number of frames to process")
st.sidebar.markdown("---")
st.sidebar.markdown("ℹ️ Adjust settings for optimal performance")

mode = st.radio("🛠️ Select Mode", ["📤 Upload Video", "📂 Use Test Dataset"], horizontal=True)

if mode == "📤 Upload Video":
    uploaded_video = st.file_uploader("📽️ Upload a video (.mp4, .webm, .mpg)", type=["mp4", "webm", "mpg"])

    if uploaded_video:
        try:
            # Create a temp directory for conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_video_path = save_and_convert_video(uploaded_video, temp_dir)
                st.markdown("<div class='video-container'>", unsafe_allow_html=True)
                st.video(temp_video_path)
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"⚠️ Video preview failed. Ensure the file is a valid video format (.mp4, .webm, .mpg). Error: {e}")
            logger.error(f"Video preview error: {e}")

        if st.button("🚀 Predict Sentence", key="predict_upload"):
            with st.spinner("🔄 Processing video..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    tmp_path = tmp.name

                try:
                    result = predict_video(tmp_path)
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.success(f"🎉 Predicted Sentence: {result}")
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
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
                        temp_video_path = save_and_convert_video(open(video_path, 'rb'), temp_dir)
                        st.markdown("<div class='video-container'>", unsafe_allow_html=True)
                        st.video(temp_video_path)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='video-container'>", unsafe_allow_html=True)
                    st.video(video_path)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"⚠️ Video preview failed. Ensure the file is a valid video format. Error: {e}")
                logger.error(f"Video preview error: {e}")

            if st.button("🚀 Predict & Compare", key="predict_test"):
                with st.spinner("🔄 Processing video..."):
                    try:
                        pred_sentence = predict_video(video_path)
                        actual_sentence = parse_alignment(align_path)

                        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                        st.success("✅ Prediction Complete")
                        st.markdown(f"**Predicted Sentence:** {pred_sentence}")
                        st.markdown(f"**Actual Sentence:** {actual_sentence}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with MouthMap-TEAM | © 2025</p>", unsafe_allow_html=True)
