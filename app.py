

# --- Now import packages ---
import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from typing import List
import tempfile
import logging


# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_WEIGHTS_PATH = "Model/40th-epoch-model-checkpoint-keras-default-v1/checkpoint.weights.h5"
TEST_DATASET_FOLDER = "test_video_dataset"  # contains videos and .txt alignments

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789'?! "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

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

# --- UI ---
st.markdown("<h1 style='text-align:center;'>📹 Lip Reading App</h1>", unsafe_allow_html=True)

mode = st.radio("Select Mode", ["Upload Video", "Use Test Dataset"])

target_frames = st.sidebar.slider("Target Frames", 50, 100, 75)

if mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video (.mp4, .webm, .mpg)", type=["mp4", "webm", "mpg"])
    if uploaded_video:
        st.video(uploaded_video)

        if st.button("🚀 Predict Sentence"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name
            try:
                st.info("Predicting...")
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

        if st.button("🚀 Predict & Compare"):
            try:
                pred_sentence = predict_video(video_path)
                actual_sentence = parse_alignment(align_path)
                st.success("✅ Prediction Complete")
                st.markdown(f"**Predicted Sentence:** {pred_sentence}")
                st.markdown(f"**Actual Sentence:** {actual_sentence}")
            except Exception as e:
                st.error(f"Error: {e}")
