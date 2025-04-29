# import streamlit as st
# import cv2
# import tensorflow as tf
# import numpy as np
# from typing import List
# import os
# import tempfile
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Path to the pre-exported model weights in the backend
# MODEL_WEIGHTS_PATH = "40th-epoch-model-checkpoint-keras-default-v1/checkpoint.weights.h5"

# # Define vocabulary and lookup layers
# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789'?! "]
# char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# def load_video(path: str, target_frames: int = 75) -> tf.Tensor:
#     try:
#         # Validate file existence and size
#         if not os.path.exists(path):
#             raise ValueError(f"Video file {path} does not exist.")
#         if os.path.getsize(path) == 0:
#             raise ValueError(f"Video file {path} is empty.")
        
#         cap = cv2.VideoCapture(path)
#         if not cap.isOpened():
#             raise ValueError(
#                 f"Failed to open video file: {path}. Ensure the file is a valid .mp4, .mpg, or .webm video. "
#                 "Verify that ffmpeg is installed (`ffmpeg -version`) and OpenCV is built with ffmpeg support. "
#                 "Try converting the video with: `ffmpeg -i input.webm -c:v libx264 -c:a aac output.mp4`"
#             )
        
#         frames = []
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if total_frames == 0:
#             raise ValueError(f"Video file {path} appears to be empty or corrupted.")
        
#         frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int) if total_frames > 0 else []
        
#         for i in range(total_frames):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if i in frame_indices or (i == total_frames - 1 and len(frames) < target_frames):
#                 frame = tf.image.rgb_to_grayscale(frame)
#                 frame = frame[190:236, 80:220, :]
#                 frames.append(frame)
        
#         cap.release()

#         while len(frames) < target_frames:
#             frames.append(frames[-1] if frames else tf.zeros((46, 140, 1), dtype=tf.uint8))

#         frames = frames[:target_frames]
        
#         frames = tf.stack(frames)
#         mean = tf.math.reduce_mean(frames)
#         std = tf.math.reduce_std(tf.cast(frames, tf.float32))
#         frames = tf.cast((frames - mean), tf.float32) / std
        
#         frames = tf.expand_dims(frames, axis=0)  # Shape: (1, 75, 46, 140, 1)
#         return frames
#     except Exception as e:
#         logger.error(f"Error in load_video: {e}")
#         raise

# def CTCLoss(y_true, y_pred):
#     batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
#     input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
#     label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
#     input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#     label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#     loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
#     return loss

# def build_model():
#     try:
#         input_shape = (75, 46, 140, 1)
#         inputs = tf.keras.layers.Input(shape=input_shape, name="input")
        
#         x = tf.keras.layers.Conv3D(128, 3, activation=None, padding='same')(inputs)
#         x = tf.keras.layers.Activation('relu')(x)
#         x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(x)
        
#         x = tf.keras.layers.Conv3D(256, 3, activation=None, padding='same')(x)
#         x = tf.keras.layers.Activation('relu')(x)
#         x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(x)
        
#         x = tf.keras.layers.Conv3D(75, 3, activation=None, padding='same')(x)
#         x = tf.keras.layers.Activation('relu')(x)
#         x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(x)
        
#         x = tf.keras.layers.Reshape((75, -1))(x)
        
#         x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
#         x = tf.keras.layers.Dropout(0.5)(x)
#         x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
#         x = tf.keras.layers.Dropout(0.5)(x)
        
#         outputs = tf.keras.layers.Dense(len(char_to_num.get_vocabulary()), activation='softmax')(x)
        
#         model = tf.keras.Model(inputs, outputs)
#         return model
#     except Exception as e:
#         logger.error(f"Error in build_model: {e}")
#         raise

# def clean_prediction(predicted_text: str) -> str:
#     cleaned = predicted_text.replace("!", " ")
#     cleaned = ' '.join(cleaned.split())
#     return cleaned

# def predict_video(video_path: str):
#     try:
#         model = build_model()
#         if not os.path.exists(MODEL_WEIGHTS_PATH):
#             raise FileNotFoundError(f"Model weights file not found: {MODEL_WEIGHTS_PATH}")
#         model.load_weights(MODEL_WEIGHTS_PATH)
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=CTCLoss)

#         video_input = load_video(video_path)
#         yhat = model.predict(video_input)
#         decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
#         predicted_text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')
        
#         return clean_prediction(predicted_text)
#     except Exception as e:
#         logger.error(f"Error in predict_video: {e}")
#         return f"Error: {str(e)}"

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .main {background-color: #f5f5f5;}
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #45a049;
#     }
#     .header {text-align: center; color: #2c3e50; font-size: 2.5em; margin-bottom: 20px;}
#     .subheader {color: #34495e; font-size: 1.2em;}
#     .footer {text-align: center; color: #7f8c8d; margin-top: 50px;}
#     .icon {font-size: 1.5em; margin-right: 10px;}
#     </style>
# """, unsafe_allow_html=True)

# # Streamlit app
# st.markdown('<div class="header">📹 Lip Reading Prediction App</div>', unsafe_allow_html=True)
# st.markdown('<p class="subheader">Upload a video (.mp4, .mpg, or .webm) to predict the spoken sentence using the pre-loaded model.</p>', unsafe_allow_html=True)

# # Sidebar for settings
# st.sidebar.header("⚙️ Settings")
# target_frames = st.sidebar.slider("Target Frames", min_value=50, max_value=100, value=75, help="Number of frames to process from the video.")
# show_video = st.sidebar.checkbox("Show Video Preview", value=True, help="Display the uploaded video in the app.")

# # Check dependencies
# try:
#     import streamlit, tensorflow, cv2, numpy
# except ImportError as e:
#     st.error(f"Missing dependency: {e}. Please install required packages: streamlit, tensorflow, opencv-python, numpy.")
#     st.stop()

# # Check if model weights exist
# if not os.path.exists(MODEL_WEIGHTS_PATH):
#     st.error(f"Model weights file not found at: {MODEL_WEIGHTS_PATH}. Please ensure the file is available in the backend.")
#     st.stop()

# # File uploader for video
# video_file = st.file_uploader("🎥 Upload Video File", type=["mp4", "mpg", "webm"], help="Supported formats: mp4, mpg, webm")

# # Display video file info
# if video_file:
#     st.markdown("### 📋 Video File Info")
#     st.write(f"**File Name:** {video_file.name}")
#     st.write(f"**File Size:** {video_file.size / 1024:.2f} KB")
    
#     # Validate video file
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_video:
#             tmp_video.write(video_file.read())
#             video_path = tmp_video.name
#             if os.path.getsize(video_path) == 0:
#                 raise ValueError("Uploaded video file is empty.")
#     except Exception as e:
#         st.error(f"Failed to process video file: {e}")
#         if os.path.exists(video_path):
#             os.unlink(video_path)
#         st.stop()

#     # Video preview
#     if show_video:
#         st.markdown("### 📺 Video Preview")
#         try:
#             st.video(video_path)
#         except Exception as e:
#             st.warning(f"Could not display video preview: {e}. Prediction may still work.")

# # Prediction
# if video_file:
#     if st.button("🚀 Predict Sentence"):
#         progress = st.progress(0)
#         status_text = st.empty()
        
#         try:
#             status_text.text("Processing video... (1/3)")
#             progress.progress(33)
            
#             status_text.text("Loading model... (2/3)")
#             progress.progress(66)
            
#             status_text.text("Predicting... (3/3)")
#             result = predict_video(video_path)
            
#             progress.progress(100)
#             status_text.text("Done!")
            
#             if result.startswith("Error"):
#                 st.error(result)
#             else:
#                 st.success("🎉 Prediction completed!")
#                 st.markdown(f"**Predicted Sentence:** {result}")
#         except Exception as e:
#             st.error(f"Unexpected error: {e}")
#             logger.error(f"Unexpected error: {e}")
#         finally:
#             # Clean up temporary file
#             if os.path.exists(video_path):
#                 os.unlink(video_path)
# else:
#     st.info("ℹ️ Please upload a video file (.mp4, .mpg, or .webm) to proceed.")

# # Footer
# st.markdown('<div class="footer">Built with ❤️ using Streamlit | © 2025 Lip Reading App</div>', unsafe_allow_html=True)




import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from typing import List
import os
import tempfile
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_WEIGHTS_PATH = "40th-epoch-model-checkpoint-keras-default-v1/checkpoint.weights.h5"
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
