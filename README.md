
# 📹 MouthMap – Lip Reading App

MouthMap is a deep learning-powered lip reading application that processes silent videos and predicts the spoken sentence. It uses a custom-trained 3D CNN + BiLSTM model and provides a simple web interface via Streamlit.

---

## 🚀 Features

- Upload or choose test videos to predict spoken sentences from lip movements.
- Side-by-side comparison with ground-truth alignment files.
- Clean and intuitive Streamlit interface.
- Grayscale video preprocessing and robust deep learning pipeline.
- Real-time video processing with frame normalization and padding.

---

## 📁 Project Structure

```
MouthMap-App/
├── app.py                        # Streamlit application
├── Model/
│   └── 40th-epoch-model-checkpoint-keras-default-v1/
│       └── checkpoint.weights.h5
├── test_video_dataset/          # Test videos and alignment files
├── requirements.txt             # Python dependencies
└── README.md                    # You're here
```

---

## ⚙️ Setup Instructions

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/your-username/MouthMap-App.git
cd MouthMap-App
```

### ✅ 2. System Dependencies (Linux/Debian-based)

```bash
sudo apt update && sudo apt install -y libgl1
```

> `libGL` is required for OpenCV to function properly in headless environments.

### ✅ 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Architecture

- **Input Shape:** `(75 frames, 46 height, 140 width, 1 channel)`
- **Backbone:** 3D Convolutions + BiLSTM
- **Loss Function:** CTC Loss (Connectionist Temporal Classification)
- **Output:** Decoded sentence from lip movements

---

## 🧪 Test Dataset

- Include your test `.mp4` / `.webm` / `.mpg` videos inside `test_video_dataset/`.
- Each video must have a corresponding `.align` file with word timings and alignment, e.g.:

```
test_video_dataset/
├── sample1.mp4
├── sample1.align
```

---

## 🖥️ Run the App

```bash
streamlit run app.py
```

Then open in your browser:
- Local: `http://localhost:8502`
- Network: `http://<your-ip>:8502`

---

## 📤 Upload Mode vs Dataset Mode

- **Upload Mode**: Upload your own video, predict sentence.
- **Use Test Dataset**: Select from sample videos and compare with actual alignment.

---

## 📦 Requirements

```txt
streamlit
tensorflow
numpy
opencv-python-headless
```

Installed automatically via:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Troubleshooting

- **OpenCV ImportError: `libGL.so.1` not found**  
  → Run: `sudo apt install -y libgl1`

- **Streamlit not found**  
  → Run: `pip install streamlit`

- **GPU Acceleration**  
  → For training or inference acceleration, install TensorFlow with GPU support.

---

## 📜 License

MIT License. Feel free to use, fork, and contribute.

---

---

## 🌟 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss.
