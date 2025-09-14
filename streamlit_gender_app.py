"""
Streamlit Gender Classification App
File: streamlit_gender_app.py

What this does:
- Downloads face-detection and gender-classification models (Caffe) if missing
- Lets user upload an image or use webcam (st.camera_input)
- Detects faces, runs gender classifier and shows bounding boxes + labels + confidence

Notes:
- Model files are downloaded from public GitHub repositories (learnopencv / opencv). Internet is required the first time.
- If you already have faster/specialized models, point the MODEL_* paths to your files.

Run:
    pip install -r requirements.txt
    streamlit run streamlit_gender_app.py

Requirements (example):
    streamlit
    opencv-python
    numpy
    pillow

This is a demo. Please ensure you use the app ethically and comply with privacy laws when classifying people from images.

"""

import streamlit as st
import os
from pathlib import Path
import urllib.request
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

st.set_page_config(page_title="Gender Classifier", layout="centered")
st.title("Male / Female Image Classification — Demo")
st.markdown(
    """
    Upload a face image or use your webcam. This demo uses OpenCV DNN face detector and a small gender classifier (Caffe models).

    **Important:** This is a demonstration only. Use responsibly and respect privacy and consent.
    """
)

# Model files and URLs
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Face detector (OpenCV SSD)
FACE_PROTO = MODEL_DIR / "deploy.prototxt"
FACE_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/master/res10_300x300_ssd_iter_140000.caffemodel"

# Gender model (from LearnOpenCV repo)
GENDER_PROTO = MODEL_DIR / "age_gender_deploy.prototxt"
GENDER_MODEL = MODEL_DIR / "gender_net.caffemodel"
GENDER_PROTO_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_gender_deploy.prototxt"
GENDER_MODEL_URL = "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel"

LABELS = ["Male", "Female"]

# Helper: download if missing
@st.cache_data(show_spinner=False)
def download_if_missing(path: Path, url: str):
    if path.exists():
        return str(path)
    try:
        st.info(f"Downloading {path.name} ...")
        urllib.request.urlretrieve(url, str(path))
        st.success(f"Downloaded {path.name}")
    except Exception as e:
        st.error(f"Failed to download {path.name}: {e}")
        raise
    return str(path)

# Ensure models exist
try:
    download_if_missing(FACE_PROTO, FACE_PROTO_URL)
    download_if_missing(FACE_MODEL, FACE_MODEL_URL)
    download_if_missing(GENDER_PROTO, GENDER_PROTO_URL)
    download_if_missing(GENDER_MODEL, GENDER_MODEL_URL)
except Exception:
    st.warning("Model download failed. If you have the model files locally, place them in a `models/` folder and refresh.")

# Load nets
@st.cache_resource
def load_nets():
    face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_MODEL))
    gender_net = cv2.dnn.readNetFromCaffe(str(GENDER_PROTO), str(GENDER_MODEL))
    return face_net, gender_net

try:
    face_net, gender_net = load_nets()
except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

# Image utilities
def read_image_from_bytes(file_bytes) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def detect_faces(net, image: np.ndarray, conf_threshold=0.6):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # clamp
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            faces.append(((startX, startY, endX, endY), float(confidence)))
    return faces


def predict_gender(net, face_img: np.ndarray):
    # face_img: BGR crop
    blob = cv2.dnn.blobFromImage(cv2.resize(face_img, (227, 227)), 1.0,
                                 (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    net.setInput(blob)
    preds = net.forward()
    # preds is shape (1,2)
    if preds.shape[1] == 2:
        idx = int(np.argmax(preds[0]))
        conf = float(preds[0][idx])
        label = LABELS[idx]
    else:
        # fallback
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = LABELS[idx] if idx < len(LABELS) else str(idx)
    return label, conf


def annotate_image(pil_img: Image.Image, detections):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for (box, conf, label) in detections:
        (startX, startY, endX, endY) = box
        draw.rectangle([startX, startY, endX, endY], outline=(255, 0, 0), width=3)
        text = f"{label}: {conf*100:.1f}%"
        text_w, text_h = draw.textsize(text, font=font)
        draw.rectangle([startX, startY - text_h - 4, startX + text_w + 4, startY], fill=(255, 0, 0))
        draw.text((startX + 2, startY - text_h - 2), text, fill=(255, 255, 255), font=font)
    return pil_img

# Sidebar options
st.sidebar.header("Options")
conf_thresh = st.sidebar.slider("Face detection confidence threshold", 0.1, 0.95, 0.6, 0.05)
use_camera = st.sidebar.checkbox("Use webcam (camera input)", value=False)

# Input
uploaded_file = None
if use_camera:
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        uploaded_file = camera_image
else:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"] )

if not uploaded_file:
    st.info("Upload an image or enable webcam in the sidebar to start.")
    st.stop()

# Process image
try:
    img_bgr = read_image_from_bytes(uploaded_file)
    orig = img_bgr.copy()
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

faces = detect_faces(face_net, img_bgr, conf_threshold=conf_thresh)
if len(faces) == 0:
    st.warning("No faces detected. Try a clearer photo or lower the confidence threshold.")

results = []
for ((startX, startY, endX, endY), fconf) in faces:
    face_crop = img_bgr[startY:endY, startX:endX]
    if face_crop.size == 0:
        continue
    label, gconf = predict_gender(gender_net, face_crop)
    results.append(((startX, startY, endX, endY), gconf, label))

# Convert for display
img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
pil = Image.fromarray(img_rgb)
if results:
    pil = annotate_image(pil, results)

st.image(pil, use_column_width=True)

# Show detections in a table
if results:
    st.subheader("Detections")
    for i, ((sx, sy, ex, ey), conf, label) in enumerate(results, start=1):
        st.markdown(f"**Face {i}** — {label} ({conf*100:.1f}% confidence) — bbox: [{sx}, {sy}, {ex}, {ey}]")

st.caption("Model sources: OpenCV face SSD and a small gender net. This demo is not perfect — accuracy varies with pose, lighting, occlusion, and dataset bias.")

# Footer
st.write("\n")
st.info("If you want to use your own trained model (TensorFlow/PyTorch), upload it into the `models/` folder and modify the model-loading code accordingly.")
