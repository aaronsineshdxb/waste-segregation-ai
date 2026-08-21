import json
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="Waste Segregation Assistant", layout="wide")

CATEGORIES = ["recyclable", "compost", "landfill"]
MODEL_PATH = Path("models/best_model.keras")
CLASS_MAP_PATH = Path("models/class_indices.json")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_map():
    if CLASS_MAP_PATH.exists():
        with open(CLASS_MAP_PATH) as f:
            return json.load(f)
    return {cat: i for i, cat in enumerate(CATEGORIES)}


def preprocess_image(image, target_size=(224, 224)):
    img = cv2.resize(image, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(model, image):
    """Return (label, confidence, {label: probability}) using the saved class map."""
    processed = preprocess_image(image)
    pred = model.predict(processed, verbose=0)[0]
    class_map = load_class_map()
    inv_map = {v: k for k, v in class_map.items()}
    idx = int(np.argmax(pred))
    label = inv_map.get(idx, CATEGORIES[idx])
    # Map probabilities to labels by their saved indices, not by position
    prob_by_label = {inv_map.get(i, CATEGORIES[i]): float(p) for i, p in enumerate(pred)}
    return label, float(pred[idx]), prob_by_label


def get_waste_tip(label):
    tips = {
        "recyclable": "Clean and dry recyclable items before disposing. Remove labels if possible.",
        "compost": "Keep food scraps separate from plastics. Compost fruit and vegetable waste.",
        "landfill": "Minimize landfill waste by reusing items and avoiding single-use plastics.",
    }
    return tips.get(label, "Sort waste carefully.")


def render_result(label, confidence, probs):
    st.metric("Prediction", label.title(), f"{confidence:.2%}")
    st.progress(min(confidence, 1.0))
    st.info(get_waste_tip(label))
    st.write("Class probabilities")
    st.bar_chart(probs)


st.title("♻️ Waste Segregation Assistant")
st.caption("Real-time AI waste classification for recyclable, compost, and landfill items")

model = load_model()

if model is None:
    st.error("Model not found. Train the model first by running scripts/train_model.py")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam Classifier")
    run = st.checkbox("Start webcam")
    frame_window = st.empty()
    result_slot = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame from webcam")
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                label, confidence, probs = predict_image(model, frame_rgb)
                frame_window.image(frame_rgb, channels="RGB", use_container_width=True)
                with result_slot.container():
                    st.metric("Prediction", label.title(), f"{confidence:.2%}")
                    st.info(get_waste_tip(label))
            cap.release()
            frame_window.empty()
            result_slot.empty()

with col2:
    st.subheader("Upload an Image")
    uploaded = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not read that image file. Try a different JPG/PNG.")
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Uploaded Image", use_container_width=True)
            label, confidence, probs = predict_image(model, image_rgb)
            st.success(f"Prediction: {label.title()} ({confidence:.2%})")
            render_result(label, confidence, probs)

st.divider()
st.subheader("Project Notes")
st.write(
    "This app is a prototype for class XII capstone work. "
    "For the final version, collect real waste images in school with scripts/collect_data.py "
    "and retrain with scripts/train_model.py."
)
