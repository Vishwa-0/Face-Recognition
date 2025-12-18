import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Face Recognition",
    layout="centered"
)

# -----------------------------
# Minimal premium styling
# -----------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 32px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}
.title {
    text-align: center;
    color: white;
    font-size: 36px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #cfd8dc;
    font-size: 14px;
    margin-bottom: 25px;
}
.result {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    color: #00e5ff;
}
.conf {
    text-align: center;
    color: #cfd8dc;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 224
CLASS_NAMES = ["Santhosh", "Swathi", "Vishwa"]

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Face-Recognition.h5")

model = load_model()

# -----------------------------
# Main card
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>Face Recognition</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>CNN-based identity classification for internal team members</div>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

predict_btn = st.button("Predict Identity")

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file and predict_btn:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    idx = int(np.argmax(preds))

    identity = CLASS_NAMES[idx]
    confidence = preds[idx] * 100

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.15)'>", unsafe_allow_html=True)
    st.markdown(f"<div class='result'>{identity}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='conf'>Confidence: {confidence:.2f}%</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer (quiet, not empty)
# -----------------------------
st.markdown(
    "<p style='text-align:center;color:#90a4ae;font-size:12px;'>"
    "Model trained using a custom CNN on internal facial image data"
    "</p>",
    unsafe_allow_html=True
)
