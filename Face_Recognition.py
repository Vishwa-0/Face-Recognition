import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Team Face Recognition",
    layout="centered"
)

# -----------------------------
# Glass UI styling
# -----------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.glass {
    background: rgba(255, 255, 255, 0.14);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.title {
    color: white;
    text-align: center;
}
.sub {
    color: #cfd8dc;
    text-align: center;
    font-size: 14px;
}
.metric {
    font-size: 22px;
    color: white;
    font-weight: 600;
    text-align: center;
}
.name {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #00e5ff;
}
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 224

# -----------------------------
# Class names (from training)
# -----------------------------
CLASS_NAMES = ["Santhosh", "Swathi", "Vishwa"]

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Face-Recognition.h5")

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<h1 class='title'>Face Recognition System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub'>CNN-based identity classification for internal team members</p>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Upload section
# -----------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a face image for identification",
    type=["jpg", "jpeg", "png"]
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Inference
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    idx = int(np.argmax(preds))
    identity = CLASS_NAMES[idx]
    confidence = preds[idx] * 100

    # -------------------------
    # Result card
    # -------------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='metric'>Identified Person</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='name'>{identity}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='sub'>Confidence: {confidence:.2f}%</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # Confidence distribution
    # -------------------------
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='metric'>Confidence Distribution</div>", unsafe_allow_html=True)

    for name, score in zip(CLASS_NAMES, preds):
        st.write(name)
        st.progress(float(score))

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer filler (no emptiness)
# -----------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub'>Model trained on labeled facial images using a custom CNN architecture</p>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)
