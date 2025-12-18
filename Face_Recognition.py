import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page config (polish first)
# -----------------------------
st.set_page_config(
    page_title="Face Recognition System",
    layout="centered"
)

# -----------------------------
# Glass-style CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.glass {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
h1, h3, p {
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants
# -----------------------------
IMG_SIZE = 224
CLASS_NAMES = ["Person_1", "Person_2", "Person_3"]  

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Face-Recognition.h5")

model = load_model()

# -----------------------------
# UI Header
# -----------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<h1>Face Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<p>Deep learning–based identity classification</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Image Upload
# -----------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction button
# -----------------------------
predict_btn = st.button("Identify Person")

if uploaded_file and predict_btn:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.markdown("### Prediction Result")
    st.success(f"**Identity:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Confidence breakdown
    st.markdown("### Confidence per Class")
    for name, score in zip(CLASS_NAMES, preds):
        st.progress(float(score))
        st.write(f"{name}: {score * 100:.2f}%")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<p style="color:white; text-align:center; opacity:0.7;">
Built with TensorFlow & Streamlit
</p>
""", unsafe_allow_html=True)
