import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="FaceID",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Global CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #e5e7eb;
}

p, li {
    color: #cbd5e1;
}

.hero {
    padding: 2.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #1e293b, #020617);
    margin-bottom: 2rem;
}

.glass-card {
    background: rgba(30, 41, 59, 0.65);
    backdrop-filter: blur(8px);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.05);
}

.result-card {
    padding: 1.8rem;
    border-radius: 16px;
    text-align: center;
    background: linear-gradient(135deg, #2563eb, #1e3a8a);
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Constants ----------------
IMG_SIZE = 224
CLASS_NAMES = ["Santhosh", "Swathi", "Vishwa"]

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Face-Recognition.h5")

model = load_model()

# ---------------- Hero Section ----------------
st.markdown("""
<div class="hero">
    <h1>FaceID</h1>
    <h3>Deep Learning–Based Identity Recognition</h3>
    <p>
        FaceID identifies individuals from facial images using a
        convolutional neural network trained on internal team data.
        The system performs supervised classification — not clustering
        or similarity guessing.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Metrics ----------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model Type", "CNN")
m2.metric("Classes", "3")
m3.metric("Input", "Face Image")
m4.metric("Inference", "< 1 sec")

# ---------------- Main Layout ----------------
left, center, right = st.columns([1.3, 1.8, 1.4])

# -------- Left --------
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("About the Model")
    st.write("""
    This system uses a **Convolutional Neural Network (CNN)**
    trained on labeled facial images of team members.

    The model learns facial patterns such as structure, texture,
    and spatial features to perform **identity classification**.

    This is a **supervised learning** system.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -------- Center --------
with center:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Identity Recognition")

    uploaded_file = st.file_uploader(
        "Upload a face image for identification",
        type=["jpg", "jpeg", "png"]
    )

    predict = st.button("Identify Person", use_container_width=True)

    if uploaded_file and predict:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        idx = int(np.argmax(preds))
        identity = CLASS_NAMES[idx]
        confidence = preds[idx] * 100

        st.markdown(
            f"""
            <div class="result-card">
                <h2>{identity}</h2>
                <p>Recognition Confidence: {confidence:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# -------- Right --------
with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Recognized Identities")

    with st.expander("Limitations"):
        st.write("""
        - Works only on trained identities  
        - Sensitive to lighting and image quality  
        - Not a face verification system
        """)

    st.markdown('</div>', unsafe_allow_html=True)
