from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "garbage_classifier.keras"
CLASSES_PATH = PROJECT_DIR / "models" / "class_names.json"

st.set_page_config(page_title="Smart Garbage Classifier", page_icon="♻️")
st.title("♻️ Smart Garbage Classifier")
st.write("Upload a waste image to predict its category. This software prototype uses a trained image-classification model.")

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.warning("The trained model has not been created yet. Run `python train.py` in the project terminal first.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
model = load_model()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    resized = image.resize((224, 224))
    pixels = np.asarray(resized, dtype=np.float32)
    probabilities = model.predict(np.expand_dims(pixels, axis=0), verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    st.subheader(f"Prediction: {class_names[best_index].title()}")
    st.metric("Model confidence", f"{probabilities[best_index]:.1%}")
    st.caption("Confidence is the model's score, not a guarantee that the classification is correct.")
    st.write("Scores by category")
    ranking = np.argsort(probabilities)[::-1]
    for i in ranking:
        st.write(f"**{class_names[int(i)].title()}** — {probabilities[int(i)]:.1%}")
        st.progress(float(probabilities[int(i)]))
