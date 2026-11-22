import numpy as np
import pandas as pd
import streamlit as st
import joblib
from PIL import Image

class_names = ['child (0-12)', 'teen (13-19)', 'young adult (20-35)', 'adult (36-59)', 'senior (60+)']
IMG_SIZE = 32

model = joblib.load("age_knn.pkl")

st.title("Age Group Prediction (KNN)")
st.write("Take a photo or upload one. Put your face in the middle and close to the camera.")

photo = st.camera_input("Webcam")
upload = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

file = photo if photo is not None else upload

if file is not None:
    img = Image.open(file).convert("RGBA")
    white = Image.new("RGBA", img.size, "white")          # transparent background -> white
    img = Image.alpha_composite(white, img).convert("L")  # gray

    # the training images are close-up faces, so keep the middle square of the photo
    w, h = img.size
    side = int(min(w, h) * 0.7)
    left, top = (w - side) // 2, (h - side) // 2
    face = img.crop((left, top, left + side, top + side)).resize((IMG_SIZE, IMG_SIZE))

    x = np.array(face) / 255
    x = x.reshape(1, -1)                                  # one row of pixels, like the notebook

    probs = model.predict_proba(x)[0]
    pred = class_names[probs.argmax()]

    col1, col2 = st.columns(2)
    col1.image(file, width=250)
    col2.image(face.resize((150, 150)), caption="What the model sees")

    st.subheader(f"Prediction: {pred} ({probs.max() * 100:.1f}%)")
    st.bar_chart(pd.DataFrame({"probability": probs.astype(float)}, index=class_names))
