import numpy as np
import pandas as pd
import streamlit as st
import cv2
import joblib
from PIL import Image

class_names = ['child (0-12)', 'teen (13-19)', 'young adult (20-35)', 'adult (36-59)', 'senior (60+)']
IMG_SIZE = 32

model = joblib.load("age_knn.pkl")
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("Age Group Prediction (KNN)")
st.write("Take a photo or upload one. Keep your face close to the camera.")

photo = st.camera_input("Webcam")
upload = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

file = photo if photo is not None else upload

if file is not None:
    img = Image.open(file).convert("RGBA")
    white = Image.new("RGBA", img.size, "white")          # transparent background -> white
    gray = np.array(Image.alpha_composite(white, img).convert("L"))

    # the training images are close-up faces, so cut the face out first
    faces = face_detector.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        gray = gray[y:y + h, x:x + w]
    else:
        st.warning("No face found, using the whole image.")

    face = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))         # same size as the training data
    x = face / 255
    x = x.reshape(1, -1)                                  # one row of pixels, like the notebook

    probs = model.predict_proba(x)[0]
    pred = class_names[probs.argmax()]

    col1, col2 = st.columns(2)
    col1.image(file, width=250)
    col2.image(face, caption="What the model sees", width=150)

    st.subheader(f"Prediction: {pred} ({probs.max() * 100:.1f}%)")
    st.bar_chart(pd.DataFrame({"probability": probs.astype(float)}, index=class_names))
