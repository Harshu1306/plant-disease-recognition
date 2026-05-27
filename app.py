import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Title
st.title("Plant Disease Recognition")

st.write("Upload a leaf image to predict disease")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prediction
    results = model.predict(image)

    names = model.names

    pred_class = results[0].probs.top1
    confidence = results[0].probs.top1conf.item()

    disease = names[pred_class]

    # Output
    st.success(f"Predicted Disease: {disease}")

    st.info(f"Confidence: {confidence:.2f}")