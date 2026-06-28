import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "TrafficSign_Best_Model.keras"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "Label_Mapping.csv"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

# ---------------------------------------------------
# Load Labels
# ---------------------------------------------------

@st.cache_data
def load_labels():

    return pd.read_csv(LABEL_PATH)

try:

    model = load_model()
    labels = load_labels()

except Exception as e:

    st.error(f"❌ Error loading model:\n\n{e}")
    st.stop()

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("🚦 Traffic Sign Recognition")
st.write("Upload a traffic sign image to predict its class.")

# ---------------------------------------------------
# Upload Image
# ---------------------------------------------------

uploaded = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    img = np.array(image)

    img = cv2.resize(
        img,
        (64, 64)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    with st.spinner("Predicting..."):

        prediction = model.predict(
            img,
            verbose=0
        )

    class_id = int(np.argmax(prediction))

    confidence = float(np.max(prediction))

    # ---------------------------------------------------
    # Read Label Name
    # ---------------------------------------------------

    if "Label" in labels.columns:

        class_name = labels.loc[
            labels["ClassId"] == class_id,
            "Label"
        ].values[0]

    elif "SignName" in labels.columns:

        class_name = labels.loc[
            labels["ClassId"] == class_id,
            "SignName"
        ].values[0]

    else:

        class_name = f"Class {class_id}"

    st.success(f"Prediction : {class_name}")

    st.info(f"Confidence : {confidence*100:.2f}%")

    # ---------------------------------------------------
    # Top 5 Predictions
    # ---------------------------------------------------

    st.subheader("Top 5 Predictions")

    top5 = np.argsort(prediction[0])[::-1][:5]

    for idx in top5:

        if "Label" in labels.columns:

            name = labels.loc[
                labels["ClassId"] == idx,
                "Label"
            ].values[0]

        elif "SignName" in labels.columns:

            name = labels.loc[
                labels["ClassId"] == idx,
                "SignName"
            ].values[0]

        else:

            name = f"Class {idx}"

        st.write(
            f"**{name}** : {prediction[0][idx]*100:.2f}%"
        )

        st.progress(float(prediction[0][idx]))