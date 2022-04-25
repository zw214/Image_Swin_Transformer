import os
import json
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from utils import load_and_prep_image, classes_and_models, predict_json

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "aipi-540-4a9a95f14ee5.json" # change for your GCP key
PROJECT = "aipi-540" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)

### Streamlit code (works as a straigtht-forward script) ###
st.title("Welcome to Mammography Classifier 📸")
st.header("Identify what class it is for this mammography image!")

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.
    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image = load_and_prep_image(image)
    
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=image)
    pred_class = class_names[np.argmax(preds[0])]
    confidence = np.max(preds[0])
    return image, pred_class, confidence

# Pick the model version
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (all mammo images)", # original all images
     "Model 2 (11 food classes)", # original 10 classes + donuts
     "Model 3 (11 food classes + non-food class)") # 11 classes (same as above) + not_food class
)

# Model choice logic
if choose_model == "Model 1 (all mammo images)":
    CLASSES = classes_and_models["model_1"]["classes"]
    MODEL = classes_and_models["model_1"]["model_name"]

# Display info about model and classes
if st.checkbox("Show classes"):
    st.write(f"You chose {MODEL}, these are the classes of images it can identify:\n", CLASSES)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image of mammo image",
                                 type=["png", "jpeg", "jpg"])

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    uploaded_image = uploaded_file.read()
    st.image(uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    pred_button = True 

# And if they did...
if pred_button:
    image, preds, conf = make_prediction(uploaded_image, model=MODEL, class_names=CLASSES)
    st.write(f"Prediction: {preds}, \
               Confidence: {conf:.3f}")