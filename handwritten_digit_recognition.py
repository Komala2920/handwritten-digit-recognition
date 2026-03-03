# handwritten_digit_recognition.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------- Title --------------------
st.title("Handwritten Digit Recognition")

# -------------------- Load Pre-trained Model --------------------
MODEL_PATH = "digit_model.h5"

try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Pre-trained model not found. Train locally and upload 'digit_model.h5'.")
    st.stop()  # Stop app if model not found

# -------------------- Upload and Predict --------------------
uploaded_file = st.file_uploader("Upload a digit image (PNG or JPG)", type=['png','jpg'])
if uploaded_file is not None:
    # Convert image to grayscale and resize to 28x28
    img = Image.open(uploaded_file).convert('L').resize((28,28))
    st.image(img, caption='Uploaded Digit', use_column_width=True)
    
    # Prepare image for prediction
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,28,28,1)
    
    # Predict digit
    prediction = np.argmax(model.predict(img_array))
    st.write("**Predicted Digit:**", prediction)

#     pred = np.argmax(model.predict(x_test[idx].reshape(1,28,28,1)))
#     st.write("Predicted Digit:", pred)
