# handwritten_digit_recognition.py

import streamlit as st
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os

# -------------------- Title --------------------
st.title("Handwritten Digit Recognition")

# -------------------- Load Dataset --------------------
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1,28,28,1)/255.0
    x_test = x_test.reshape(-1,28,28,1)/255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

# -------------------- Build or Load Model --------------------
MODEL_PATH = "digit_model.h5"

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = build_model()
    with st.spinner("Training model... This may take a few minutes"):
        model.fit(x_train, y_train, epochs=5, validation_split=0.2)
        model.save(MODEL_PATH)
    st.success("Model trained and saved!")

# -------------------- Evaluate Model --------------------
loss, acc = model.evaluate(x_test, y_test, verbose=0)
st.write(f"**Test Accuracy:** {acc*100:.2f}%")

# -------------------- Upload and Predict --------------------
uploaded_file = st.file_uploader("Upload a digit image (PNG or JPG)", type=['png','jpg'])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L').resize((28,28))
    st.image(img, caption='Uploaded Digit', use_column_width=True)
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,28,28,1)
    prediction = np.argmax(model.predict(img_array))
    st.write("**Predicted Digit:**", prediction)

# -------------------- Test with Random Dataset Image --------------------
if st.button("Test with Random MNIST Image"):
    idx = np.random.randint(0, x_test.shape[0])
    st.image(x_test[idx].reshape(28,28), caption="Random Test Image", width=100)
    pred = np.argmax(model.predict(x_test[idx].reshape(1,28,28,1)))
    st.write("Predicted Digit:", pred)
