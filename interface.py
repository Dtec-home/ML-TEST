import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import pyttsx3

model = load_model('asl_model1.h5')
engine = pyttsx3.init()
footer = """<style> 
            footer {visibility: hidden;}
            </style>
            """
st.markdown(footer, unsafe_allow_html=True)

st.title('Sign Language made easier🥳🥳')
run = st.checkbox('Start🚦/Stop🚥')
FRAME_WINDOW = st.image([])

while run:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    FRAME_WINDOW.image(frame)

    # Convert the image to RGB and resize it
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (64, 64))

    # Reshape the image to a 4D tensor for model input
    tensor = np.reshape(resized, (1, 64, 64, 3))

    # Predict the gesture
    prediction = model.predict(tensor)

    # Convert the gesture to text and speech
    predicted_class = np.argmax(prediction)
    letter = chr(predicted_class + 65)
    st.write('Recognized Gesture:', letter)
    st.text(letter)

    # Speak the recognized gesture
    engine.say(letter)
    engine.runAndWait()

    cap.release()
