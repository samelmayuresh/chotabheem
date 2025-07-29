import cv2
import numpy as np
from deepface import DeepFace
import streamlit as st

def analyze_face(image):
    """
    Analyze the face in the given image and detect emotions.
    """
    try:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Analyze the image using DeepFace
        result = DeepFace.analyze(opencv_image, actions=['emotion'], enforce_detection=False)

        return result
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return None

def draw_emotion_on_face(image, result):
    """
    Draw the detected emotion on the face in the image.
    """
    try:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        for face in result:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            emotion = face['dominant_emotion']

            # Draw a rectangle around the face
            cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put the emotion text above the rectangle
            cv2.putText(opencv_image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return opencv_image
    except Exception as e:
        st.error(f"Error drawing on image: {e}")
        return None
