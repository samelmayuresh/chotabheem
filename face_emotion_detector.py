import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace

def analyze_face(image):
    """
    Analyze the face in the given image and detect emotions using DeepFace.
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
    Draw the detected emotion on the face in the image with enhanced visual feedback.
    """
    try:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        if result:
            # Define colors for different faces
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            
            for i, face in enumerate(result):
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                emotion = face['dominant_emotion']
                
                # Use different colors for multiple faces
                color = colors[i % len(colors)]
                
                # Draw a thicker rectangle around the face
                cv2.rectangle(opencv_image, (x, y), (x + w, y + h), color, 3)

                # Put the emotion text above the rectangle with better formatting
                cv2.putText(opencv_image, f"{emotion.title()}", (x, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                # Add face number for multiple faces
                if len(result) > 1:
                    cv2.putText(opencv_image, f"Face {i+1}", (x, y + h + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return opencv_image
    except Exception as e:
        st.error(f"Error drawing on image: {e}")
        return None

def format_emotion_results(result):
    """
    Format emotion analysis results for better display.
    """
    if not result:
        return None
    
    formatted_results = []
    for i, face in enumerate(result):
        face_data = {
            'face_id': i + 1,
            'dominant_emotion': face['dominant_emotion'].title(),
            'region': {
                'x': int(face['region']['x']),
                'y': int(face['region']['y']),
                'w': int(face['region']['w']),
                'h': int(face['region']['h'])
            },
            'emotion_scores': {}
        }
        
        # Format emotion scores as percentages
        for emotion, score in face['emotion'].items():
            face_data['emotion_scores'][emotion.title()] = f"{score:.1%}"
        
        formatted_results.append(face_data)
    
    return formatted_results
