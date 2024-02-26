# age_detection_code.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained age detection model
model_path = "C:\\project\\board_pachi_herney_wala\\age_model_checkpoint.h5"
age_model = load_model(model_path)

# Define the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the face image for the age model
def preprocess_face_for_age_model(face_image):
    # Resize the face image to match the expected input shape of the model
    face_image = cv2.resize(face_image, (200, 200))
    face_image = face_image / 255.0  # Normalize pixel values between 0 and 1
    # Add an additional dimension to represent the single channel (grayscale)
    face_image = np.expand_dims(face_image, axis=-1)
    return face_image

# Function to postprocess the age prediction
def postprocess_age_prediction(age_prediction):
    # Find the index with the highest probability
    predicted_age_index = np.argmax(age_prediction)
    
    # Convert the index to an age range using your custom function
    predicted_age = class_labels_reassign(predicted_age_index)
    
    return predicted_age

# Function to reassign age labels to ranges
def class_labels_reassign(age_label):
    if age_label == 0:
        return "1-7"
    elif age_label == 1:
        return "7-17"
    elif age_label == 2:
        return "18-25"
    elif age_label == 3:
        return "26-35"
    elif age_label == 4:
        return "35-45"
    elif age_label == 5:
        return "46-65"
    else:
        return "66+"

import base64
def get_permission_and_age(image_data):
    
    

    image_bytes = base64.b64decode(image_data)
    
    # Convert bytes to NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode NumPy array as an image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    


    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face (assuming only one face for simplicity)
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_input = preprocess_face_for_age_model(face_roi)
        age_prediction = age_model.predict(face_input.reshape(1, *face_input.shape))
        predicted_age_range = postprocess_age_prediction(age_prediction)
        return predicted_age_range
        

    # Return a default value if no face is detected
    return "Unknown"
