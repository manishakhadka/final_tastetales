import cv2
import os
import random
import numpy as np

import base64
from io import BytesIO
from PIL import Image

from train import (
    CHECKPOINTS_DIR,
    NUM_AGE_CATEGORIES,
    AGE_CATEGORY_MAP,
    initialize_model,
    parse_filepath,
    load_checkpoint_and_predict
)

# Load the pre-trained age detection model
FCAES_DIR = "faces"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def save_and_get_path(face_image, face_number):
    face_path = FCAES_DIR + "/face_" + str(face_number) + ".jpg"
    cv2.imwrite(face_path, face_image)
    return face_path

def capture_frames():
    global cap
    cap = cv2.VideoCapture()
    while True:
        ret, frame = cap.read()
    _, jpeg = cv2.imencode('.jpg', frame)
    frame = jpeg.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def preprocess_face_for_age_model(face_image):
    # Resize the face image to match the expected input shape of the model
    face_image = cv2.resize(face_image, (200, 200))
    face_image = face_image / 255.0  # Normalize pixel values between 0 and 1
    # Add an additional dimension to represent the single channel (grayscale)
    face_image = np.expand_dims(face_image, axis=-1)
    return face_image


def stop_camera():
    cap.release()
    cv2.destroyAllWindows()
    return "Camera stopped"




def preprocess_face_for_age_model(face_image):
    # Resize the face image to match the expected input shape of the model
    face_image = cv2.resize(face_image, (200, 200))
    face_image = face_image / 255.0  # Normalize pixel values between 0 and 1
    # Add an additional dimension to represent the single channel (grayscale)
    face_image = np.expand_dims(face_image, axis=-1)
    return face_image


def process_image_and_predict_age(image_data_base64):
    age_model = initialize_model()

    # Decode the base64 image data to an image
    image_data = base64.b64decode(image_data_base64.split(',')[1])
    image = Image.open(BytesIO(image_data))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected.")
        return -1  # Or handle no face detected case

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_input = preprocess_face_for_age_model(face_roi)
        face_number = random.randint(1, 1000)
        face_path = save_and_get_path(face_roi, face_number)
        print("Saved face image to:", face_path)
        y_pred = load_checkpoint_and_predict(
            age_model, CHECKPOINTS_DIR, face_path)
        y_pred_age, y_pred_category = y_pred['age_output'], y_pred['age_category_output']
        predicted_age_int = int(y_pred_age[0][0])

        # Return the first detected face's predicted age for simplicity
        return predicted_age_int

    # In a real scenario, handle cases for multiple faces, etc.
    return -1  # Or appropriate error handling
