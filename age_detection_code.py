# age_detection_code.py

import cv2
import random
import numpy as np
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

# custom cnn code
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

age_model = initialize_model()

# Open the webcam
cap = cv2.VideoCapture(0)

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
        return "1-2"
    elif age_label == 1:
        return "3-9"
    elif age_label == 2:
        return "10-17"
    elif age_label == 3:
        return "18-27"
    elif age_label == 4:
        return "28-45"
    elif age_label == 5:
        return "46-65"
    else:
        return "66-100"

def save_and_get_path(face_image, face_number):
    face_path = FCAES_DIR + "/face_" + str(face_number) + ".jpg"
    cv2.imwrite(face_path, face_image)
    return face_path


def get_permission_and_age():
    print("Attempting to access the webcam...")
    # Check if the webcam is accessible
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return -1  # or handle the error accordingly
    
    # Add code to request permission from the user
    # Example: permission_result = input("Do you give permission to access the age detection feature? (yes/no): ")
    permission_result = "yes"  # Replace with actual code
    
    if permission_result.lower() == 'yes':
        # Capture frame from webcam and process age detection
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Couldn't read frame from webcam.")
            return -1  # or handle the error accordingly

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        print("faces", faces)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_input = preprocess_face_for_age_model(face_roi)
            face_number = random.randint(1, 1000)
            face_path = save_and_get_path(face_roi, face_number)
            # age_prediction = age_model.predict(face_input.reshape(1, *face_input.shape))
            print("face_path", face_path)
            y_pred = load_checkpoint_and_predict(age_model, CHECKPOINTS_DIR, face_path)
            y_pred_age, y_pred_category = y_pred['age_output'], y_pred['age_category_output']
            predicted_age_int = int(y_pred_age[0][0])
            cap.release()
            return predicted_age_int

            # predicted_age_range = postprocess_age_prediction(age_prediction)
            # return predicted_age_range
    else:
        # Return a value indicating that permission was denied
        print("Permission denied!")
        return -1


if __name__ == "__main__":
    age = get_permission_and_age()
    if age != -1:
        print(f"The predicted age range is: {age}")
    else:
        print("Error: Age detection failed.")
    cap.release()
 
# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

