# age_detection_code.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained age detection model
model_path = "E:\\youtube\\age_model_checkpoint.h5"
age_model = load_model(model_path)

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
        return "10-19"
    elif age_label == 3:
        return "20-27"
    elif age_label == 4:
        return "28-45"
    elif age_label == 5:
        return "46-65"
    else:
        return "66+"

# Function to capture permission and return age prediction
# def get_permission_and_age():
#     # Add code to request permission from the user
#     # Example: permission_result = input("Do you give permission to access the age detection feature? (yes/no): ")
#     permission_result = "yes"  # Replace with actual code
    
#     if permission_result.lower() == 'yes':
#         # Add code to capture frame from webcam and process age detection
#         ret, frame = cap.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
#         for (x, y, w, h) in faces:
#             face_roi = gray[y:y+h, x:x+w]
#             face_input = preprocess_face_for_age_model(face_roi)
#             age_prediction = age_model.predict(face_input.reshape(1, *face_input.shape))
#             predicted_age_range = postprocess_age_prediction(age_prediction)
#             return predicted_age_range
#     else:
#         # Return a value indicating that permission was denied
#         return -1

# # Release the webcam and close the OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
    

# age_detection_code.py

# ...

# Function to capture permission and return age prediction
# Inside get_permission_and_age function in age_detection_code.py

# ...

# Function to capture permission and return age prediction
# def get_permission_and_age():






    
    

#     # Add code to request permission from the user
#     # Example: permission_result = input("Do you give permission to access the age detection feature? (yes/no): ")
#     permission_result = "yes"  # Replace with actual code
    
#     if permission_result.lower() == 'yes':
#         # Add code to capture frame from webcam and process age detection
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("Error: Couldn't read frame from webcam.")
#             return -1  # or handle the error accordingly

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
#         for (x, y, w, h) in faces:
#             face_roi = gray[y:y+h, x:x+w]
#             face_input = preprocess_face_for_age_model(face_roi)
#             age_prediction = age_model.predict(face_input.reshape(1, *face_input.shape))
#             predicted_age_range = postprocess_age_prediction(age_prediction)
#             return predicted_age_range
#     else:
#         # Return a value indicating that permission was denied
#         print("Permission denied!")
#         return -1
    



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
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_input = preprocess_face_for_age_model(face_roi)
            age_prediction = age_model.predict(face_input.reshape(1, *face_input.shape))
            predicted_age_range = postprocess_age_prediction(age_prediction)
            return predicted_age_range
    else:
        # Return a value indicating that permission was denied
        print("Permission denied!")
        return -1
    








# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

