import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load the posture detection model
processor = AutoImageProcessor.from_pretrained("ronka/postureDetection")
model = AutoModelForImageClassification.from_pretrained("ronka/postureDetection")

# Load a pre-trained emotion detection model using OpenCV
emotion_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Real emotion detection model or predefined set of emotions
# Use a better model or library like "deepface" or "fer" to classify facial expressions more accurately.
emotion_classifier = cv2.face.LBPHFaceRecognizer_create()


# Function to preprocess frames for posture detection
def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    return inputs


# Function to predict posture from a video frame
def predict_posture(frame):
    inputs = preprocess_frame(frame)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = probabilities.argmax().item()
    confidence_score = probabilities[0, predicted_class].item()
    labels = model.config.id2label
    predicted_posture = labels[predicted_class]
    flag = "green" if "good" in predicted_posture.lower() else "red"
    return predicted_posture, confidence_score, flag


# Function to detect emotions using OpenCV
def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = emotion_model.detectMultiScale(gray_frame, 1.1, 4)

    if len(faces) == 0:
        return "No Face Detected", 0.0

    # Assume the first face is the target for emotion detection
    x, y, w, h = faces[0]
    face_roi = gray_frame[y:y + h, x:x + w]

    # For simplicity, use a mock emotion detection here
    # Replace with a real emotion detection model for more accurate results
    emotion = classify_emotion(face_roi)

    return emotion, 0.85  # Placeholder confidence


# Mock emotion classifier (Replace with a deep learning model or better approach)
def classify_emotion(face_roi):
    # Simple mock emotion classifier, replace with actual deep learning model or rule-based classifier
    return "Happy"  # Placeholder emotion


# Real-time detection
def live_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect emotion
        emotion, emotion_confidence = detect_emotion(frame)

        # Predict posture
        predicted_posture, posture_confidence, flag = predict_posture(frame)

        # Display emotion and posture on the frame
        emotion_text = f"Emotion: {emotion} ({emotion_confidence:.2f})"
        posture_text = f"Posture: {predicted_posture} ({posture_confidence:.2f})"

        color = (0, 255, 0) if flag == "green" else (0, 0, 255)
        cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, posture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display the frame
        cv2.imshow("Facial Expression and Posture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_detection()

# # Import necessary libraries
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from PIL import Image
# import torch
# import cv2
# import numpy as np
#
# # Load the processor and model
# processor = AutoImageProcessor.from_pretrained("ronka/postureDetection")
# model = AutoModelForImageClassification.from_pretrained("ronka/postureDetection")
#
#
# # Function to preprocess the frame
# def preprocess_frame(frame):
#     """
#     Preprocess a video frame to make it suitable for the model.
#     :param frame: Input video frame.
#     :return: Preprocessed image tensor.
#     """
#     # Convert the frame to a PIL image
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#     # Preprocess the image
#     inputs = processor(images=image, return_tensors="pt")
#     return inputs
#
#
# # Function to predict posture from a video frame
# def predict_posture(frame):
#     """
#     Predict the posture from a video frame.
#     :param frame: Input video frame.
#     :return: Predicted posture, confidence score, and flag for proper/improper posture.
#     """
#     # Preprocess the frame
#     inputs = preprocess_frame(frame)
#
#     # Perform inference
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     # Get predicted class and confidence score
#     logits = outputs.logits
#     probabilities = torch.nn.functional.softmax(logits, dim=-1)
#     predicted_class = probabilities.argmax().item()
#     confidence_score = probabilities[0, predicted_class].item()
#
#     # Get the class labels
#     labels = model.config.id2label
#     predicted_posture = labels[predicted_class]
#
#     # Determine if posture is correct or not (green = correct, red = incorrect)
#     flag = "green" if "correct" in predicted_posture.lower() else "red"
#
#     return predicted_posture, confidence_score, flag
#
#
# # Real-time posture detection using webcam
# def live_posture_detection():
#     """
#     Use webcam to capture frames and perform live posture detection.
#     """
#     # Open webcam
#     cap = cv2.VideoCapture(0)
#
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
#
#     print("Press 'q' to quit.")
#
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             print("Error: Could not read frame.")
#             break
#
#         # Predict posture for the current frame
#         predicted_posture, confidence, flag = predict_posture(frame)
#
#         # Display results on the frame
#         color = (0, 255, 0) if flag == "green" else (0, 0, 255)
#         text = f"Posture: {predicted_posture} ({confidence:.2f})"
#         cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#
#         # Show the frame
#         cv2.imshow("Live Posture Detection", frame)
#
#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release the capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# # Run the live posture detection
# if __name__ == "__main__":
#     live_posture_detection()
