# This define logic (greet method) and UI in the same file, probably is better in separated files for complex UI

# LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC 
# Define a simple function that will be the core of your Gradio app
def extract_faces(video_input):

    return []
import cv2
import dlib
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLOv8 model (you can switch to YOLOv9 when available)
model = YOLO("yolov8-face.pt")

# Load dlib's face recognition model for identifying faces across frames
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize video and output
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Set up data structure for transcription intervals
face_timeline = defaultdict(list)

def get_face_embedding(face_image):
    """Extracts 128D facial embedding using dlib"""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return None
    shape = sp(gray, faces[0])
    return np.array(face_rec_model.compute_face_descriptor(face_image, shape))

def detect_faces(frame, timestamp):
    """Detect faces using YOLOv8 and extract embeddings"""
    results = model(frame)
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result
        face_img = frame[int(y1):int(y2), int(x1):int(x2)]
        embedding = get_face_embedding(face_img)
        if embedding is not None:
            face_timeline[tuple(embedding)].append((timestamp, (x1, y1, x2, y2)))

def generate_intervals(face_timeline):
    """Convert face timeline to intervals"""
    face_intervals = {}
    for face_id, frames in face_timeline.items():
        intervals = []
        start = None
        last_timestamp = None
        for timestamp, bbox in frames:
            if start is None:
                start = timestamp
            elif last_timestamp is not None and timestamp - last_timestamp > 1:
                intervals.append((start, last_timestamp))
                start = timestamp
            last_timestamp = timestamp
        intervals.append((start, last_timestamp))  # Closing the last interval
        face_intervals[face_id] = intervals
    return face_intervals

# Process video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_count / fps
    detect_faces(frame, timestamp)
    frame_count += 1

cap.release()

# Generate intervals
face_intervals = generate_intervals(face_timeline)

# Print results in transcription format (JSON-like)
for face_id, intervals in face_intervals.items():
    print(f"Face ID {face_id}:")
    for interval in intervals:
        print(f"  Appears from {interval[0]}s to {interval[1]}s")

# Optionally save as JSON or CSV
