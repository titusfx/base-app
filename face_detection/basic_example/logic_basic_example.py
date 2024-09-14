# This define logic (greet method) and UI in the same file, probably is better in separated files for complex UI

# LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC 
# Define a simple function that will be the core of your Gradio app


import cv2
import dlib
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from collections import defaultdict
from ultralytics import YOLO
import torch
from collections import defaultdict, Counter

# Load the facial embedding model (using dlib)
face_rec_model = dlib.face_recognition_model_v1("/home/titusfx/Projects/AI/gradios/face-detection/face_detection/models/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("/home/titusfx/Projects/AI/gradios/face-detection/face_detection/models/shape_predictor_68_face_landmarks.dat")

# Function to extract face embedding
def get_face_embedding(image, bbox):
    """
    Extract the face embedding vector from the given image and bounding box.

    Args:
        image (np.array): The image containing the face.
        bbox (list): Bounding box [x1, y1, x2, y2] coordinates of the face.

    Returns:
        np.array: A 128-dimensional embedding vector for the face.
    """
    x1, y1, x2, y2 = bbox
    face_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
    shape = shape_predictor(image, face_rect)
    embedding = np.array(face_rec_model.compute_face_descriptor(image, shape))
    return embedding

# Check if the new face is already detected using cosine similarity
# def face_exists(embedding, embeddings, threshold=0.6):
def face_exists(embedding, embeddings, threshold=0.4):
    """
    Check if a face already exists in the detected embeddings using cosine similarity.

    Args:
        embedding (np.array): The embedding vector for the current face.
        embeddings (dict): A dictionary of previously detected face embeddings.
        threshold (float): The threshold for similarity (default is 0.4).

    Returns:
        str or None: Returns the face ID if the face already exists, else returns None.
    """
    for face_id, emb in embeddings.items():
        if np.linalg.norm(embedding - emb) < threshold:
            return face_id
    return None
import os
def detect_faces(video_path, use_gpu=False, interval=1.0):
    """
    Detect faces in a video, extract face embeddings, and save face images at regular intervals.

    Args:
        video_path (str): Path to the video file.
        use_gpu (bool): Whether to use GPU for face detection (default is False).
        interval (float): Time interval (in seconds) between frames to process (default is 1.0).

    Returns:
        tuple: A tuple containing:
            - face_intervals (dict): Dictionary of face IDs and their occurrence intervals.
            - face_embeddings (dict): Dictionary of face IDs and their embedding vectors.
            - face_occurrences (Counter): A counter of how many times each face appears.
    """

    output_dir="detected_faces"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8x.pt').to(device)  # Adjust model path if using YOLOv9
    # model = YOLO('yolov8x.pt').to(device)  # Adjust model path if using YOLOv9
    
    video = VideoFileClip(video_path)
    duration = video.duration
    face_intervals = defaultdict(list)
    face_embeddings = {}
    face_counter = 0
    face_occurrences = Counter()
    # Process the video frame by frame at the given interval
    for time in np.arange(0, duration, interval):
        frame = video.get_frame(time)
        results = model(frame)

        # Extract bounding boxes for faces and check if they are new
        # Loop through detected faces in the frame
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates from YOLO result
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Adjust for bounding box coordinates
                # Extract the face embedding from the bounding box

                embedding = get_face_embedding(frame, [x1, y1, x2, y2])
                face_id = face_exists(embedding, face_embeddings)
                if not face_id:
                    face_id = f"face_{face_counter}"
                    face_embeddings[face_id] = embedding
                    face_counter += 1

                 # Store the face intervals (start and end times)
                face_intervals[face_id].append({
                    "start": time,
                    "end": time + interval,
                    "box": [x1, y1, x2, y2]
                })
                # Update face occurrences
                face_occurrences[face_id] += 1

                # Saving image
                # Extract and save the face image
                face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                face_filename = os.path.join(output_dir, f"{face_id}_occurrence_{face_occurrences[face_id]}.jpg")
                cv2.imwrite(face_filename, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                print(f"Saved {face_filename}")
    
    return face_intervals, face_embeddings, face_occurrences
def save_transcription(face_intervals, output_file="face_transcription.csv"):
    """
    Save the face detection intervals to a CSV file.

    Args:
        face_intervals (dict): Dictionary of face detection intervals.
        output_file (str): The CSV file to save the transcription (default is 'face_transcription.csv').
    """
    rows = []
    for face_id, intervals in face_intervals.items():
        for interval in intervals:
            rows.append({
                "face_id": face_id,
                "start_time": interval["start"],
                "end_time": interval["end"],
                "box_coordinates": interval["box"]
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Transcription saved to {output_file}")

def save_embeddings(face_embeddings, embedding_file="face_embeddings.csv"):
    """
    Save the face embeddings to a CSV file.

    Args:
        face_embeddings (dict): Dictionary of face embeddings.
        embedding_file (str): The CSV file to save the embeddings (default is 'face_embeddings.csv').
    """
    rows = [{"face_id": face_id, "embedding": embedding.tolist()} for face_id, embedding in face_embeddings.items()]
    df = pd.DataFrame(rows)
    df.to_csv(embedding_file, index=False)
    print(f"Embeddings saved to {embedding_file}")


def extract_faces(video_input):
    """
    Extract faces from a video and save the top 10 most frequently detected faces.

    Args:
        video_input (str): Path to the video file.

    Returns:
        list: A list of the top 10 face images.
    """
    face_intervals, face_embeddings, face_occurrences = detect_faces(video_input, use_gpu=False, interval=1)
    
    # Sort faces by occurrences and get the top 10
    top_faces = face_occurrences.most_common(10)
    
    top_face_images = []
    for face_id, _ in top_faces:
        # Use the first occurrence of the face to extract the image
        first_interval = face_intervals[face_id][0]
        x1, y1, x2, y2 = first_interval['box']
        frame = VideoFileClip(video_input).get_frame(first_interval['start'])
        face_img = frame[int(y1):int(y2), int(x1):int(x2)]
        top_face_images.append(face_img)
    
    # Save the transcription and embeddings
    save_transcription(face_intervals, "face_transcription.csv")
    save_embeddings(face_embeddings, "face_embeddings.csv")

    while len(top_face_images) < 10:
        top_face_images.append(None)

    return top_face_images  # Return the top 10 face images

# import cv2
# import dlib
# import numpy as np
# from ultralytics import YOLO
# from collections import defaultdict

# # Load the YOLOv8 model (you can switch to YOLOv9 when available)
# model = YOLO("yolov8-face.pt")

# # Load dlib's face recognition model for identifying faces across frames
# face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
# detector = dlib.get_frontal_face_detector()
# sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# # Initialize video and output
# video_path = "input_video.mp4"
# cap = cv2.VideoCapture(video_path)

# # Set up data structure for transcription intervals
# face_timeline = defaultdict(list)

# def get_face_embedding(face_image):
#     """Extracts 128D facial embedding using dlib"""
#     gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray, 1)
#     if len(faces) == 0:
#         return None
#     shape = sp(gray, faces[0])
#     return np.array(face_rec_model.compute_face_descriptor(face_image, shape))

# def detect_faces(frame, timestamp):
#     """Detect faces using YOLOv8 and extract embeddings"""
#     results = model(frame)
#     for result in results.xyxy[0]:
#         x1, y1, x2, y2, confidence, class_id = result
#         face_img = frame[int(y1):int(y2), int(x1):int(x2)]
#         embedding = get_face_embedding(face_img)
#         if embedding is not None:
#             face_timeline[tuple(embedding)].append((timestamp, (x1, y1, x2, y2)))

# def generate_intervals(face_timeline):
#     """Convert face timeline to intervals"""
#     face_intervals = {}
#     for face_id, frames in face_timeline.items():
#         intervals = []
#         start = None
#         last_timestamp = None
#         for timestamp, bbox in frames:
#             if start is None:
#                 start = timestamp
#             elif last_timestamp is not None and timestamp - last_timestamp > 1:
#                 intervals.append((start, last_timestamp))
#                 start = timestamp
#             last_timestamp = timestamp
#         intervals.append((start, last_timestamp))  # Closing the last interval
#         face_intervals[face_id] = intervals
#     return face_intervals

# # Process video
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     timestamp = frame_count / fps
#     detect_faces(frame, timestamp)
#     frame_count += 1

# cap.release()

# # Generate intervals
# face_intervals = generate_intervals(face_timeline)

# # Print results in transcription format (JSON-like)
# for face_id, intervals in face_intervals.items():
#     print(f"Face ID {face_id}:")
#     for interval in intervals:
#         print(f"  Appears from {interval[0]}s to {interval[1]}s")

# # Optionally save as JSON or CSV
# def extract_faces(video_input):

#     return []