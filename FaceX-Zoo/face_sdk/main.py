import cv2
import os
import numpy as np
import logging.config
import shutil

from api_usage.face_detection import FaceDetection
from api_usage.face_alignment import FaceAlignment
from api_usage.face_recognition import FaceRecognition

# Configure logging
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('main')

# Initialize modules
app_det = FaceDetection('cpu')
app_alignment = FaceAlignment('cpu')
app_rec = FaceRecognition('cpu')

# Paths
faces_dir = 'Twarze'  # Changed path to be relative to current directory

# Create directory if it doesn't exist
os.makedirs(faces_dir, exist_ok=True)
logger.info(f"Reference face directory: {faces_dir}")

# Check if reference directory is empty
if len([f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.png'))]) == 0:
    logger.info("No reference faces found. Copying sample images...")
    
    # Create sample directory if needed
    sample_dir = os.path.join("api_usage", "test_images")
    if os.path.exists(sample_dir):
        # Copy some sample images to the reference folder
        for sample_file in os.listdir(sample_dir):
            if sample_file.endswith(('.jpg', '.png')):
                src_path = os.path.join(sample_dir, sample_file)
                dst_path = os.path.join(faces_dir, f"sample_{sample_file}")
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied sample image: {sample_file} to reference directory")

# Load reference face features
reference_features = {}
image_files = [f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.png'))]

if image_files:
    logger.info(f"Found {len(image_files)} reference images")
    
    # Process each reference image
    for file_name in image_files:
        image_path = os.path.join(faces_dir, file_name)
        logger.info(f"Processing reference image: {file_name}")
        
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to read image: {file_name}, skipping")
                continue
                
            # Detect face
            dets = app_det.detect(image)
            if len(dets) > 0:
                # Get top detection
                det = dets[0]
                # Get landmarks
                landmarks = app_alignment.get_landmarks(image, det)
                # Align and crop face
                aligned_face = app_alignment.align(image, landmarks)
                # Extract feature
                feature = app_rec.get_feature(aligned_face)
                # Store feature with name
                reference_features[file_name] = feature
                logger.info(f"Successfully loaded reference face: {file_name}")
            else:
                logger.warning(f"No face detected in {file_name}, skipping")
        except Exception as e:
            logger.error(f"Error processing reference image {file_name}: {str(e)}")
else:
    logger.warning("No reference image files found. Face recognition will not identify any faces.")

logger.info(f"Loaded {len(reference_features)} reference faces")

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Error: Could not open webcam")
    exit()

logger.info("Starting face recognition...")

# Add text to display number of reference faces
info_text = f"{len(reference_features)} reference faces loaded"

while True:
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to capture frame from camera")
        break

    # Detect faces in the frame
    dets = app_det.detect(frame)
    
    # Display the number of reference faces in the top-left corner
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    for det in dets:
        x1, y1, x2, y2, _ = map(int, det)
        
        # Get landmarks
        landmarks = app_alignment.get_landmarks(frame, det)
        # Align and crop face
        aligned_face = app_alignment.align(frame, landmarks)
        # Extract feature
        feature = app_rec.get_feature(aligned_face)

        # Compare with reference features
        best_match = None
        best_score = -1
        threshold = 0.5  # Adjust as needed
        
        if reference_features:
            for name, ref_feature in reference_features.items():
                # Compute similarity score
                score = np.dot(feature, ref_feature)
                if score > best_score:
                    best_score = score
                    best_match = name

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display name and score if above threshold
            if best_score > threshold:
                label = f"{os.path.splitext(best_match)[0]} ({best_score:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                label = f"Unknown ({best_score:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # No reference faces to compare with
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = "No reference faces"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw landmarks
        for i in range(0, landmarks.shape[0]):
            point_x = int(landmarks[i][0])
            point_y = int(landmarks[i][1])
            cv2.circle(frame, (point_x, point_y), 1, (0, 255, 0), 2)

    # Display usage instructions at the bottom of the frame
    cv2.putText(frame, "Press 'q' to quit", 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
logger.info("Face recognition ended")