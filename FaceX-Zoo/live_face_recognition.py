import sys
import cv2
import yaml
import logging.config
import numpy as np
from face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from face_sdk.core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from face_sdk.core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler
from face_sdk.core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

# Configure logging
logging.config.fileConfig("face_sdk/config/logging.conf")
logger = logging.getLogger('api')

# Load model configurations
with open('face_sdk/config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

def initialize_models(scene='non-mask'):
    """Initialize face detection, alignment, and recognition models."""
    model_path = 'face_sdk/models'

    # Face detection
    face_det_model_name = model_conf[scene]['face_detection']
    faceDetModelLoader = FaceDetModelLoader(model_path, 'face_detection', face_det_model_name)
    face_det_model, face_det_cfg = faceDetModelLoader.load_model()
    faceDetModelHandler = FaceDetModelHandler(face_det_model, 'cuda:0', face_det_cfg)

    # Face alignment
    face_align_model_name = model_conf[scene]['face_alignment']
    faceAlignModelLoader = FaceAlignModelLoader(model_path, 'face_alignment', face_align_model_name)
    face_align_model, face_align_cfg = faceAlignModelLoader.load_model()
    faceAlignModelHandler = FaceAlignModelHandler(face_align_model, 'cuda:0', face_align_cfg)

    # Face recognition
    face_rec_model_name = model_conf[scene]['face_recognition']
    faceRecModelLoader = FaceRecModelLoader(model_path, 'face_recognition', face_rec_model_name)
    face_rec_model, face_rec_cfg = faceRecModelLoader.load_model()
    faceRecModelHandler = FaceRecModelHandler(face_rec_model, 'cuda:0', face_rec_cfg)

    return faceDetModelHandler, faceAlignModelHandler, faceRecModelHandler

def recognize_faces_from_camera():
    """Perform live face recognition using the laptop camera."""
    faceDetModelHandler, faceAlignModelHandler, faceRecModelHandler = initialize_models()
    face_cropper = FaceRecImageCropper()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open the camera.")
        return

    logger.info("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame from camera.")
            break

        try:
            # Detect faces
            dets = faceDetModelHandler.inference_on_image(frame)
            for det in dets:
                # Align face
                landmarks = faceAlignModelHandler.inference_on_image(frame, det)
                landmarks_list = [coord for (x, y) in landmarks.astype(np.int32) for coord in (x, y)]

                # Crop face
                cropped_face = face_cropper.crop_image_by_mat(frame, landmarks_list)

                # Recognize face
                feature = faceRecModelHandler.inference_on_image(cropped_face)
                logger.info(f"Extracted feature: {feature[:5]}...")  # Log first 5 feature values

                # Draw bounding box
                x1, y1, x2, y2 = map(int, det[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            logger.error(f"Error during face recognition: {e}")

        # Display the frame
        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces_from_camera()
