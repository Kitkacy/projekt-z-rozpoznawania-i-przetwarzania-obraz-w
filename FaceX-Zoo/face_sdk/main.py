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
logger.info(f"Folder z referencyjnymi twarzami: {faces_dir}")

# Check if reference directory is empty
if len([f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.png'))]) == 0:
    logger.info("Brak referencyjnych twarzy. Kopiowanie przykładowych obrazów...")
    
    # Create sample directory if needed
    sample_dir = os.path.join("api_usage", "test_images")
    if os.path.exists(sample_dir):
        # Copy some sample images to the reference folder
        for sample_file in os.listdir(sample_dir):
            if sample_file.endswith(('.jpg', '.png')):
                src_path = os.path.join(sample_dir, sample_file)
                dst_path = os.path.join(faces_dir, f"sample_{sample_file}")
                shutil.copy2(src_path, dst_path)
                logger.info(f"Skopiowano obraz przykładowy: {sample_file} do folderu referencyjnego")

# Load reference face features
def load_faces():
    reference_features = {}
    image_files = [f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.png'))]

    if image_files:
        logger.info(f"Znaleziono {len(image_files)} obrazy referencyjne")
        
        # Process each reference image
        for file_name in image_files:
            image_path = os.path.join(faces_dir, file_name)
            logger.info(f"Przetwarzanie obrazu referencyjnego: {file_name}")
            
            try:
                # Read the image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Nie udało się wczytać obrazu: {file_name}, pomijam")
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
                    logger.info(f"Pomyślnie załadowano referencyjną twarz: {file_name}")
                else:
                    logger.warning(f"Nie wykryto twarzy w {file_name}, pomijam")
            except Exception as e:
                logger.error(f"Błąd podczas przetwarzania obrazu referencyjnego {file_name}: {str(e)}")
    else:
        logger.warning("Brak plików obrazów referencyjnych. Rozpoznawanie twarzy nie będzie możliwe.")

    logger.info(f"Zaladowano {len(reference_features)} twarzy do porownania")

    return reference_features

# Option 1: Take a photo
def option_1_take_photo():
    cap = cv2.VideoCapture(0)

    print("Zrób zdjęcie wciskając 'q'.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Nagrywanie", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            filename = input("Podaj nazwę pliku (bez rozszerzenia): ")
            filepath = os.path.join("Twarze", filename + ".jpg")
            cv2.imwrite(filepath, frame)
            print("Zdjęcie zapisane:", filepath)
            break
    cap.release()
    cv2.destroyAllWindows()

# Option 2: Record a video
def option_2_record_video():
    cap = cv2.VideoCapture(0)
    filename = input("Podaj nazwę pliku (bez rozszerzenia): ")
    filepath = os.path.join("Twarze", filename + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 20.0, (640,480))
    print("Nagrywanie... Nacisnij q, by zakończyć.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
        cv2.imshow("Nagrywanie", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Film zapisany:", filepath)

# Option 3: Face recognition from webcam
# Option 3: Face recognition from webcam with facial features and boundaries drawn
def option_3_live_recognition():
    reference_features = load_faces()
    cap = cv2.VideoCapture(0)
    info_text = f"{len(reference_features)} referencyjne twarze załadowane"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nie udało się uzyskać obrazu z kamery.")
            break

        # Detect faces in the frame
        dets = app_det.detect(frame)
        
        # Display the number of reference faces in the top-left corner
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for det in dets:
            x1, y1, x2, y2, _ = map(int, det)
            landmarks = app_alignment.get_landmarks(frame, det)
            aligned_face = app_alignment.align(frame, landmarks)
            feature = app_rec.get_feature(aligned_face)

            # Compare with reference features
            best_match = None
            best_score = -1
            threshold = 0.5  # Set threshold to adjust the confidence level
            
            if reference_features:
                for name, ref_feature in reference_features.items():
                    # Compute similarity score
                    score = np.dot(feature, ref_feature)
                    if score > best_score:
                        best_score = score
                        best_match = name

                # Draw bounding box around the detected face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display name and score if above threshold
                if best_score > threshold:
                    label = f"{os.path.splitext(best_match)[0]} ({best_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    label = f"Nie rozpoznano ({best_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            else:
                # No reference faces to compare with
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "brak twarzy do porównania"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw landmarks on the face (e.g., eyes, nose, mouth)
            for i in range(landmarks.shape[0]):
                point_x = int(landmarks[i][0])
                point_y = int(landmarks[i][1])
                cv2.circle(frame, (point_x, point_y), 2, (0, 255, 0), 2)  # Draw circles at landmarks points

            # Draw facial contours (lines connecting key facial points)
            for i in range(0, 16):  # Jawline
                cv2.line(frame, (int(landmarks[i][0]), int(landmarks[i][1])),
                         (int(landmarks[i+1][0]), int(landmarks[i+1][1])), (255, 0, 0), 2)
            # Eyes, eyebrows, nose, mouth (using landmarks indices, specific to the face detection model)
            for i in range(17, 22):  # Right eyebrow
                cv2.line(frame, (int(landmarks[i][0]), int(landmarks[i][1])),
                         (int(landmarks[i+1][0]), int(landmarks[i+1][1])), (255, 0, 0), 2)
            # Repeat for other features (left eyebrow, nose, etc.)
            # You can add more lines to connect additional points as needed

        # Display usage instructions
        cv2.putText(frame, "Naciśnij 'q' aby powrocic do menu", 
                    (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Rozpoznanie twarzy', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# Option 4: Face recognition from video file
def option_4_video_file():
    reference_features = load_faces()
    path = input("Podaj ścieżkę do pliku wideo: ")
    if not os.path.exists(path):
        print("Nie znaleziono pliku.")
        return
    
    cap = cv2.VideoCapture(path)
    info_text = f"{len(reference_features)} twarze do porównania załadowane"
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Detect faces in the frame
        dets = app_det.detect(frame)
        
        # Display the number of reference faces in the top-left corner
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for det in dets:
            x1, y1, x2, y2, _ = map(int, det)
            landmarks = app_alignment.get_landmarks(frame, det)
            aligned_face = app_alignment.align(frame, landmarks)
            feature = app_rec.get_feature(aligned_face)

            # Compare with reference features
            best_match = None
            best_score = -1
            threshold = 0.5
            
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
                    label = f"Nieznane ({best_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display usage instructions
        cv2.putText(frame, "Press 'q' to quit", 
                    (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Rozpoznawanie twarzy', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Menu options
def show_menu():
    print("\nWybierz opcję:")
    print("1. Zrób zdjęcie")
    print("2. Nagraj film")
    print("3. Rozpoznawanie twarzy na żywo")
    print("4. Rozpoznawanie twarzy z pliku wideo")
    print("0. Zakończ")

while True:
    show_menu()
    choice = input("Wybór: ")

    if choice == '1':
        option_1_take_photo()
    elif choice == '2':
        option_2_record_video()
    elif choice == '3':
        option_3_live_recognition()
    elif choice == '4':
        option_4_video_file()
    elif choice == '0':
        print("Zakończono program.")
        break
    else:
        print("Niepoprawny wybór. Spróbuj ponownie.")
