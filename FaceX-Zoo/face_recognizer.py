import cv2
import numpy as np
import os

# Załaduj zdjęcia z bazy danych
def load_face_images_from_directory(directory):
    faces = []
    labels = []
    label = 0
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Konwersja do odcieni szarości
            faces.append(gray)
            labels.append(label)
            label += 1
    return faces, labels

def load_faces_from_video(video_path):
    faces = []
    labels = []
    label = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Nie udało się otworzyć pliku wideo: {video_path}")
        return faces, labels

    print(f"Przetwarzanie wideo: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in detected_faces:
            face_region = gray[y:y+h, x:x+w]
            faces.append(face_region)
            labels.append(label)
            label += 1

    cap.release()
    return faces, labels

def process_face_live(frame, recognizer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_region)

        if label == 0:
            face_name = "Face1"
        elif label == 1:
            face_name = "Face2"
        else:
            face_name = "Unknown"

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Add text with the name and confidence
        cv2.putText(frame, f"{face_name} ({100 - confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw crosshair lines on the face
        cv2.line(frame, (x, y + h // 2), (x + w, y + h // 2), (0, 255, 255), 1)  # Horizontal line
        cv2.line(frame, (x + w // 2, y), (x + w // 2, y + h), (0, 255, 255), 1)  # Vertical line
        
        # Add alert if confidence is low
        if confidence > 50:  # Adjust threshold as needed
            cv2.putText(frame, "Low Confidence!", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw a tag line pointing to the face
        cv2.line(frame, (x + w, y), (x + w + 50, y - 30), (0, 255, 0), 2)
        cv2.putText(frame, "Detected", (x + w + 55, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def main():
    # Ścieżka do pliku wideo z twarzami
    database_video = 'faces_database/video.mp4'  # Podaj ścieżkę do pliku wideo z twarzami

    # Załaduj twarze z wideo i trenuj rozpoznawanie twarzy
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = load_faces_from_video(database_video)
    if not faces:
        print("Nie udało się załadować twarzy z wideo.")
        return
    recognizer.train(faces, np.array(labels))

    # Otwórz kamerę
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie udało się otworzyć kamery.")
        return

    print("Naciśnij 'q', aby zakończyć.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nie udało się odczytać klatki z kamery.")
            break

        # Przetwórz klatkę w celu rozpoznania twarzy
        processed_frame = process_face_live(frame, recognizer)

        # Wyświetl przetworzoną klatkę
        cv2.imshow("Live Face Recognition", processed_frame)

        # Wyjdź po naciśnięciu 'q' lub zamknięciu okna
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Live Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Zwolnij zasoby
    cap.release()
    cv2.destroyAllWindows()
    print("Program zakończony.")

if __name__ == "__main__":
    main()
