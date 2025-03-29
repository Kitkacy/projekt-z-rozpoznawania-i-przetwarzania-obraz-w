import cv2
import numpy as np
import os

# Tworzenie folderów do przechowywania zdjęć i filmów
faces_test_folder = "faces_to_test"
os.makedirs(faces_test_folder, exist_ok=True)

def get_all_mp4_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]

def load_faces_from_videos(video_paths):
    faces = []
    labels = []
    label_map = {}
    label = 0
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Nie udało się otworzyć pliku wideo: {video_path}")
            continue

        print(f"Przetwarzanie wideo: {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(50, 50))

            for (x, y, w, h) in detected_faces:
                face_region = gray[y:y+h, x:x+w]
                face_region = cv2.resize(face_region, (200, 200))
                faces.append(face_region)
                labels.append(label)
                label_map[label] = os.path.basename(video_path)
                label += 1

        cap.release()
    
    return faces, labels, label_map

def process_face(frame, recognizer, label_map):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_region = cv2.resize(face_region, (200, 200))
        label, confidence = recognizer.predict(face_region)
        face_name = label_map.get(label, "Unknown")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{face_name} ({100 - confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def recognize_from_camera(recognizer, label_map):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie udało się otworzyć kamery.")
        return
    
    print("Naciśnij 'q', aby zakończyć.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_face(frame, recognizer, label_map)
        cv2.imshow("Live Face Recognition", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def recognize_from_video(video_path, recognizer, label_map):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Nie udało się otworzyć pliku wideo: {video_path}")
        return
    
    print(f"Analiza pliku: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_face(frame, recognizer, label_map)
        cv2.imshow("Video Face Recognition", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def take_photo():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie udało się otworzyć kamery.")
        return

    ret, frame = cap.read()
    if ret:
        photo_path = os.path.join(faces_test_folder, "photo.png")
        cv2.imwrite(photo_path, frame)
        print(f"Zdjęcie zapisane jako {photo_path}")
    
    cap.release()
    cv2.destroyAllWindows()

def record_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie udało się otworzyć kamery.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(faces_test_folder, "recorded_video.mp4")
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    print("Nagrywanie... Naciśnij 'q', aby zakończyć.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Nagranie zapisane jako {video_path}")

def main():
    directory = "faces_database"
    mp4_files = get_all_mp4_files(directory)
    if not mp4_files:
        print("Brak plików .mp4 w podanym katalogu.")
        return
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels, label_map = load_faces_from_videos(mp4_files)
    if not faces:
        print("Nie udało się załadować twarzy z wideo.")
        return
    recognizer.train(faces, np.array(labels))

    while True:
        print("\nWybierz opcję:")
        print("1 - Analiza obrazu z kamery na żywo")
        print("2 - Analiza obrazu z zapisanego filmu")
        print("3 - Zrób zdjęcie i zapisz jako PNG")
        print("4 - Nagraj film i zapisz jako MP4")
        print("5 - Wyjście")
        choice = input("Wybór: ")
        
        if choice == '1':
            recognize_from_camera(recognizer, label_map)
        elif choice == '2':
            print("Podaj ścieżkę do pliku wideo:")
            video_path = input("Ścieżka: ")
            if not os.path.exists(video_path):
                print("Nie znaleziono pliku.")
            else:
                recognize_from_video(video_path, recognizer, label_map)
        elif choice == '3':
            take_photo()
        elif choice == '4':
            record_video()
        elif choice == '5':
            break
        else:
            print("Niepoprawny wybór. Spróbuj ponownie.")
    
    print("Program zakończony.")

if __name__ == "__main__":
    main()
