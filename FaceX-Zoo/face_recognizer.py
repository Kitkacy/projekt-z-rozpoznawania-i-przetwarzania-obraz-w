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

# Funkcja rozpoznawania twarzy
def recognize_face(face, database_faces):
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Trenowanie rozpoznawania twarzy z bazy danych
    faces, labels = load_face_images_from_directory(database_faces)
    recognizer.train(faces, np.array(labels))

    # Rozpoznanie twarzy w przekazanym obrazie
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Upewnij się, że obraz jest w skali szarości
    label, confidence = recognizer.predict(gray)
    
    return label, confidence

# Funkcja, która przetwarza obraz
def process_face(image, database_faces):
    label, confidence = recognize_face(image, database_faces)
    
    if label == 0:
        face_name = "Face1"
    elif label == 1:
        face_name = "Face2"
    else:
        face_name = "Face3"
    
    print(f"Face recognized as {face_name} with confidence {100 - confidence:.2f}")
    
    return image

# Przykładowe wywołanie głównej funkcji
def main():
    # Ścieżka do bazy zdjęć z twarzami (face1.jpg, face2.jpg, etc.)
    database_faces = 'faces_database'  # Podaj ścieżkę do folderu z obrazkami (np. face1.jpg, face2.jpg)

    # Ścieżka do obrazu wejściowego, na którym chcesz rozpoznać twarz
    input_image_path = 'images/leon.png'  # Ścieżka do obrazu wejściowego
    
    # Załaduj obraz
    image = cv2.imread(input_image_path)
    
    if image is None:
        print("Nie udało się załadować obrazu.")
        return

    # Przetwórz obraz w celu rozpoznania twarzy
    processed_image = process_face(image, database_faces)

    # Pokaż przetworzony obraz z oznaczoną twarzą
    cv2.imshow("Recognized Face", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
