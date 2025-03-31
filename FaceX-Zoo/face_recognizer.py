import cv2
import numpy as np
import os

# Tworzenie folderów do przechowywania zdjęć i filmów
faces_test_folder = "faces_to_test"
os.makedirs(faces_test_folder, exist_ok=True)

def get_all_mp4_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]


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

    while True:
        print("\nWybierz opcję:")
        print("1 - Zrób zdjęcie i zapisz jako PNG")
        print("2 - Nagraj film i zapisz jako MP4")
        print("3 - Wyjście")
        choice = input("Wybór: ")
        
        if choice == '1':
            take_photo()
        elif choice == '2':
            record_video()
        elif choice == '3':
            break
        else:
            print("Niepoprawny wybór. Spróbuj ponownie.")
    
    print("Program zakończony.")

if __name__ == "__main__":
    main()
