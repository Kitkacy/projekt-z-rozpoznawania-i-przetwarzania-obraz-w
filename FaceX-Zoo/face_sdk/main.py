import cv2
import os
import numpy as np
import logging.config
import shutil
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog

from api_usage.face_detection import FaceDetection
from api_usage.face_alignment import FaceAlignment
from api_usage.face_recognition import FaceRecognition


logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('main')


app_det = FaceDetection('cpu')
app_alignment = FaceAlignment('cpu')
app_rec = FaceRecognition('cpu')


faces_dir = 'Twarze'  


os.makedirs(faces_dir, exist_ok=True)
logger.info(f"Folder z referencyjnymi twarzami: {faces_dir}")


if len([f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.png'))]) == 0:
    logger.info("Brak referencyjnych twarzy.")


def load_faces():
    reference_features = {}
    image_files = [f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.png'))]

    if image_files:
        logger.info(f"Znaleziono {len(image_files)} obrazy referencyjne")
        
        
        for file_name in image_files:
            image_path = os.path.join(faces_dir, file_name)
            logger.info(f"Przetwarzanie obrazu referencyjnego: {file_name}")
            
            try:
                
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Nie udało się wczytać obrazu: {file_name}, pomijam")
                    continue
                    
                
                dets = app_det.detect(image)
                if len(dets) > 0:
                    
                    det = dets[0]
                    
                    landmarks = app_alignment.get_landmarks(image, det)
                    
                    aligned_face = app_alignment.align(image, landmarks)
                    
                    feature = app_rec.get_feature(aligned_face)
                    
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


def option_1_take_photo_gui():
    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Instrukcja", "Zrób zdjęcie wciskając 'q'.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Nagrywanie", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            filename = simpledialog.askstring("Nazwa pliku", "Podaj nazwę pliku (bez rozszerzenia):")
            if filename:
                filepath = os.path.join(faces_dir, filename + ".jpg")
                cv2.imwrite(filepath, frame)
                messagebox.showinfo("Sukces", f"Zdjęcie zapisane: {filepath}")
            break
    cap.release()
    cv2.destroyAllWindows()

def option_2_record_video_gui():
    cap = cv2.VideoCapture(0)
    filename = simpledialog.askstring("Nazwa pliku", "Podaj nazwę pliku (bez rozszerzenia):")
    if not filename:
        cap.release()
        return
    filepath = os.path.join(faces_dir, filename + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 20.0, (640,480))
    messagebox.showinfo("Nagrywanie", "Nagrywanie... Naciśnij 'q', by zakończyć.")
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
    messagebox.showinfo("Sukces", f"Film zapisany: {filepath}")

def option_3_live_recognition_gui():
    reference_features = load_faces()
    cap = cv2.VideoCapture(0)
    info_text = f"{len(reference_features)} referencyjne twarze załadowane"
    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Błąd", "Nie udało się uzyskać obrazu z kamery.")
            break
        dets = app_det.detect(frame)
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for det in dets:
            x1, y1, x2, y2, _ = map(int, det)
            landmarks = app_alignment.get_landmarks(frame, det)
            aligned_face = app_alignment.align(frame, landmarks)
            feature = app_rec.get_feature(aligned_face)
            best_match = None
            best_score = -1
            threshold = 0.5
            if reference_features:
                for name, ref_feature in reference_features.items():
                    score = np.dot(feature, ref_feature)
                    if score > best_score:
                        best_score = score
                        best_match = name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if best_score > threshold:
                    label = f"{os.path.splitext(best_match)[0]} ({best_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    label = f"Nie rozpoznano ({best_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "brak twarzy do porównania"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            for i in range(landmarks.shape[0]):
                point_x = int(landmarks[i][0])
                point_y = int(landmarks[i][1])
                cv2.circle(frame, (point_x, point_y), 2, (0, 255, 0), 2)
            for i in range(0, 16):
                cv2.line(frame, (int(landmarks[i][0]), int(landmarks[i][1])),
                         (int(landmarks[i+1][0]), int(landmarks[i+1][1])), (255, 0, 0), 2)
            for i in range(17, 22):
                cv2.line(frame, (int(landmarks[i][0]), int(landmarks[i][1])),
                         (int(landmarks[i+1][0]), int(landmarks[i+1][1])), (255, 0, 0), 2)
        cv2.putText(frame, "Naciśnij 'q' aby zakończyć", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Rozpoznanie twarzy', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def option_4_video_file_gui():
    reference_features = load_faces()
    path = filedialog.askopenfilename(title="Wybierz plik wideo", filetypes=[("Pliki wideo", "*.mp4;*.avi;*.mov;*.mkv")])
    if not path or not os.path.exists(path):
        messagebox.showerror("Błąd", "Nie znaleziono pliku.")
        return
    cap = cv2.VideoCapture(path)
    info_text = f"{len(reference_features)} twarze do porównania załadowane"
    while True:
        ret, frame = cap.read()
        if not ret: break
        dets = app_det.detect(frame)
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for det in dets:
            x1, y1, x2, y2, _ = map(int, det)
            landmarks = app_alignment.get_landmarks(frame, det)
            aligned_face = app_alignment.align(frame, landmarks)
            feature = app_rec.get_feature(aligned_face)
            best_match = None
            best_score = -1
            threshold = 0.5
            if reference_features:
                for name, ref_feature in reference_features.items():
                    score = np.dot(feature, ref_feature)
                    if score > best_score:
                        best_score = score
                        best_match = name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if best_score > threshold:
                    label = f"{os.path.splitext(best_match)[0]} ({best_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    label = f"Nieznane ({best_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Naciśnij 'q' aby zakończyć", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Rozpoznawanie twarzy', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- GUI ---
def run_gui():
    root = tk.Tk()
    root.title("FaceX-Zoo - Rozpoznawanie twarzy")
    root.geometry("400x350")
    label = tk.Label(root, text="Wybierz opcję:", font=("Arial", 14))
    label.pack(pady=20)
    btn1 = tk.Button(root, text="Zrób zdjęcie", width=30, command=option_1_take_photo_gui)
    btn1.pack(pady=5)
    btn2 = tk.Button(root, text="Nagraj film", width=30, command=option_2_record_video_gui)
    btn2.pack(pady=5)
    btn3 = tk.Button(root, text="Rozpoznawanie twarzy na żywo", width=30, command=option_3_live_recognition_gui)
    btn3.pack(pady=5)
    btn4 = tk.Button(root, text="Rozpoznawanie twarzy z pliku wideo", width=30, command=option_4_video_file_gui)
    btn4.pack(pady=5)
    btn_exit = tk.Button(root, text="Zakończ", width=30, command=root.destroy)
    btn_exit.pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
