#!/usr/bin/env python3
# filepath: /home/wicek/Documents/GitHub/projekt-z-rozpoznawania-i-przetwarzania-obraz-w/FaceX-Zoo/face_sdk/main.py
import cv2
import os
import numpy as np
import logging.config
import shutil
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import datetime
import time

from api_usage.face_detection import FaceDetection
from api_usage.face_alignment import FaceAlignment
from api_usage.face_recognition import FaceRecognition


logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('main')


app_det = FaceDetection('cpu')
app_alignment = FaceAlignment('cpu')
app_rec = FaceRecognition('cpu')


faces_dir = 'Twarze'  
logs_dir = 'Logs'
log_file = os.path.join(logs_dir, 'recognition_logs.txt')


# Zmienna globalna do przechowywania stanu logowania
enable_logging = False


os.makedirs(faces_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
logger.info(f"Folder z referencyjnymi twarzami: {faces_dir}")
logger.info(f"Folder z logami rozpoznania: {logs_dir}")


if len([f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.png'))]) == 0:
    logger.info("Brak referencyjnych twarzy.")


def log_recognition_result(name, score, timestamp):
    """
    Zapisuje wynik rozpoznawania do pliku log.
    
    Args:
        name (str): Nazwa rozpoznanej osoby lub "Nieznane"
        score (float): Wynik pewności rozpoznania
        timestamp (str): Znacznik czasu rozpoznania
    """
    if not enable_logging:
        return
        
    os.makedirs(logs_dir, exist_ok=True)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp},{name},{score:.4f}\n")
    
    logger.info(f"Zapisano log rozpoznania: {name}, {score:.4f}, {timestamp}")


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
    messagebox.showinfo("Instrukcja", "Zrób zdjęcie wciskając 'Q'.")
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
    messagebox.showinfo("Nagrywanie", "Nagrywanie... Naciśnij 'Q', by zakończyć.")
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
    global enable_logging
    reference_features = load_faces()
    cap = cv2.VideoCapture(0)
    info_text = f"{len(reference_features)} referencyjne twarze zaladowane"
    
    # Zmienne do śledzenia detekcji twarzy
    face_tracking = {}  # {face_id: {'last_name': name, 'last_score': score}}
    face_id_counter = 0
    
    # Zmienna do śledzenia czy w bieżącej klatce należy logować twarze
    log_current_frame = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Błąd", "Nie udało się uzyskać obrazu z kamery.")
            break
        
        current_time = time.time()
        dets = app_det.detect(frame)
        
        # Informacja o logowaniu
        log_status = "Logowanie: Tak" if enable_logging else "Logowanie: Nie"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, log_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Lista aktywnych twarzy w bieżącej klatce
        active_faces = []
        
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
                
                # Określ rozpoznaną nazwę do wyświetlenia
                if best_score > threshold:
                    display_name = os.path.splitext(best_match)[0]
                    label = f"{display_name} ({best_score:.2f})"
                    color = (0, 255, 0)  # zielony
                    
                    # Przygotuj dane do logowania
                    name_to_log = display_name
                else:
                    label = f"Nieznane ({best_score:.2f})"
                    display_name = "Nieznane"
                    color = (0, 0, 255)  # czerwony
                    
                    # Przygotuj dane do logowania
                    name_to_log = "Nieznane"
                
                # Śledzenie twarzy dla logowania przy naciśnięciu klawisza L
                face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                face_found = False
                face_id = None
                
                # Sprawdź, czy to ta sama twarz co wcześniej (proste porównanie pozycji)
                for fid, data in face_tracking.items():
                    if 'center' in data:
                        prev_center = data['center']
                        # Jeśli pozycja twarzy jest bliska poprzedniej pozycji, uznaj że to ta sama twarz
                        if abs(prev_center[0] - face_center[0]) < 50 and abs(prev_center[1] - face_center[1]) < 50:
                            face_id = fid
                            face_found = True
                            break
                
                # Jeśli to nowa twarz, przydziel jej nowe ID
                if not face_found:
                    face_id = face_id_counter
                    face_id_counter += 1
                    face_tracking[face_id] = {'last_name': name_to_log, 
                                             'last_score': best_score, 'center': face_center}
                else:
                    # Aktualizuj dane dla istniejącej twarzy
                    face_tracking[face_id]['last_name'] = name_to_log
                    face_tracking[face_id]['last_score'] = best_score
                    face_tracking[face_id]['center'] = face_center
                
                # Dodaj do listy aktywnych twarzy
                active_faces.append(face_id)
                
                # Loguj twarz, jeśli naciśnięto klawisz L i logowanie jest włączone
                if enable_logging and log_current_frame and face_id in face_tracking:
                    # Zapisz log dla tej twarzy
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_recognition_result(
                        face_tracking[face_id]['last_name'],
                        face_tracking[face_id]['last_score'],
                        timestamp
                    )
                    # Dodaj informację o zalogowaniu na ekranie
                    cv2.putText(frame, "Zalogowano!", (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Wyświetl etykietę z rozpoznaną twarzą
                try:
                    fontpath = os.path.join(os.path.dirname(__file__), "arial.ttf")
                    if os.path.exists(fontpath):
                        font = ImageFont.truetype(fontpath, 22)
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_img)
                        draw.text((x1, y1 - 30), label, font=font, fill=color[::-1])  # RGB vs BGR
                        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    else:
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception:
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
        # Usuń twarze, które przestały być widoczne (po 1 sekundzie nieobecności)
        faces_to_remove = []
        for face_id, data in face_tracking.items():
            if face_id not in active_faces:
                # Jeśli twarz nie została wykryta w tej klatce
                if 'last_seen' not in data:
                    data['last_seen'] = current_time
                elif current_time - data['last_seen'] > 1.0:  # Usuń po 1 sekundzie braku detekcji
                    faces_to_remove.append(face_id)
            else:
                # Twarz jest aktywna, zresetuj licznik
                if 'last_seen' in data:
                    del data['last_seen']
        
        # Usuń twarze zaznaczone do usunięcia
        for face_id in faces_to_remove:
            del face_tracking[face_id]

        cv2.putText(frame, "Nacisnij 'L' aby zapisac wynik", (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Nacisnij 'Q' aby zakonczyc", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Rozpoznanie twarzy', frame)
        
        # Sprawdź czy naciśnięto klawisz
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l') or key == ord('L'):
            if enable_logging:
                log_current_frame = True
                print("Zapisywanie logów dla bieżącej klatki")
            else:
                messagebox.showinfo("Info", "Logowanie jest wyłączone. Włącz je w menu głównym.")
        else:
            log_current_frame = False  # Reset flagi po każdej klatce, która nie ma 'L'
    cap.release()
    cv2.destroyAllWindows()

def option_4_video_file_gui():
    reference_features = load_faces()
    path = filedialog.askopenfilename(title="Wybierz plik wideo", filetypes=[("Pliki wideo", "*.mp4;*.avi;*.mov;*.mkv")])
    if not path or not os.path.exists(path):
        messagebox.showerror("Błąd", "Nie znaleziono pliku.")
        return
    cap = cv2.VideoCapture(path)
    info_text = f"{len(reference_features)} twarze do porownania zaladowane"
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
                    # Użyj czcionki obsługującej polskie znaki
                    try:
                        fontpath = os.path.join(os.path.dirname(__file__), "arial.ttf")
                        if os.path.exists(fontpath):
                            font = ImageFont.truetype(fontpath, 22)
                            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_img)
                            draw.text((x1, y1 - 30), label, font=font, fill=(0,255,0))
                            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        else:
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception:
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    label = f"Nieznane ({best_score:.2f})"
                    try:
                        fontpath = os.path.join(os.path.dirname(__file__), "arial.ttf")
                        if os.path.exists(fontpath):
                            font = ImageFont.truetype(fontpath, 22)
                            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_img)
                            draw.text((x1, y1 - 30), label, font=font, fill=(0,0,255))
                            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        else:
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    except Exception:
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Nacisnij 'Q' aby zakonczyc", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Rozpoznawanie twarzy', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- GUI ---
def run_gui():
    global enable_logging
    
    def toggle_logging():
        global enable_logging
        enable_logging = logging_var.get()
        if enable_logging:
            logger.info("Włączono zapisywanie logów rozpoznawania")
        else:
            logger.info("Wyłączono zapisywanie logów rozpoznawania")
    
    root = tk.Tk()
    root.title("FaceX-Zoo - Rozpoznawanie twarzy")
    root.geometry("600x700")  # Increased height from 700 to 750
    root.resizable(False, False)

    # Wczytaj obraz jako tło
    try:
        logo_path = os.path.join(os.path.dirname(__file__), "face_logo.png")
        bg_img = Image.open(logo_path)
        bg_img = bg_img.resize((600, 700), Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_img)
        bg_label = tk.Label(root, image=bg_photo)
        bg_label.image = bg_photo
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    except Exception as e:
        logger.warning(f"Nie udało się załadować grafiki tła: {e}")

    # Półprzezroczysty panel na opcje
    panel = tk.Frame(root, bg="#ffffff", bd=0)
    panel.place(relx=0.5, rely=0.5, anchor="center", width=400, height=500)  # Zwiększona wysokość panelu

    label = tk.Label(panel, text="Wybierz opcję:", font=("Segoe UI", 20, "bold"), bg="#ffffff", fg="#1a237e")
    label.pack(pady=(30, 18))

    button_style = {
        "font": ("Segoe UI", 14, "bold"),
        "bg": "#1976d2",
        "fg": "white",
        "activebackground": "#63a4ff",
        "activeforeground": "#1a237e",
        "relief": tk.RAISED,
        "bd": 3,
        "highlightthickness": 0,
        "cursor": "hand2",
        "height": 2,
        "width": 28
    }

    btn1 = tk.Button(panel, text="Zrób zdjęcie", command=option_1_take_photo_gui, **button_style)
    btn1.pack(pady=7)
    btn2 = tk.Button(panel, text="Nagraj film", command=option_2_record_video_gui, **button_style)
    btn2.pack(pady=7)
    btn3 = tk.Button(panel, text="Rozpoznawanie twarzy na żywo", command=option_3_live_recognition_gui, **button_style)
    btn3.pack(pady=7)
    btn4 = tk.Button(panel, text="Rozpoznawanie twarzy\nz pliku wideo", command=option_4_video_file_gui, **button_style, justify="center")
    btn4.pack(pady=7)
    
    # Checkbox do włączania/wyłączania logowania rozpoznań
    logging_frame = tk.Frame(panel, bg="#ffffff")
    logging_frame.pack(pady=10)
    
    logging_var = tk.BooleanVar(value=enable_logging)
    logging_checkbox = tk.Checkbutton(
        logging_frame, 
        text="Zapisuj logi rozpoznawania", 
        variable=logging_var,
        command=toggle_logging,
        bg="#ffffff",
        fg="#1a237e",
        font=("Segoe UI", 11),
        activebackground="#ffffff",
        selectcolor="#e3f2fd"
    )
    logging_checkbox.pack(side=tk.LEFT)
    
    exit_style = button_style.copy()
    exit_style["bg"] = "#b71c1c"
    exit_style["activebackground"] = "#ef5350"
    btn_exit = tk.Button(panel, text="Zakończ", command=root.destroy, **exit_style)
    btn_exit.pack(pady=20)  # Increased padding for exit button

    root.mainloop()

if __name__ == "__main__":
    run_gui()
