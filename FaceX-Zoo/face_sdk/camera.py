import os
import cv2
import numpy as np
import torch
import yaml
from datetime import datetime

# === Załaduj FaceX-Zoo modele ===
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

def load_models():
    with open('config/model_conf.yaml') as f:
        model_conf = yaml.load(f, Loader=yaml.FullLoader)

    model_path = 'models'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scene = 'non-mask'

    det_loader = FaceDetModelLoader(model_path, 'face_detection', model_conf[scene]['face_detection'])
    det_model, det_cfg = det_loader.load_model()
    det_handler = FaceDetModelHandler(det_model, device, det_cfg)

    align_loader = FaceAlignModelLoader(model_path, 'face_alignment', model_conf[scene]['face_alignment'])
    align_model, align_cfg = align_loader.load_model()
    align_handler = FaceAlignModelHandler(align_model, device, align_cfg)

    rec_loader = FaceRecModelLoader(model_path, 'face_recognition', model_conf[scene]['face_recognition'])
    rec_model, rec_cfg = rec_loader.load_model()
    rec_handler = FaceRecModelHandler(rec_model, device, rec_cfg)

    cropper = FaceRecImageCropper()

    return det_handler, align_handler, rec_handler, cropper

def extract_embeddings_from_image(image, det_handler, align_handler, rec_handler, cropper):
    dets = det_handler.inference_on_image(image)
    if dets.shape[0] == 0:
        return None
    det = dets[0]
    landmarks = align_handler.inference_on_image(image, det)
    landmark_list = [int(coord) for point in landmarks for coord in point]
    cropped = cropper.crop_image_by_mat(image, landmark_list)
    embedding = rec_handler.inference_on_image(cropped)
    return embedding

def load_face_database(det_handler, align_handler, rec_handler, cropper):
    db = {}
    for filename in os.listdir("base"):
        if filename.endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join("base", filename))
            embedding = extract_embeddings_from_image(img, det_handler, align_handler, rec_handler, cropper)
            if embedding is not None:
                name = os.path.splitext(filename)[0]
                db[name] = embedding
    return db

def recognize_face_in_frame(frame, db, det_handler, align_handler, rec_handler, cropper):
    dets = det_handler.inference_on_image(frame)
    for det in dets:
        x1, y1, x2, y2, _ = det.astype(int)
        embedding = extract_embeddings_from_image(frame, det_handler, align_handler, rec_handler, cropper)
        if embedding is None:
            continue
        best_match, best_score = "Unknown", -1
        for name, db_emb in db.items():
            score = np.dot(embedding, db_emb)
            if score > best_score:
                best_score = score
                best_match = name
        if best_score < 0.4:
            best_match = "Unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{best_match} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    return frame

def option_1_take_photo():
    cap = cv2.VideoCapture(0)
    print("Naciśnij ENTER, aby zrobić zdjęcie...")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Zrób zdjęcie", frame)
        if cv2.waitKey(1) == 13:  # ENTER
            filename = input("Podaj nazwę pliku (bez rozszerzenia): ")
            filepath = os.path.join("base", filename + ".jpg")
            cv2.imwrite(filepath, frame)
            print("Zdjęcie zapisane:", filepath)
            break
    cap.release()
    cv2.destroyAllWindows()

def option_2_record_video():
    cap = cv2.VideoCapture(0)
    filename = input("Podaj nazwę pliku (bez rozszerzenia): ")
    filepath = os.path.join("base", filename + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 20.0, (640,480))
    print("Nagrywanie... Naciśnij ENTER, by zakończyć.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
        cv2.imshow("Nagrywanie", frame)
        if cv2.waitKey(1) == 13:  # ENTER
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Film zapisany:", filepath)

def option_3_live_recognition(det_handler, align_handler, rec_handler, cropper):
    db = load_face_database(det_handler, align_handler, rec_handler, cropper)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = recognize_face_in_frame(frame, db, det_handler, align_handler, rec_handler, cropper)
        cv2.imshow("Rozpoznawanie na żywo", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()

def option_4_video_file(det_handler, align_handler, rec_handler, cropper):
    path = input("Podaj ścieżkę do pliku wideo: ")
    if not os.path.exists(path):
        print("Nie znaleziono pliku.")
        return
    db = load_face_database(det_handler, align_handler, rec_handler, cropper)
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = recognize_face_in_frame(frame, db, det_handler, align_handler, rec_handler, cropper)
        cv2.imshow("Rozpoznawanie z pliku", frame)
        if cv2.waitKey(30) == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    det_handler, align_handler, rec_handler, cropper = load_models()

    while True:
        print("\n--- MENU ---")
        print("1. Zrób sobie zdjęcie")
        print("2. Nagraj filmik")
        print("3. Rozpoznawanie twarzy z kamery na żywo")
        print("4. Rozpoznawanie twarzy z pliku wideo")
        print("0. Wyjście")

        choice = input("Wybierz opcję: ")
        if choice == "1":
            option_1_take_photo()
        elif choice == "2":
            option_2_record_video()
        elif choice == "3":
            option_3_live_recognition(det_handler, align_handler, rec_handler, cropper)
        elif choice == "4":
            option_4_video_file(det_handler, align_handler, rec_handler, cropper)
        elif choice == "0":
            break
        else:
            print("Nieprawidłowa opcja!")

if __name__ == "__main__":
    main()
