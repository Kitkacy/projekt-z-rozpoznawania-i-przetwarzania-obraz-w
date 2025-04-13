import cv2
import numpy as np
import os
import torch
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler
import yaml

# Wczytaj konfigurację modeli
with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

model_path = 'models'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scene = 'non-mask'

# === Inicjalizacja modeli ===
# Detekcja
det_loader = FaceDetModelLoader(model_path, 'face_detection', model_conf[scene]['face_detection'])
det_model, det_cfg = det_loader.load_model()
det_handler = FaceDetModelHandler(det_model, device, det_cfg)

# Landmarki
align_loader = FaceAlignModelLoader(model_path, 'face_alignment', model_conf[scene]['face_alignment'])
align_model, align_cfg = align_loader.load_model()
align_handler = FaceAlignModelHandler(align_model, device, align_cfg)

# Rozpoznawanie
rec_loader = FaceRecModelLoader(model_path, 'face_recognition', model_conf[scene]['face_recognition'])
rec_model, rec_cfg = rec_loader.load_model()
rec_handler = FaceRecModelHandler(rec_model, device, rec_cfg)

cropper = FaceRecImageCropper()

# === Wczytaj twarze z bazy ===
face_db = {}  # nazwa: wektor_cech

for filename in os.listdir("base"):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join("base", filename)
        image = cv2.imread(image_path)
        dets = det_handler.inference_on_image(image)
        if dets.shape[0] == 0:
            continue
        landmarks = align_handler.inference_on_image(image, dets[0])
        landmark_list = [int(coord) for point in landmarks for coord in point]
        cropped = cropper.crop_image_by_mat(image, landmark_list)
        embedding = rec_handler.inference_on_image(cropped)
        name = os.path.splitext(filename)[0]
        face_db[name] = embedding

print(f"Wczytano bazę twarzy: {list(face_db.keys())}")

# === Kamera na żywo ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    dets = det_handler.inference_on_image(frame)
    for det in dets:
        x1, y1, x2, y2, _ = det.astype(int)
        landmarks = align_handler.inference_on_image(frame, det)
        landmark_list = [int(coord) for point in landmarks for coord in point]
        cropped = cropper.crop_image_by_mat(frame, landmark_list)
        embedding = rec_handler.inference_on_image(cropped)

        # Porównanie z bazą
        best_score = -1
        best_match = "Unknown"
        for name, base_embedding in face_db.items():
            score = np.dot(embedding, base_embedding)
            if score > best_score:
                best_score = score
                best_match = name

        # Próg podobieństwa - ustaw według potrzeb (np. 0.4 - 0.6)
        if best_score < 0.4:
            best_match = "Unknown"

        # Wyświetl wynik
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{best_match} ({best_score:.2f})', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("FaceX-Zoo Live Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
