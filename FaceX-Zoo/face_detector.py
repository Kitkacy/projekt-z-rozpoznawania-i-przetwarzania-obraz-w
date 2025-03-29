import torch
import cv2
import numpy as np

# Zakładając, że masz już załadowany model PRNet
class PRNet(torch.nn.Module):
    def __init__(self):
        super(PRNet, self).__init__()
        # Implementacja Twojego modelu PRNet
        pass

    def forward(self, x):
        # Zdefiniuj przepływ przez model
        return x

def load_prnet_model(model_path):
    model = PRNet()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)  # Ignorowanie niepasujących wag
    model.eval()
    return model

def detect_face(image):
    # Wczytanie klasyfikatora Haar Cascade do detekcji twarzy
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Konwersja obrazu do skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Wykrywanie twarzy na obrazie
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

def process_face(image, model):
    faces = detect_face(image)
    
    for (x, y, w, h) in faces:
        # Wycinanie regionu twarzy
        face_region = image[y:y+h, x:x+w]
        
        # Konwertowanie regionu twarzy do formatu odpowiedniego do modelu
        face_tensor = preprocess_image(face_region)
        
        # Przetwarzanie twarzy przez model
        output = model(face_tensor)
        
        # Opcjonalnie: Można tu przeprowadzić dalsze operacje na wyniku (np. wizualizacja 3D)
        
        # Rysowanie prostokąta wokół wykrytej twarzy
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image

def preprocess_image(image):
    # Przygotowanie obrazu do modelu (np. zmiana rozmiaru, normalizacja, konwersja na tensor)
    image = cv2.resize(image, (224, 224))  # Zmieniamy rozmiar obrazu na 224x224 (jeśli model tego wymaga)
    image = image.astype(np.float32) / 255.0  # Normalizacja (jeśli model wymaga)
    image = np.transpose(image, (2, 0, 1))  # Zmiana kształtu na (C, H, W)
    image = torch.tensor(image).unsqueeze(0)  # Dodanie wymiaru batch
    return image

def main():
    # Zmień ścieżkę na swoją
    PRNET_MODEL_PATH = "D:/szkola/semestrVI/RiPO/projekt/FaceX-Zoo/addition_module/face_mask_adding/FMA-3D/model/prnet.pth"
    
    # Załadowanie modelu
    model = load_prnet_model(PRNET_MODEL_PATH)
    
    # Wczytanie obrazu (np. z pliku)
    image = cv2.imread(r"D:\szkola\semestrVI\RiPO\projekt\FaceX-Zoo\images\query_face.jpg")

    
    # Przetwarzanie obrazu i rozpoznawanie twarzy
    processed_image = process_face(image, model)
    
    # Wyświetlanie przetworzonego obrazu z wykrytą twarzą
    cv2.imshow("Detected Face", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
