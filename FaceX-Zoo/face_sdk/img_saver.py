import os
import cv2

def crop_center(image, crop_size=(112, 112)):
    """Crop the image to the specified size from the center."""
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size
    start_x = max((w - crop_w) // 2, 0)
    start_y = max((h - crop_h) // 2, 0)
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    return image[start_y:end_y, start_x:end_x]

# Ścieżki
input_folder = r'D:\szkola\semestrVI\RiPO\projekt\FaceX-Zoo\Nowy folder\MeGlass_120x120\MeGlass_120x120'
output_base_folder = r'C:\Users\Lemon\OneDrive\Dokumenty\GitHub\projekt-z-rozpoznawania-i-przetwarzania-obraz-w\FaceX-Zoo\train_data\faces_with_glasses'

# Liczniki zdjęć w każdym folderze
folder_image_counters = {}

# Iteracja po plikach w folderze wejściowym
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        try:
            folder_name = filename[:10]  # np. "7134850@N0"
            output_folder = os.path.join(output_base_folder, folder_name)
            os.makedirs(output_folder, exist_ok=True)

            # Inicjalizacja licznika jeśli potrzeba
            if folder_name not in folder_image_counters:
                folder_image_counters[folder_name] = 0

            # Wczytaj obraz
            input_image_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_image_path)
            if image is None:
                print(f"Nie można wczytać obrazu: {input_image_path}")
                continue

            # Przytnij obraz do 112x112
            cropped_face = crop_center(image, crop_size=(112, 112))

            # Stwórz nazwę nowego pliku wg LFW schematu
            index = folder_image_counters[folder_name]
            new_filename = f"{folder_name}_{index:04d}.jpg"
            output_image_path = os.path.join(output_folder, new_filename)

            # Zapisz przycięty obraz
            cv2.imwrite(output_image_path, cropped_face)
            print(f"Zapisano: {output_image_path}")

            # Zwiększ licznik
            folder_image_counters[folder_name] += 1

        except Exception as e:
            print(f"Błąd podczas przetwarzania pliku {filename}: {e}")
