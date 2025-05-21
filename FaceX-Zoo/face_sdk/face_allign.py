import os
import cv2

def crop_center(image, crop_size=(112, 112)):
    """Crop the image to the specified size from the center.
    
    Args:
        image: Input image (numpy.ndarray).
        crop_size: Tuple specifying the crop size (width, height).
        
    Returns:
        numpy.ndarray: Cropped image.
    """
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size

    # Oblicz współrzędne wycinka względem środka
    start_x = max((w - crop_w) // 2, 0)
    start_y = max((h - crop_h) // 2, 0)
    end_x = start_x + crop_w
    end_y = start_y + crop_h

    # Przytnij obraz
    return image[start_y:end_y, start_x:end_x]

# Ścieżki
input_folder = r'D:\szkola\semestrVI\RiPO\projekt\FaceX-Zoo\Nowy folder\MeGlass_120x120\MeGlass_120x120'
output_base_folder = r'C:\Users\Lemon\OneDrive\Dokumenty\GitHub\projekt-z-rozpoznawania-i-przetwarzania-obraz-w\FaceX-Zoo\train_data\faces_with_glasses'

# Iteracja po plikach w folderze wejściowym
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        try:
            # Wyodrębnij pierwsze 10 znaków z nazwy pliku
            folder_name = filename[:10]
            output_folder = os.path.join(output_base_folder, folder_name)
            
            # Utwórz folder, jeśli nie istnieje
            os.makedirs(output_folder, exist_ok=True)
            
            # Ścieżki do pliku wejściowego i wyjściowego
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            
            # Wczytaj obraz
            image = cv2.imread(input_image_path)
            if image is None:
                print(f"Nie można wczytać obrazu: {input_image_path}")
                continue
            
            # Przytnij obraz względem środka do rozmiaru 112x112
            cropped_face = crop_center(image, crop_size=(112, 112))
            
            # Zapisz wynik
            cv2.imwrite(output_image_path, cropped_face)
            print(f"Przetworzono i zapisano: {output_image_path}")
        except Exception as e:
            print(f"Błąd podczas przetwarzania pliku {filename}: {e}")