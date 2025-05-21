import os
from itertools import combinations

# Ścieżka do folderu z podfolderami osób
base_dir = r'C:\Users\Lemon\OneDrive\Dokumenty\GitHub\projekt-z-rozpoznawania-i-przetwarzania-obraz-w\FaceX-Zoo\train_data\faces_with_glasses'
output_file = r'C:\Users\Lemon\OneDrive\Dokumenty\GitHub\projekt-z-rozpoznawania-i-przetwarzania-obraz-w\FaceX-Zoo\train_data\train_list_glasses.txt'


with open(output_file, 'w') as f:
    for person in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person)
        if not os.path.isdir(person_dir):
            continue
        images = sorted([img for img in os.listdir(person_dir) if img.lower().endswith(('.jpg', '.png', '.jpeg'))])
        # Tworzymy wszystkie możliwe pary indeksów pozycji obrazów
        for i, j in combinations(range(len(images)), 2):
            f.write(f"{person}\t{i}\t{j}\n")

print(f"Plik {output_file} został wygenerowany.")
