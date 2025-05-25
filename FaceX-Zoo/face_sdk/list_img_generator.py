import os

base_folder = r'c:/Users/Lemon/OneDrive/Dokumenty/GitHub/projekt-z-rozpoznawania-i-przetwarzania-obraz-w/FaceX-Zoo/train_data/faces_with_glasses'
output_file = r'c:/Users/Lemon/OneDrive/Dokumenty/GitHub/projekt-z-rozpoznawania-i-przetwarzania-obraz-w/FaceX-Zoo/train_data/glasses_img_list.txt'

lines = []
for person in sorted(os.listdir(base_folder)):
    person_folder = os.path.join(base_folder, person)
    if not os.path.isdir(person_folder):
        continue
    for img_name in sorted(os.listdir(person_folder)):
        # Sprawdź, czy plik zaczyna się jak folder (osoba)
        if img_name.startswith(person):
            rel_path = f"{person}/{img_name}"
            lines.append(rel_path)

with open(output_file, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line + '\n')

print(f"Wygenerowano {len(lines)} ścieżek do pliku {output_file}")