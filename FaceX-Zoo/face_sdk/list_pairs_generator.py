import os
import random

# Ścieżki
base_folder = r'C:\Users\Lemon\OneDrive\Dokumenty\GitHub\projekt-z-rozpoznawania-i-przetwarzania-obraz-w\FaceX-Zoo\train_data\faces_with_glasses'
output_file = r'C:\Users\Lemon\OneDrive\Dokumenty\GitHub\projekt-z-rozpoznawania-i-przetwarzania-obraz-w\FaceX-Zoo\train_data\train_list_glasses_pairs.txt'

pairs = []
max_pairs_per_person = 10  # Maksymalna liczba par dopasowanych na osobę
max_mismatch_pairs = 100   # Maksymalna liczba par niedopasowanych

# Iteracja po pierwszych 20 folderach osób
persons = sorted(os.listdir(base_folder))[:100]  # Weź tylko pierwsze 20 folderów
for person in persons:
    person_folder = os.path.join(base_folder, person)
    if os.path.isdir(person_folder):
        images = sorted(os.listdir(person_folder))  # Posortuj obrazy dla spójności
        # Tworzenie par dopasowanych (1)
        if len(images) > 1:
            selected_pairs = random.sample(
                [(images[i], images[j]) for i in range(len(images)) for j in range(i + 1, len(images))],
                min(max_pairs_per_person, len(images) * (len(images) - 1) // 2)
            )
            for img1, img2 in selected_pairs:
                pairs.append(f"{os.path.join('faces_with_glasses', person, img1)} {os.path.join('faces_with_glasses', person, img2)} 1")

# Tworzenie par niedopasowanych (0)
mismatch_pairs = []
for i, person1 in enumerate(persons):
    for person2 in persons[i + 1:]:
        person1_folder = os.path.join(base_folder, person1)
        person2_folder = os.path.join(base_folder, person2)

        if os.path.isdir(person1_folder) and os.path.isdir(person2_folder):
            person1_images = sorted(os.listdir(person1_folder))
            person2_images = sorted(os.listdir(person2_folder))

            # Losowo wybierz maksymalnie 5 par między dwiema osobami
            mismatch_pairs.extend(
                random.sample(
                    [(img1, img2) for img1 in person1_images for img2 in person2_images],
                    min(5, len(person1_images) * len(person2_images))
                )
            )

# Losowo wybierz ograniczoną liczbę par niedopasowanych
selected_mismatches = random.sample(mismatch_pairs, min(len(mismatch_pairs), max_mismatch_pairs))
for img1, img2 in selected_mismatches:
    pairs.append(f"{os.path.join('faces_with_glasses', person1, img1)} {os.path.join('faces_with_glasses', person2, img2)} 0")

# Zapis do pliku
with open(output_file, 'w') as f:
    f.write('\n'.join(pairs))

print(f"Plik train_list_glasses.txt został wygenerowany i zapisany w {output_file}")