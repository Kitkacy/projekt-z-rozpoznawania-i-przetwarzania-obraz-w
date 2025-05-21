import os

base_dir = r'C:\Users\Lemon\OneDrive\Dokumenty\GitHub\projekt-z-rozpoznawania-i-przetwarzania-obraz-w\FaceX-Zoo\train_data\faces_with_glasses'
output_file = r'C:\Users\Lemon\OneDrive\Dokumenty\GitHub\projekt-z-rozpoznawania-i-przetwarzania-obraz-w\FaceX-Zoo\train_data\train_list_glasses.txt'

with open(output_file, 'w') as f:
    for label, person in enumerate(sorted(os.listdir(base_dir))):
        person_dir = os.path.join(base_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for img in os.listdir(person_dir):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.join(person, img).replace("\\", "/")  # unix-style path
                f.write(f"{rel_path} {label}\n")

print(f"Plik '{output_file}' został wygenerowany poprawnie.")