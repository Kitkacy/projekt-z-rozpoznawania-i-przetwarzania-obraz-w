pairs_file = r'c:/Users/Lemon/OneDrive/Dokumenty/GitHub/projekt-z-rozpoznawania-i-przetwarzania-obraz-w/FaceX-Zoo/train_data/train_list_glasses.txt'
output_file = r'c:/Users/Lemon/OneDrive/Dokumenty/GitHub/projekt-z-rozpoznawania-i-przetwarzania-obraz-w/FaceX-Zoo/test_protocol/glasses_img_list.txt'

unique_images = set()
with open(pairs_file, 'r') as f:
    for line in f:
        parts = line.strip().replace('\\', '/').split()
        if len(parts) >= 2:
            # Usuń prefix 'faces_with_glasses/' jeśli jest, bo cropped_face_folder już go zawiera
            img1 = parts[0].replace('faces_with_glasses/', '')
            img2 = parts[1].replace('faces_with_glasses/', '')
            unique_images.add(img1)
            unique_images.add(img2)

with open(output_file, 'w') as f:
    for img in sorted(unique_images):
        f.write(f"{img}\n")