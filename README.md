# projekt-z-rozpoznawania-i-przetwarzania-obraz-w

## Wykorzystanie API FaceX-Zoo

Program korzysta z trzech głównych modułów API FaceX-Zoo: `FaceDetection`, `FaceAlignment` i `FaceRecognition`. Poniżej opisano, jak każdy z nich jest używany.

### 1. Detekcja twarzy (FaceDetection)
Moduł `FaceDetection` służy do wykrywania twarzy na obrazie. W programie jest inicjalizowany w następujący sposób:
```python
from api_usage.face_detection import FaceDetection
app_det = FaceDetection('cpu')
```
- Metoda `detect` zwraca listę wykrytych twarzy w formie współrzędnych prostokątów:
  ```python
  dets = app_det.detect(frame)
  ```
- Wynik detekcji jest używany do zaznaczania twarzy na obrazie oraz jako dane wejściowe dla kolejnych modułów.

### 2. Wyrównanie twarzy (FaceAlignment)
Moduł `FaceAlignment` identyfikuje punkty charakterystyczne twarzy i wyrównuje ją. Inicjalizacja:
```python
from api_usage.face_alignment import FaceAlignment
app_alignment = FaceAlignment('cpu')
```
- Metoda `get_landmarks` zwraca 68 punktów charakterystycznych twarzy:
  ```python
  landmarks = app_alignment.get_landmarks(frame, det)
  ```
- Metoda `align` wyrównuje twarz na podstawie punktów charakterystycznych:
  ```python
  aligned_face = app_alignment.align(frame, landmarks)
  ```

### 3. Rozpoznawanie twarzy (FaceRecognition)
Moduł `FaceRecognition` generuje wektory cech twarzy i porównuje je z referencyjnymi. Inicjalizacja:
```python
from api_usage.face_recognition import FaceRecognition
app_rec = FaceRecognition('cpu')
```
- Metoda `get_feature` generuje wektor cech dla wyrównanej twarzy:
  ```python
  feature = app_rec.get_feature(aligned_face)
  ```
- Porównanie twarzy odbywa się poprzez obliczenie iloczynu skalarnego między wektorami cech:
  ```python
  score = np.dot(feature, ref_feature)
  ```

## Przepływ danych w programie

1. **Inicjalizacja modułów API**:
   - Moduły `FaceDetection`, `FaceAlignment` i `FaceRecognition` są inicjalizowane z parametrem `'cpu'`.

2. **Wczytywanie twarzy referencyjnych**:
   - Zdjęcia z folderu `Twarze` są przetwarzane w funkcji `load_faces`:
     - Wykrycie twarzy → Wyrównanie → Generowanie wektora cech
     - Wektory cech są przechowywane w słowniku z nazwami plików jako kluczami.

3. **Rozpoznawanie twarzy**:
   - Dla każdej klatki obrazu:
     - Wykrycie twarzy za pomocą `app_det.detect`
     - Wyrównanie twarzy za pomocą `app_alignment.align`
     - Generowanie wektora cech za pomocą `app_rec.get_feature`
     - Porównanie z referencyjnymi wektorami cech
     - Wyświetlenie wyniku rozpoznania na obrazie.

## Konfiguracja
- Wszystkie moduły API są uruchamiane na CPU (`'cpu'`), ale można użyć GPU.
- Próg rozpoznawania (domyślnie 0.5) można dostosować w kodzie:
  ```python
  threshold = 0.5
  ```

