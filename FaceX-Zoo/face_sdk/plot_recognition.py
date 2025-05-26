import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Kategorie okularów w ustalonej kolejności
okulary_labels = [
    'Brak okularow',
    'Sportowe',
    'Przeciwsloneczne',
    'Mirror',
    'Korekcyjne'
]

# Wczytaj dane
log_path = os.path.join('logs', 'recognition_logs.txt')
df = pd.read_csv(log_path, header=None, names=['datetime', 'person', 'score'])

# Grupowanie po osobie
persons = df['person'].unique()

# Dla każdej osoby znajdź bloki po 5 wyników i przypisz do kategorii okularów
okulary_data = {person: [] for person in persons}
for person in persons:
    scores = df[df['person'] == person]['score'].tolist()
    if len(scores) < 5:
        scores += [None] * (5 - len(scores))
    elif len(scores) > 5:
        scores = scores[:5]
    okulary_data[person] = scores

# Przygotuj dane: dla każdego przypadku okularów policz średnią ze wszystkich osób
okulary_scores = {label: [] for label in okulary_labels}
for person in persons:
    scores = okulary_data[person]
    for idx, label in enumerate(okulary_labels):
        if scores[idx] is not None:
            okulary_scores[label].append(scores[idx])

okulary_means = [np.mean(okulary_scores[label]) for label in okulary_labels]

# Wykres: jeden słupek na przypadek okularów (średnia ze wszystkich osób)
plt.figure(figsize=(10, 6))
bars = plt.bar(okulary_labels, okulary_means, color='skyblue')
plt.ylabel('Średni procent podobienstwa')
plt.title('Średni wynik rozpoznawania twarzy w zależności od rodzaju okularów (średnia z wszystkich osób)')
plt.ylim(0, 1)

# Dodaj wartości nad słupkami
for bar, mean in zip(bars, okulary_means):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{mean:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('logs/wykres_okulary_srednia.png')
plt.show()