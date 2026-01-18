# Klasyfikacja Obrazów Komórek Krwi (BloodMNIST)

Projekt stanowi implementację oraz porównanie klasycznych i głębokich modeli uczenia maszynowego do zadania klasyfikacji obrazów komórek krwi z wykorzystaniem zbioru danych BloodMNIST.

W ramach projektu analizowane są:
- klasyczne modele uczenia maszynowego (scikit-learn),
- konwolucyjne sieci neuronowe (CNN) zaimplementowane w PyTorch,
- metody interpretowalności modeli (Grad-CAM),
- wizualizacja wyników oraz interfejs GUI do predykcji.

---

## Struktura projektu

Projekt posiada modularną strukturę z wyraźnym podziałem odpowiedzialności pomiędzy obsługę danych, modele, narzędzia pomocnicze oraz testy.

```
.
├── output/ # Wyniki eksperymentów, modele, wizualizacje
├── src/ # Główny pakiet Python
│ ├── data/ # Obsługa danych
│ │ ├── init.py
│ │ └── datasets.py # Ładowanie i transformacja BloodMNIST
│ │
│ ├── models/ # Definicje modeli
│ │ ├── init.py
│ │ ├── cnn_models.py # SimpleBloodCNN, DeepBloodCNN
│ │ └── trad_models.py # LR, SVM, RF, MLP
│ │
│ ├── utils/ # Narzędzia pomocnicze
│ │ ├── init.py
│ │ ├── train_utils.py # Trening i ewaluacja (PyTorch)
│ │ └── plot_utils.py # Wykresy i macierze pomyłek
│ │
│ ├── tests/ # Testy i walidacja
│ │ ├── test_api_roundtrip.py
│ │ └── test_model_direct.py
│ │
│ ├── init.py
│ ├── api.py # Interfejs API do predykcji
│ ├── config.py # Konfiguracja globalna
│ ├── grad_cam.py # Wizualizacja Grad-CAM
│ ├── gui_app.py # Aplikacja GUI (Streamlit)
│ ├── inference.py # Predykcja na zbiorze testowym
│ ├── run_cnn.py # Trening modeli CNN
│ ├── run_trad.py # Trening modeli klasycznych
│ └── save_samples.py # Zapis poprawnych i błędnych predykcji
└── README.md
```


---

## Uruchamianie projektu

### Wymagania
- Python 3.8+
- Biblioteki wymienione w pliku requirements.txt

### Instalacja zależności

```bash
pip install -r requirements.txt
```

---

## Modele i eksperymenty

### Modele klasyczne (scikit-learn)

```bash
python src/run_trad.py
```

---

### Modele CNN (PyTorch)

```bash
python src/run_cnn.py
```

---

## Inne funkcjonalności

### Wizualizacja uwagi modelu (Grad-CAM)

```bash
python src/grad_cam.py
```

---

### Aplikacja GUI

```bash
streamlit run src/gui_app.py
```

---

### Testy i walidacja

```bash
python src/tests/test_api_roundtrip.py
python src/tests/test_model_direct.py
```

---

## Dataset

Projekt wykorzystuje **BloodMNIST**, będący częścią zbioru **MedMNIST**.
Zbiór zawiera obrazy mikroskopowe komórek krwi (28×28 RGB) przypisane do 8 klas.

---

**Authors:** **Jakub Pedryc** and **Maciej Łabuz**
