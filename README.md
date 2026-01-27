# Semantic Search Engine Prototype

Wyszukiwarka dokumentów tekstowych porównująca dwa podejścia do wyszukiwania informacji: **leksykalne (TF-IDF)** oraz **semantyczne (Doc2Vec)**. Projekt został stworzony w celu demonstracji różnic między dopasowaniem słów kluczowych a zrozumieniem kontekstu zapytania.

## Funkcje

- **Dwa silniki wyszukiwania:**
  - **TF-IDF:** Skuteczny przy wyszukiwaniu dokładnych fraz i nazw własnych.
  - **Doc2Vec:** Rozpoznaje synonimy i powiązania tematyczne (np. znajduje artykuły o "director", gdy zapytasz o "filmmaker").
- **Interfejs Webowy:** Intuicyjny interfejs zbudowany w Streamlit.
- **Personalizacja wyszukiwania:**
  - Inteligentna kategoryzacja dokumentów oparta na **punktacji wagowej** (Weighted Scoring).
  - Filtrowanie wyników według typu (Naukowe, Biznesowe, Polityczne, Sport, Rozrywka).
  - Możliwość zdefiniowania liczby zwracanych wyników.
- **Dynamiczne zarządzanie danymi:**
  - Automatyczne wykrywanie zmian w katalogu `documents/`.
  - Propozycja dotrenowania modeli po dodaniu nowych plików.
- **Preprocessing:** Zaawansowane czyszczenie tekstu (tokenizacja, usuwanie stopwords, lematyzacja).
- **Eksperymenty:** Pełna analiza danych (EDA) oraz strojenie hiperparametrów (Grid Search).

## Technologie

- **Python 3.9+**
- **Streamlit** (Interfejs użytkownika)
- **Gensim** (Doc2Vec)
- **Scikit-learn** (TF-IDF, Cosine Similarity)
- **NLTK** (Preprocessing, Lemmatyzacja)
- **Pandas** (Analiza wyników)

## Instalacja

1. Sklonuj repozytorium:
   ```bash
   git clone <url-repozytorium>
   cd PJN
   ```

2. Stwórz i aktywuj środowisko wirtualne:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate   # Windows
   ```

3. Zainstaluj wymagane biblioteki:
   ```bash
   pip install streamlit gensim scikit-learn nltk pandas tqdm
   ```

## Uruchomienie

### Aplikacja Webowa (Zalecane)
```bash
streamlit run app.py
```
Aplikacja pozwala na przesyłanie nowych plików `.txt` przez przeglądarkę i natychmiastowe testowanie wyników dla obu modeli obok siebie.

### Interfejs CLI
```bash
python main.py
```
Proste menu w terminalu pozwalające na wybór silnika wyszukiwania.

## Badania i Rozwój

Projekt zawiera dwa notatniki Jupyter z procesem badawczym:
- `EDA.ipynb`: Eksploracyjna analiza zbioru dokumentów.
- `HyperparameterTuning.ipynb`: Proces doboru najlepszych parametrów.

### Najlepsze parametry (wyniki eksperymentów):
Dla uzyskania najwyższej precyzji wyszukiwania zaleca się stosowanie następujących ustawień:

- **Doc2Vec (Wyszukiwanie Semantyczne):**
  - `dm=0` (PV-DBOW) z `dbow_words=1` (najlepiej radzi sobie z synonimami).
  - `vector_size: 200-300` oraz `epochs: 100+`.
  - `steps` (inferencja): 100 dla wysokiej stabilności wektorów zapytań.
- **TF-IDF (Wyszukiwanie Leksykalne):**
  - `ngram_range: (1, 2)` – pozwala na dopasowanie fraz (np. "Supreme Court").
  - `min_df: 1` – kluczowe dla małych zbiorów, aby nie pomijać unikalnych słów kluczowych.
  - `sublinear_tf: True` – zapobiega dominacji słów o bardzo wysokiej częstotliwości.
- **Preprocessing (Wspólny):**
  - Minimalna długość słowa `len >= 2` (zachowanie kontekstu skrótów takich jak "US", "6", "VP").

## Struktura Projektu

- `documents/` - Korpus dokumentów tekstowych (pliki .txt).
- `data/` - Przechowuje zserializowane modele (`.pkl`, `.model`) oraz pliki statusu.
- `service/` - Logika biznesowa (serwisy wyszukiwania i ładowania danych).
- `model/` - Klasy encji danych (np. `Document`).
- `app.py` - Główny plik aplikacji Streamlit (Web UI).
- `main.py` - Główny plik aplikacji CLI.
- `HyperparameterTuning.ipynb` - Notatnik ze strojeniem modeli.
- `EDA.ipynb` - Eksploracyjna analiza danych.
