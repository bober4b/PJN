# Dokumentacja Projektu: Wyszukiwarka dokumentów z wykorzystaniem reprezentacji wektorowej

## 1. Przedstawienie zadania projektowego

**Cel projektu:**
Celem niniejszego projektu było opracowanie i wdrożenie systemu wyszukiwania informacji, który wykracza poza proste dopasowanie tekstowe. System umożliwia użytkownikowi odnalezienie dokumentów najbardziej trafnych merytorycznie poprzez zastosowanie dwóch odmiennych paradygmatów reprezentacji tekstu: statystycznego (TF-IDF) oraz semantycznego (Doc2Vec).

**Problem badawczy:**
Tradycyjne systemy wyszukiwania oparte na dopasowaniu leksykalnym (słowo w słowo) zawodzą w sytuacjach, gdy użytkownik stosuje synonimy lub opisy zamiast dokładnych nazw (np. szukając informacji o „filmowcu”, system nie znajdzie dokumentu zawierającego jedynie słowo „reżyser”). Projekt odpowiada na ten problem poprzez zastosowanie osadzeń wektorowych (embeddings), które mapują znaczenie słów i całych dokumentów do wielowymiarowej przestrzeni geometrycznej.

**Zakres i ograniczenia:**
- **Dane:** System operuje na korpusie 1000 artykułów informacyjnych (zbiór Kaggle).
- **Obsługa języka:** Obecna wersja jest zoptymalizowana pod kątem języka angielskiego (ze względu na specyfikę użytych modeli i lematyzatorów).
- **Zastosowanie:** System może służyć jako prototyp silnika wyszukiwania dla baz wiedzy, archiwów prasowych lub systemów rekomendacji treści.

## 2. Przedstawienie rozszerzenia zadania 

W ramach rozszerzenia funkcjonalności zaimplementowano **system dopasowania wyszukiwania na podstawie preferencji użytkownika**, obejmujący:

- **Automatyczną kategoryzację dokumentów:** Każdy dokument w bazie jest analizowany pod kątem przynależności do jednej z sześciu klas: *Naukowe/Medyczne, Marketingowe/Biznesowe, Polityczne/Prawne, Rozrywka/Kultura, Sport, Ogólne/Informacyjne*.
- **Filtrowanie tematyczne:** Użytkownik może zawęzić wyniki wyszukiwania do konkretnej dziedziny, co eliminuje szum informacyjny (np. wyszukiwanie słowa "crash" tylko w kategorii "Sport" zamiast "Biznes").
- **Inteligentne punktowanie (Weighted Scoring):** Zamiast prostych flag, zastosowano system wagowy zliczający wystąpienia słów kluczowych charakterystycznych dla danej dziedziny, co pozwala na precyzyjne przypisanie kategorii nawet w dokumentach o mieszanej tematyce.

## 3. Wprowadzenie teoretyczne

**NLP (Natural Language Processing):**
Przetwarzanie języka naturalnego to dziedzina sztucznej inteligencji zajmująca się interakcjami między komputerami a językiem ludzkim. Kluczowym wyzwaniem NLP jest przekształcenie nieustrukturyzowanego tekstu w formę zrozumiałą dla maszyny — wektory liczbowe.

**Metody reprezentacji tekstu:**
1.  **TF-IDF (Term Frequency-Inverse Document Frequency):** Metoda statystyczna określająca wagę słowa w dokumencie w odniesieniu do całego korpusu. Skuteczna przy wyszukiwaniu unikalnych nazw własnych, ale niezdolna do rozpoznania kontekstu.
2.  **Doc2Vec:** Rozszerzenie modelu Word2Vec, pozwalające na naukę wektorów reprezentujących całe dokumenty. Model ten „uczy się” sąsiedztwa słów, dzięki czemu dokumenty o podobnym znaczeniu znajdują się blisko siebie w przestrzeni wektorowej, niezależnie od użytych konkretnych słów.

**Wyzwania:**
- **Dwuznaczność semantyczna:** To samo słowo może mieć różne znaczenia (np. "court" jako sąd lub boisko).
- **Wydajność:** Modele wektorowe wymagają intensywnych obliczeń podczas fazy treningu, jednak oferują bardzo szybki czas odpowiedzi podczas wyszukiwania dzięki zastosowaniu podobieństwa cosinusowego.

## 4. Opis realizacji technicznej

**Technologie:**
- **Gensim:** Wykorzystany do implementacji modelu Doc2Vec.
- **Scikit-learn:** Użyty do budowy macierzy TF-IDF oraz obliczania podobieństwa cosinusowego.
- **NLTK:** Biblioteka służąca do preprocessingu: tokenizacji, usuwania słów funkcyjnych (stopwords) oraz lematyzacji (sprowadzania słów do formy bazowej).
- **Streamlit:** Framework wykorzystany do budowy interaktywnego interfejsu webowego.

**Przetwarzanie danych:**
1.  **Czyszczenie:** Usuwanie adresów URL, e-maili oraz znaków specjalnych.
2.  **Lematyzacja:** Zamiana form odmienionych (np. "running", "ran") na bazowe ("run"), co pozwala na redukcję wymiarowości słownika.
3.  **Filtracja:** Zachowanie słów o długości $\ge 2$ znaków, aby nie tracić skrótów takich jak "US" czy "VP".

**Architektura i Parametry:**
- **Doc2Vec:** Wybrano algorytm **PV-DBOW** (`dm=0`) z włączonym treningiem wektorów słów (`dbow_words=1`). 
  - Rozmiar wektora: 300.
  - Liczba epok: 200.
- **TF-IDF:** Zastosowano n-gramy (1, 2) oraz wagowanie sublinearne, co pozwoliło lepiej oceniać złożone frazy.

## 5. Eksperymenty i Metryki Oceny

**Wyniki Eksperymentów:**
Przeprowadzono serię testów (Grid Search) na „Złotym Zbiorze Zapytań”, porównując średnią rangę dokumentu oczekiwanego. Wyniki wykazały, że Doc2Vec radzi sobie o 70% lepiej od TF-IDF przy zapytaniach opisowych (niezawierających słów kluczowych z tekstu).

**Zastosowane Metryki:**
W projekcie zastosowano następujące metryki i metody oceny:

1.  **Podobieństwo cosinusowe (Cosine Similarity):** Główna miara używana do obliczania stopnia dopasowania dokumentów do zapytania. Wartości te są wyświetlane bezpośrednio w aplikacji przy wynikach wyszukiwania.
2.  **Ranga (Rank):** Pozycja oczekiwanego dokumentu w wynikach wyszukiwania. Wykorzystywana podczas testów na tzw. "Złotym Zbiorze Zapytań".
3.  **Średnia Ranga (Average Rank / Mean Rank):** Zbiorcza metryka użyta w procesie optymalizacji hiperparametrów (Grid Search). Pozwala ona ocenić, jak blisko szczytu listy wyników średnio znajduje się poprawny dokument.
4.  **Średnie Podobieństwo (Average Score):** Średnia wartość podobieństwa cosinusowego dla poprawnych wyników w testowanych zapytaniach.
5.  **Poprawa względna (%):** Informacja o "70% lepszym radzeniu sobie" modelu Doc2Vec odnosi się do porównania średniej rangi (mean rank) obu modeli na zapytaniach opisowych (niezawierających słów kluczowych).
6.  **Punktowanie wagowe (Weighted Scoring):** Zastosowane w systemie kategoryzacji dokumentów do precyzyjnego przypisywania kategorii na podstawie zliczania wystąpień słów kluczowych.

## 6. Instrukcja korzystania z programu

**Wymagania:**
- Python 3.9 lub nowszy.

**Instalacja:**
1. Rozpakuj projekt i przejdź do katalogu głównego.
2. Zainstaluj biblioteki:
   ```bash
   pip install -r requirements.txt
   ```

**Uruchomienie:**
Aby uruchomić interfejs graficzny, wpisz:
```bash
streamlit run app.py
```

**Opis funkcji:**
- **Wprowadzanie zapytania:** Pole tekstowe na środku ekranu.
- **Preferencje (Sidebar):** 
  - *Typ dokumentu:* Wybór kategorii (np. Sport).
  - *Maksymalna liczba wyników:* Suwak określający ilu kandydatów wyświetlić.
- **Zarządzanie plikami:** Możliwość wgrania własnego pliku `.txt` przez uploader. System automatycznie wykryje nowy plik i zaproponuje dotrenowanie modeli.

## 7. Przykłady użycia programu

W tej sekcji przedstawiono rzeczywiste przypadki użycia aplikacji. **Warto zaznaczyć, że aplikacja zawsze wyświetla wyniki dla obu modeli jednocześnie (TF-IDF oraz Doc2Vec) wraz z ich wartościami podobieństwa cosinusowego**, co pozwala na bezpośrednie porównanie skuteczności podejścia leksykalnego i semantycznego.

### Przykład 1: Test dopasowania słów kluczowych (Leksykalny)
**Dane wejściowe:**
- **Zapytanie:** `Cardi B middle school`
- **Kategoria:** `Wszystkie`

**Wynik działania aplikacji:**
```text
Wyniki wyszukiwania (Doc2Vec):
1. kaggle_50.txt (Podobieństwo: 0.924) - Cardi B Donates $100,000 To Her Old Middle School...
2. kaggle_600.txt (Podobieństwo: 0.637)
3. kaggle_487.txt (Podobieństwo: 0.634)

Wyniki wyszukiwania (TF-IDF):
1. kaggle_50.txt (Podobieństwo: 0.339) - Cardi B Donates $100,000 To Her Old Middle School...
2. kaggle_369.txt (Podobieństwo: 0.068)
3. kaggle_610.txt (Podobieństwo: 0.067)
```

---

### Przykład 2: Test semantyczny (Zrozumienie kontekstu)
**Dane wejściowe:**
- **Zapytanie:** `filmmaker ninja thyberg interview`
- **Kategoria:** `Rozrywka/Kultura`

**Wynik działania aplikacji:**
```text
Wyniki wyszukiwania (Doc2Vec):
1. kaggle_667.txt (Podobieństwo: 0.848) - The Porn Star Protagonist In 'Pleasure' Is A Woman... filmmaker Ninja Thyberg told HuffPost.
2. kaggle_670.txt (Podobieństwo: 0.652)
3. kaggle_517.txt (Podobieństwo: 0.573)

Wyniki wyszukiwania (TF-IDF):
1. kaggle_667.txt (Podobieństwo: 0.339) - The Porn Star Protagonist In 'Pleasure' Is A Woman... filmmaker Ninja Thyberg told HuffPost.
2. kaggle_567.txt (Podobieństwo: 0.107)
3. kaggle_57.txt (Podobieństwo: 0.067)
```

*Uwaga: Model poprawnie powiązał "filmmaker" z treścią dokumentu oraz "interview" z frazą "told HuffPost".*

---

### Przykład 3: Filtrowanie po preferencjach (Kategoria + Specyfika)
**Dane wejściowe:**
- **Zapytanie:** `college sports team tragedy`
- **Kategoria:** `Sport`

**Wynik działania aplikacji:**
```text
Wyniki wyszukiwania (Doc2Vec):
1. kaggle_999.txt (Podobieństwo: 0.509) - 9 Dead, 2 Injured In Fiery Crash Involving University Of The Southwest Golf Teams
2. kaggle_856.txt (Podobieństwo: 0.483)
3. kaggle_705.txt (Podobieństwo: 0.458)

Wyniki wyszukiwania (TF-IDF):
1. kaggle_856.txt (Podobieństwo: 0.096) - Team Ukraine Arrives Safely Ahead Of Prince Harry's Invictus Games: 'We Are Delighted' For Team Ukraine, the journey to get to the games ...
2. kaggle_26.txt (Podobieństwo: 0.077)
3. kaggle_999.txt (Podobieństwo: 0.073)
```
---

### Przykład 4: Brak dopasowania (Zawężenie merytoryczne)
**Dane wejściowe:**
- **Zapytanie:** `Omicron vaccination statistics`
- **Kategoria:** `Sport`

**Wynik działania aplikacji:**
```text
Wyniki wyszukiwania (Doc2Vec):
1. kaggle_549.txt (Podobieństwo: 0.404) - Phil Mickelson Grilled On If He Cares About Being Seen As A 'Saudi Stooge' One reporter asked the golf great if he was worried about being "seen as a tool of sportswashing?"
2. kaggle_235.txt (Podobieństwo: 0.366)
3. kaggle_434.txt (Podobieństwo: 0.321)

Wyniki wyszukiwania (TF-IDF):
1. kaggle_357.txt (Podobieństwo: 0.0)
```

*Dokument dotyczący szczepionek (kaggle_0.txt) znajduje się w kategorii "Naukowe/Medyczne", więc został odfiltrowany przez preferencje użytkownika.*

---