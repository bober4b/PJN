from service.document_service import DocumentService
from service.model_service import ModelService

def retrain_models(doc_service: DocumentService) -> ModelService:
    print("Wczytywanie dokumentów z katalogu...")
    documents = doc_service.load_documents()

    print(f"Łącznie dokumentów: {len(documents)}")

    model_service = ModelService(documents)
    model_service.train_tfidf()
    model_service.train_doc2vec()

    return model_service


def load_models(doc_service: DocumentService) -> ModelService:
    documents = doc_service.load_documents()
    print(f"Łącznie dokumentów: {len(documents)}")

    model_service = ModelService(documents)
    model_service.load_tfidf()
    model_service.load_doc2vec()

    return model_service


def main():
    doc_service = DocumentService()

    # ===== START =====
    if doc_service.has_changes():
        inp = input(
            "Znaleziono nowe lub zmienione dokumenty.\n"
            "Czy chcesz ponownie wytrenować modele? (TAK/NIE): "
        ).strip().lower()

        if inp == "tak":
            model_service = retrain_models(doc_service)
        else:
            model_service = load_models(doc_service)
    else:
        model_service = load_models(doc_service)

    # ===== MENU =====
    while True:
        print("\n--- MENU ---")
        print("1. Wyszukaj dokumenty (TF-IDF)")
        print("2. Wyszukaj dokumenty (Doc2Vec)")
        print("3. Sprawdź zmiany w dokumentach")
        print("4. Koniec")

        option = input("Opcja: ").strip()

        match option:
            case "1":
                query = input("Wprowadź zapytanie: ")
                results = model_service.search_tfidf(query, top_n=5)

                print("\nWyniki (TF-IDF):")
                for name, score in results:
                    print(f"{name} | similarity={score}")

            case "2":
                query = input("Wprowadź zapytanie: ")
                results = model_service.search_doc2vec(query, top_n=5)

                print("\nWyniki (Doc2Vec):")
                for name, score in results:
                    print(f"{name} | similarity={score}")

            case "3":
                if doc_service.has_changes():
                    inp = input(
                        "Wykryto zmiany w dokumentach.\n"
                        "Czy chcesz ponownie wytrenować modele? (TAK/NIE): "
                    ).strip().lower()

                    if inp == "tak":
                        model_service = retrain_models(doc_service)
                    else:
                        print("Pominięto trenowanie.")
                else:
                    print("Brak zmian w dokumentach.")

            case "4":
                print("Koniec programu.")
                break

            case _:
                print("Nieznana opcja. Wybierz 1–4.")


if __name__ == "__main__":
    main()
