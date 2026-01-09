import os
import json
import re
import nltk
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from model.document import Document


class DocumentService:
    """
    Odpowiada za:
    - wczytanie dokumentów z katalogu
    - wstępny preprocessing tekstu
    - wykrywanie zmian w plikach
    """

    DOCS_DIR_PATH = "documents"
    DOCS_STATUS_FILE = "data/docs_status.json"
    FILE_EXTENSIONS = (".txt",)

    # proste regexy do czyszczenia szumu
    _RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
    _RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
    _RE_HTML = re.compile(r"<[^>]+>")
    _RE_WS = re.compile(r"\s+")

    def __init__(self):
        self.documents: list[Document] = []
        # upewnij się, że NLTK ma potrzebne zasoby (raz na start serwisu)
        self._ensure_nltk_resources()

    # =========================
    # Publiczne API serwisu
    # =========================

    def load_documents(self) -> list[Document]:
        """
        Wczytuje dokumenty z katalogu documents/,
        wykonuje preprocessing i zwraca listę Document.
        """
        self.documents = []

        files = self._get_document_files()

        for file in files:
            path = os.path.join(self.DOCS_DIR_PATH, file)
            content = self._read_file(path)
            processed_content = self.preprocess_text(content)

            self.documents.append(
                Document(
                    name=file,
                    mod_date=os.path.getmtime(path),
                    content=processed_content
                )
            )

        self._save_files_status(files)
        return self.documents

    def has_changes(self) -> bool:
        """
        Sprawdza, czy pliki w katalogu documents/ uległy zmianie
        (dodane/usunięte/zmodyfikowane).
        """
        if not os.path.exists(self.DOCS_STATUS_FILE):
            return True

        with open(self.DOCS_STATUS_FILE, "r", encoding="utf-8") as f:
            old_status = json.load(f)

        current_files = self._get_document_files()

        if set(current_files) != set(old_status.keys()):
            return True

        for file in current_files:
            path = os.path.join(self.DOCS_DIR_PATH, file)
            if old_status[file] != os.path.getmtime(path):
                return True

        return False

    # =========================
    # Preprocessing
    # =========================

    @classmethod
    def preprocess_text(cls, text: str, *, return_tokens: bool = False) -> str | list[str]:
        """
        Tokenizacja, normalizacja, usunięcie stopwords i lematyzacja.
        Ten SAM preprocessing musi być używany dla dokumentów i zapytań.

        Domyślnie zwraca string (pod TF-IDF).
        return_tokens=True zwraca listę tokenów (pod embeddingi).
        """
        if not text:
            return [] if return_tokens else ""

        text = cls._basic_cleanup(text)

        tokenizer = cls._get_tokenizer()
        tokens = tokenizer.tokenize(text.lower())

        sw = cls._get_stopwords()
        tokens = [t for t in tokens if t not in sw]

        lemmatizer = cls._get_lemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

        if return_tokens:
            return tokens
        return " ".join(tokens)

    @classmethod
    def _basic_cleanup(cls, text: str) -> str:
        """
        Usuwa URL/e-maile/HTML, normalizuje whitespace.
        Nie usuwa znaków interpunkcyjnych 'na ślepo' — tokenizer i tak wybierze tokeny.
        """
        text = cls._RE_HTML.sub(" ", text)
        text = cls._RE_URL.sub(" ", text)
        text = cls._RE_EMAIL.sub(" ", text)
        text = cls._RE_WS.sub(" ", text)
        return text.strip()

    # =========================
    # NLTK resources + cache
    # =========================

    @staticmethod
    def _ensure_nltk_resources() -> None:
        """
        Minimalny zestaw zasobów potrzebny do:
        - stopwords (nltk.corpus.stopwords)
        - WordNet lemmatizer
        """
        required = [
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]
        for path, pkg in required:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_stopwords() -> set[str]:
        return set(stopwords.words("english"))

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_tokenizer() -> RegexpTokenizer:
        # 2+ litery, tylko A-Z (dla EN OK). Jeśli chcesz dopuścić apostrofy: r"[a-zA-Z]{2,}(?:'[a-zA-Z]+)?"
        return RegexpTokenizer(r"[a-zA-Z]{2,}")

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_lemmatizer() -> nltk.WordNetLemmatizer:
        return nltk.WordNetLemmatizer()

    # =========================
    # Metody pomocnicze (private)
    # =========================

    def _get_document_files(self) -> list[str]:
        if not os.path.exists(self.DOCS_DIR_PATH):
            return []

        return [
            f for f in os.listdir(self.DOCS_DIR_PATH)
            if f.endswith(self.FILE_EXTENSIONS)
        ]

    @staticmethod
    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _save_files_status(self, files: list[str]) -> None:
        os.makedirs(os.path.dirname(self.DOCS_STATUS_FILE), exist_ok=True)

        status = {}
        for file in files:
            path = os.path.join(self.DOCS_DIR_PATH, file)
            status[file] = os.path.getmtime(path)

        with open(self.DOCS_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status, f)
