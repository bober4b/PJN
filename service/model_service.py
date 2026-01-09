import os
import json
import joblib
import numpy as np

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from service.document_service import DocumentService


class ModelService:
    DOC2VEC_MODEL_PATH = "data/doc2vec.model"
    DOC2VEC_VECTORS_PATH = "data/doc2vec_vectors.json"
    TFIDF_MODEL_PATH = "data/tfidf_model.pkl"

    def __init__(self, documents):
        self.documents = documents
        self.document_names = [d.name for d in documents]

        self.doc2vec_model = None
        self.doc_vectors = None

        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        self.load_doc2vec()
        self.load_tfidf()

    # =========================
    # Doc2Vec
    # =========================

    def train_doc2vec(self):
        tagged_docs = [
            TaggedDocument(
                words=doc.content.split(),
                tags=[doc.name]
            )
            for doc in self.documents
        ]

        self.doc2vec_model = Doc2Vec(
            vector_size=300,
            window=10,
            min_count=2,
            workers=4,
            epochs=120,
            dm=1,
            negative=10
        )

        self.doc2vec_model.build_vocab(tagged_docs)
        self.doc2vec_model.train(
            tagged_docs,
            total_examples=self.doc2vec_model.corpus_count,
            epochs=self.doc2vec_model.epochs
        )

        os.makedirs("data", exist_ok=True)
        self.doc2vec_model.save(self.DOC2VEC_MODEL_PATH)

        self._save_doc_vectors()
        print("Model Doc2Vec wytrenowany i zapisany.")

    def load_doc2vec(self):
        if not os.path.exists(self.DOC2VEC_MODEL_PATH):
            print("Brak zapisanego modelu Doc2Vec — rozpoczynam trenowanie...")
            self.train_doc2vec()
            return

        print("Wczytywanie modelu Doc2Vec...")
        self.doc2vec_model = Doc2Vec.load(self.DOC2VEC_MODEL_PATH)

        if os.path.exists(self.DOC2VEC_VECTORS_PATH):
            self._load_doc_vectors()
        else:
            print("Brak zapisanych wektorów dokumentów — generuję ponownie...")
            self._save_doc_vectors()

    def _save_doc_vectors(self):
        self.doc_vectors = {
            doc.name: self.doc2vec_model.dv[doc.name].tolist()
            for doc in self.documents
        }

        with open(self.DOC2VEC_VECTORS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.doc_vectors, f)

    def _load_doc_vectors(self):
        with open(self.DOC2VEC_VECTORS_PATH, "r", encoding="utf-8") as f:
            self.doc_vectors = json.load(f)

    def search_doc2vec(self, query: str, top_n: int = 5):
        if self.doc2vec_model is None:
            raise RuntimeError("Doc2Vec nie jest załadowany.")

        query_processed = DocumentService.preprocess_text(query)
        query_tokens = query_processed.split()

        query_vector = self.doc2vec_model.infer_vector(query_tokens)

        results = []
        for name, vec in self.doc_vectors.items():
            sim = cosine_similarity(
                [query_vector],
                [vec]
            )[0][0]
            results.append((name, round(float(sim), 4)))

        return sorted(results, key=lambda x: -x[1])[:top_n]

    # =========================
    # TF-IDF
    # =========================

    def train_tfidf(self):
        """
        TF-IDF na tekstach JUŻ po preprocessingu.
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=False,
            stop_words=None
        )

        contents = [d.content for d in self.documents]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)

        os.makedirs("data", exist_ok=True)
        joblib.dump(
            (self.tfidf_vectorizer, self.tfidf_matrix, self.document_names),
            self.TFIDF_MODEL_PATH
        )

        print("Model TF-IDF wytrenowany i zapisany.")

    def load_tfidf(self):
        if not os.path.exists(self.TFIDF_MODEL_PATH):
            print("Brak zapisanego modelu TF-IDF — rozpoczynam trenowanie...")
            self.train_tfidf()
            return

        self.tfidf_vectorizer, self.tfidf_matrix, self.document_names = joblib.load(
            self.TFIDF_MODEL_PATH
        )
        print("Model TF-IDF wczytany.")

    def search_tfidf(self, query: str, top_n: int = 5):
        if self.tfidf_vectorizer is None:
            raise RuntimeError("TF-IDF nie jest załadowany.")

        query_processed = DocumentService.preprocess_text(query)
        query_vector = self.tfidf_vectorizer.transform([query_processed])

        sims = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_n]

        return [
            (self.document_names[i], round(float(sims[i]), 4))
            for i in top_idx
        ]
