from service.model_service import ModelService
from service.document_service import DocumentService


def get_updated_model_service(doc_service):
    # zawsze najpierw wczytaj dokumenty
    doc_service.load_documents()

    model_service = ModelService(doc_service.documents)

    # NAZWY MUSZĄ PASOWAĆ 1:1 do ModelService
    model_service.train_doc2vec()
    model_service.train_tfidf()

    return model_service


def get_model_service(doc_service):
    # zawsze najpierw wczytaj dokumenty
    doc_service.load_documents()

    model_service = ModelService(doc_service.documents)

    model_service.load_doc2vec()
    model_service.load_tfidf()

    return model_service
