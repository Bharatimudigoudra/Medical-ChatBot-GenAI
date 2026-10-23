from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.config import settings


@lru_cache()
def _store():
    # Built on first question, so the server can start (and /api/health can report) without keys.
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    return PineconeVectorStore.from_existing_index(index_name=settings.pinecone_index, embedding=embeddings)


def search_kb(query, top_k=5):
    """Query the existing MiniLM 'medicalbot' index; returns tutorial-shaped KB hits."""
    results = _store().similarity_search_with_score(query, k=top_k)
    hits = []
    for doc, score in results:
        if score < settings.relevance_threshold:
            continue
        page = doc.metadata.get('page')
        hits.append({
            'page': int(page) + 1 if page is not None else '?',  # index stores 0-based pages; show 1-based
            'text': doc.page_content,
            'score': round(float(score), 3),
            'source': 'Gale Encyclopedia (2002)',
        })
    return hits
