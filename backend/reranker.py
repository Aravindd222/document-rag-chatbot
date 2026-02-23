from sentence_transformers import CrossEncoder

# Cross-encoder model (this is the reranker)
_model = None

def get_reranker():
    global _model
    if _model is None:
        _model = CrossEncoder("BAAI/bge-reranker-large")
        #want 4x faster but less accuracy that above
        #_model = CrossEncoder("BAAI/bge-reranker-base")
    return _model


def rerank(query: str, documents: list[str], top_n: int = 4):
    """
    Reorders retrieved chunks using cross-encoder scoring.
    Returns the best top_n documents.
    """
    model = get_reranker()

    pairs = [(query, doc) for doc in documents]

    scores = model.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:top_n]]