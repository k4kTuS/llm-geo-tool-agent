from sentence_transformers import SentenceTransformer

_encoder_instance = None

def get_encoder():
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _encoder_instance