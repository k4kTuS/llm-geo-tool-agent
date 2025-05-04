from sentence_transformers import SentenceTransformer

_encoder_instance = None

def get_encoder():
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = SentenceTransformer("intfloat/multilingual-e5-large")
    return _encoder_instance