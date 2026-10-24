
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once per application process."""
    return SentenceTransformer(settings.embedding_model)


def generate_embeddings(
    texts: list[str],
) -> tuple[list[list[float]], int]:
    """Generate normalized embeddings for a list of text chunks."""

    if not texts:
        return [], 0

    model = get_embedding_model()

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = vectors.tolist()
    dimension = len(embeddings[0])

    return embeddings, dimension
