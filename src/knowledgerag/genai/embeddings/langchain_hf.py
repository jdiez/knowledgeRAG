from langchain_huggingface import HuggingFaceEmbeddings

models = (
    "all-mpnet-base-v2",
    "multi-qa-mpnet-base-dot-v1",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v2",
    "multi-qa-distilbert-cos-v1" "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
)


def get_embeddings_from_hf(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """AI is creating summary for get_embeddings_from_hf

    Args:
        model_name (str, optional): [description]. Defaults to "all-MiniLM-L6-v2".

    Returns:
        HuggingFaceEmbeddings: [description]
    """
    return HuggingFaceEmbeddings(model_name=model_name)


EMBEDDING = get_embeddings_from_hf()


def embed_query(data: str, embeddings: HuggingFaceEmbeddings = EMBEDDING) -> list[float]:
    """AI is creating summary for embed_query

    Args:
        data (str): [description]

    Returns:
        list[float]: [description]
    """
    return embeddings.embed_query(data)


def embed_documents(data: list[str], embeddings: HuggingFaceEmbeddings = EMBEDDING) -> list[list[float]]:
    """AI is creating summary for embed_documents

    Args:
        data (list[str]): [description]

    Returns:
        list[list[float]]: [description]
    """
    return embeddings.embed_documents(data)
