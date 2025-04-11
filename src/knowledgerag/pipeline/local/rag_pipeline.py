import json
from typing import Callable

from loguru import logger


class RAGpipe:
    def __init__(self, vector_client, llm_client, embedding_function: Callable, prompt: str) -> None:
        """_summary_

        Args:
            vector_client (_type_): _description_
            llm_client (_type_): _description_
            embedding_function (Callable): _description_
            prompt (str): _description_
        """
        self.vector_client = vector_client
        self.llm_client = llm_client
        self.embedding_function = embedding_function
        self.prompt = prompt

    def process_context(self, context: str, processing_function: Callable) -> str:
        """_summary_

        Args:
            context (str): _description_
            processing_function (Callable): _description_

        Returns:
            str: _description_
        """
        return processing_function(context)

    def __call__(self, input_data: str, *args, **kwds) -> str:
        """_summary_

        Args:
            input_data (str): _description_

        Returns:
            str: _description_
        """
        context = self.vector_client.query(input_data)  # embeddings and search type.
        context = self.process_context(context)
        prompted = self.prompt.format(context)
        response = self.llm_client(prompted)  # already formatted output
        return response


def docling_context(chunks: list[dict]) -> str:
    """_summary_

    Args:
        chunks (list[dict]): _description_

    Returns:
        str: _description_
    """
    context_parts = []
    for chunk in chunks:
        # Include section information if available
        try:
            headings = json.loads(chunk["headings"])
            if headings:
                context_parts.append(f"Section: {' > '.join(headings)}")
        except Exception as e:
            logger.error(e)

        # Add page reference
        if chunk["page"]:
            context_parts.append(f"Page {chunk['page']}:")

        # Add the content
        context_parts.append(chunk["text"])
        context_parts.append("-" * 40)
    return "\n".join(context_parts)


if __name__ == "__main__":
    import lancedb
    from lancedb.embeddings import get_registry

    from knowledgerag.genai.llm import GoogleGemini

    embedding_function = get_registry().get("sentence-transformers").create(name="all-mpnet-base-v2", device="cpu")
    vector_store_client = db = lancedb.connect("/home/jdiez/Downloads/scratch/docling_bis.db")
    embedding_function = embedding_function.generate_embeddings
    table = db.open_table("example_pdf")
    # table.create_index(metric="cosine")
    query = "What disease talks the article about?"
    context = (
        table.search(query)
        .distance_type("cosine")
        .limit(3)
        .to_pandas()
        .sort_values("_distance", ascending=True)
        .head(3)
    )
    print(context)
    context = docling_context(context.to_dict("records"))
    print(context)
    llm_client = GoogleGemini()
    prompt = f"You are an expert question answering system, I'll give you question and context and you'll return the answer. Query : {query} Contexts : {context}"
    result = llm_client(prompt)
    print(result)
