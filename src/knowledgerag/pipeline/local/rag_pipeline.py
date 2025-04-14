import json
from typing import Callable, Literal

from loguru import logger


class RAGpipe:
    def __init__(
        self,
        vector_client,
        llm_client,
        processing_function: Callable,
        embedding_function: Callable | None = None,
        prompt: str | None = None,
        query_type: Literal["vector", "fts", "hybrid"] | None = "hybrid",
    ) -> None:
        """_summary_

        Args:
            vector_client (_type_): _description_
            llm_client (_type_): _description_
            embedding_function (Callable): _description_
            prompt (str): _description_
        """
        self.vector_client = vector_client
        self.llm_client = llm_client
        self.processing_function = processing_function
        self.prompt = prompt
        self.query_type = query_type

    def process_context(self, context: str) -> str:
        """_summary_

        Args:
            context (str): _description_
            processing_function (Callable): _description_

        Returns:
            str: _description_
        """
        return self.processing_function(context)

    def __call__(self, query: str, collection_name: str, *args, **kwds) -> str:
        """_summary_

        Args:
            input_data (str): _description_

        Returns:
            str: _description_
        """
        match self.query_type:
            case "vector":
                context = self.vector_client.search(
                    query=query, collection_name=collection_name
                )  # embeddings and search type.
            case "hybrid":
                context = self.vector_client.hybrid_search(
                    query=query, collection_name=collection_name
                )  # embeddings and search type.
            case "fts":
                context = self.vector_client.search(
                    query=query, collection_name=collection_name, query_type="fts"
                )  # embeddings and search type.
            case _:
                raise ValueError
        context = self.process_context(context)
        prompted = self.prompt.format(query=query, context=context)
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
            context_parts.append(f"Page: {chunk['page']}")

        # Add the content
        context_parts.append(f"Text: {chunk["text"]}")
        context_parts.append("-" * 20)

    doc_context = "\n".join(context_parts)
    print(doc_context)
    return doc_context


if __name__ == "__main__":
    from knowledgerag import read_configuration
    from knowledgerag.database.lancedb_lib.lancedb_client import LancedbDatabase
    from knowledgerag.database.lancedb_lib.lancedb_common import LanceDbSettings
    from knowledgerag.genai.llm import GoogleGemini
    from knowledgerag.genai.prompt.prompt import SIMPLE_RAG_PROMPT

    configuration = read_configuration()
    pipe = configuration["pipeline"]["default"]
    db_uri = pipe["storage"]["parameters"]["uri"]
    collection_name = pipe["storage"]["parameters"]["collection"]
    client_settings = LanceDbSettings(uri=db_uri)
    with LancedbDatabase(client_settings) as client:
        llm = GoogleGemini()
        query = "What is a Domino container?"
        rag_pipeline = RAGpipe(
            vector_client=client,
            llm_client=llm,
            prompt=SIMPLE_RAG_PROMPT,
            processing_function=docling_context,
            query_type="hybrid",
        )
        res = rag_pipeline(query=query, collection_name=collection_name)
        print(res)
