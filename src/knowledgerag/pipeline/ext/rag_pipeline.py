from typing import Callable


class RAGpipe:
    def __init__(self, vector_client,
                 llm_client,
                 embedding_function: Callable,
                 prompt: str):
        self.vector_client = vector_client
        self.llm_client = llm_client
        self.embedding_function = embedding_function
        self.prompt = prompt

    def process_context(self, context: str, processing_function: Callable) -> str:
        return processing_function(context)

    def __call__(self, input: str, *args, **kwds):
        context = self.vector_client.query(input)
        context = self.process_context(context)
        prompted = self.prompt.format(context)
        response = self.llm_client(prompted)    # already formated output
        return response