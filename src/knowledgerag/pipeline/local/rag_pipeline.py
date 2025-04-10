from typing import Callable


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
        context = self.vector_client.query(input_data)
        context = self.process_context(context)
        prompted = self.prompt.format(context)
        response = self.llm_client(prompted)  # already formatted output
        return response
