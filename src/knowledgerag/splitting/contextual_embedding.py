"""source:
https://medium.com/kx-systems/late-chunking-vs-contextual-retrieval-the-math-behind-rags-context-problem-d5a26b9bbd38
https://github.com/jina-ai/late-chunking/blob/main/explanatory_contextual_retrieval.py
"""

import asyncio
import os

import pandas as pd
import requests


class LateChunkingEmbeddingAPI:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        jina_api_key: str | None = None,
        model: str = "jina-embeddings-v3",
        task: str = "text-matching",
        dimensions: int = 1024,
        embedding_type: str = "float",
    ) -> None:
        """_summary_

        Args:
            jina_api_key (str): _description_
        """
        self.JINA_API_KEY = self._get_jina_api_key(key=jina_api_key)
        self.model = model
        self.task = task
        self.dimensions = dimensions
        self.embedding_type = embedding_type

    def _get_jina_api_key(self, key: str | None) -> str | None:
        """_summary_

        Args:
            key (str | None): _description_

        Returns:
            str | None: _description_
        """
        return key if key else os.environ.get("JINA_API_KEY", None)

    def get_embeddings(
        self,
        chunks: list[str],
        late_chunking: bool = False,
        contexts: list[str] | None = None,
        timeout: int | float = 5,
    ) -> list[float]:
        """_summary_

        Args:
            chunks (list[str]): _description_
            late_chunking (bool, optional): _description_. Defaults to False.
            contexts (list[str] | None, optional): _description_. Defaults to None.

        Returns:
            list[float]: _description_
        """
        url = "https://api.jina.ai/v1/embeddings"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.JINA_API_KEY}"}

        # If using contextual chunking, combine contexts with chunks
        input_texts = [f"{ctx} {chunk}" for ctx, chunk in zip(contexts, chunks)] if contexts else chunks

        data = {
            "model": self.model,
            "task": self.task,
            "dimensions": self.dimensions,
            "late_chunking": late_chunking,
            "embedding_type": self.embedding_type,
            "input": input_texts,
        }

        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        return [item["embedding"] for item in response.json()["data"]]

    def __call__(
        self, chunks: list[str], late_chunking: bool = False, contexts: list[str] | None = None
    ) -> list[float]:
        """_summary_

        Args:
            chunks (list[str]): _description_
            late_chunking (bool, optional): _description_. Defaults to False.
            contexts (list[str] | None, optional): _description_. Defaults to None.

        Returns:
            list[float]: _description_
        """
        return self.get_embeddings(chunks, late_chunking, contexts)


async def generate_contexts(
    document: str, chunks: list[str], client, model: str = "gpt-4o", temperature: float = 0.3, max_tokens: int = 100
):
    async def process_chunk(chunk):
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a brief context explaining how this chunk relates to the full document.",
                },
                {
                    "role": "user",
                    "content": f"<document> \n{document} \n</document> \nHere is the chunk we want to situate within the whole document \n<chunk> \n{chunk} \n</chunk> \nPlease give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.",
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        context = response.choices[0].message.content
        return f"{context} {chunk}"

    # Process all chunks concurrently
    contextual_chunks = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])

    return contextual_chunks


async def main(document: str, chunks: list[str], client) -> pd.DataFrame:
    """_summary_

    Returns:
        pd.DataFrame: _description_
    """
    get_embeddings = LateChunkingEmbeddingAPI()
    df = pd.DataFrame({"text": chunks})
    contexts = await generate_contexts(document, chunks, client)

    df["naive_embedding"] = get_embeddings(chunks, late_chunking=False)
    df["late_embedding"] = get_embeddings(chunks, late_chunking=True)
    df["contextual_embedding"] = get_embeddings(chunks, late_chunking=False, contexts=contexts)
    df["context"] = contexts
    return df


if __name__ == "__main__":
    result = asyncio.run(main())
