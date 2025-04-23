"""
https://jina.ai/es/news/what-late-chunking-really-is-and-what-its-not-part-ii/
https://blog.lancedb.com/late-chunking-aka-chunked-pooling-2/
"""

from typing import Callable, Literal

from transformers import AutoModel, AutoTokenizer


class LateChunking:
    """_summary_"""

    def __init__(
        self,
        model: str = "jinaai/jina-colbert-v2",
        tokenizer: str | None = None,
    ) -> None:
        """_summary_

        Args:
            model (str, optional): _description_. Defaults to 'jinaai/jina-colbert-v2'.
            tokenizer (str | None, optional): _description_. Defaults to None.
        """
        if not tokenizer:
            tokenizer = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True)

    def _chunk_by_sentences(self, input_text: str, tokenizer: Callable) -> tuple[list[str], list[tuple[int, int]]]:
        """_summary_

        Args:
            input_text (str): _description_
            tokenizer (Callable): _description_

        Returns:
            tuple[list[str], list[tuple[int, int]]]: _description_
        """
        inputs = self.tokenizer(input_text, return_tensors="pt", return_offsets_mapping=True)
        punctuation_mark_id = self.tokenizer.convert_tokens_to_ids(".")
        sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        token_offsets = inputs["offset_mapping"][0]
        token_ids = inputs["input_ids"][0]
        chunk_positions = [
            (i, int(start + 1))
            for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
            if token_id == punctuation_mark_id
            and (token_offsets[i + 1][0] - token_offsets[i][1] > 0 or token_ids[i + 1] == sep_id)
        ]
        chunks = [input_text[x[1] : y[1]] for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)]
        span_annotations = [(x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)]
        return chunks, span_annotations

    def _late_chunking(
        self, model_output: list[str], span_annotation: list[tuple[int, int]], max_length=None
    ) -> list[list[float]]:
        token_embeddings = model_output[0]
        outputs = []
        for embeddings, annotations in zip(token_embeddings, span_annotation):
            if max_length is not None:  # remove annotations which go beyond the max-length of the model
                annotations = [
                    (start, min(end, max_length - 1)) for (start, end) in annotations if start < (max_length - 1)
                ]
            pooled_embeddings = [
                embeddings[start:end].sum(dim=0) / (end - start) for start, end in annotations if (end - start) >= 1
            ]
            pooled_embeddings = [embedding.detach().cpu().numpy() for embedding in pooled_embeddings]
            outputs.append(pooled_embeddings)

        return outputs

    def _chunking(self, data: str) -> tuple[list[str], list[tuple[int, int]]]:
        """_summary_

        Args:
            self (_type_): _description_
            list (_type_): _description_

        Returns:
            _type_: _description_
        """
        chunks, span_annotations = self._chunk_by_sentences(data, tokenizer=self.tokenizer)
        return chunks, span_annotations

    def _late_chunk_embedding(self, data: str) -> list[dict[str, str | list[float]]]:
        """_summary_

        Args:
            data (str): _description_

        Returns:
            list[dict[str, str | list[float]]]: _description_
        """
        chunks, span_annotations = self._chunking(data)
        inputs = self.tokenizer(data, return_tensors="pt")
        model_output = self.model(**inputs)
        late_chunk_embeddings = self._late_chunking(model_output, [span_annotations])[0]
        late_chunk_data = []
        for index, chunk in enumerate(chunks):
            late_chunk_data.append({
                "text": chunk,
                "vector": late_chunk_embeddings[index],
            })
        return late_chunk_data

    def _vanilla_chunk_embedding(self, data: str) -> list[dict[str, str | list[float]]]:
        """_summary_

        Args:
            data (str): _description_

        Returns:
            list[dict[str, str | list[float]]]: _description_
        """
        chunks, _ = self._chunking(data)
        vanilla_chunk_embeddings = self.model.encode(chunks)

        vanilla_data = []
        for index, chunk in enumerate(chunks):
            vanilla_data.append({
                "text": chunk,
                "vector": vanilla_chunk_embeddings[index],
            })
        return vanilla_data

    def __call__(self, data: str, mode: Literal["late", "vanilla"] = "late") -> list[dict[str, str | list[float]]]:
        """_summary_

        Args:
            data (str): _description_
            mode (Literal[&#39;late&#39;, &#39;vanilla&#39;], optional): _description_. Defaults to 'late'.

        Raises:
            ValueError: _description_

        Returns:
            list[dict[str, str | list[float]]]: _description_
        """
        match mode:
            case "late":
                result = self._late_chunk_embedding(data)
            case "vanilla":
                result = self._vanilla_chunk_embedding(data)
            case _:
                raise ValueError
        return result
