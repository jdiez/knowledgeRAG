from loguru import logger
from sentence_transformers import SentenceTransformer

MODEL_CHOICES = ["all-MiniLM-L6-v2", "thenlper/gte-base"]


class STransformer:
    """ """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """ """
        self.model_name = self.__pickup_model(model_name)
        self.model = SentenceTransformer(self.model_name)

    def __pickup_model(self, model_name: str) -> str:
        """ """
        used = model_name if model_name in MODEL_CHOICES else MODEL_CHOICES[0]
        if used != model_name:
            logger.info(f"Model {model_name} not supported. Instead {MODEL_CHOICES[0]} will be used.")
        return used

    def __call__(self, sentences: list[str], *args, **kwds):
        """ """
        return self.model.encode(sentences)
