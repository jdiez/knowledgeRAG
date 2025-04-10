import json
import os
from abc import abstractmethod
from typing import Any, Callable

from loguru import logger
from openai import AzureOpenAI


class CredentialsValueError(Exception):
    def __init__(self, message: str, error_code: int):
        """ """
        self.message = message
        self.error_code = error_code
        super().__init__(f"Message: '{self.message}', with error_code: '{self.error_code}'.")


class LLM:
    """Wrapper of allowed models to perform desired task under the same wrapper method.

    Returns:
        _type_: _description_
    """

    def __init__(self, credentials: dict[str, str], model: str, **kwargs) -> None:
        """_summary_

        Args:
            credentials (dict[str, str]): _description_
            model (str): _description_

        Returns:
            _type_: _description_
        """
        self.credentials = credentials
        self.model = model
        self.args = kwargs

    @abstractmethod
    def _validate_args(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        pass

    @abstractmethod
    def _setup_client(
        self,
    ) -> Callable:
        """_summary_

        Returns:
            Callable: _description_
        """
        pass

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        """_summary_

        Args:
            prompt (str): _description_

        Returns:
            str: _description_
        """
        pass


class AzureOpenAiLocal(LLM):
    def __init__(self, credentials: dict[str, str]) -> None:
        """ """
        self.credentials = credentials

    def _validate_credentials(self) -> dict[str, Any]:
        credentials = {i: os.getenv(i, None) for i in self.credentials}
        result = all((j is not None) for j in credentials.values)
        if not result:
            raise CredentialsValueError(message=json.dumps(credentials), error_code=1)
        return result, credentials

    def _setup_client(self) -> Callable:
        try:
            client = AzureOpenAI(**{i: os.getenv(i, None) for i in self.credentials})
        except Exception as e:
            logger.error(e)
        return client

    def __call__(self, input_text: str, system_message: str | None = None) -> str:
        """_summary_

        Args:
            prompt (str): _description_

        Returns:
            str: _description_
        """
        system_message = system_message if not system_message else self.args.get("system_message")
        response = self.client.chat.completions.create(**{
            k: v
            for k, v in {
                "model": self.credentials["deployment"],
                "temperature": self.args.get("temperature", None),
                "max_tokens": self.args.get("max_tokens", None),
                "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": input_text}],
            }
            if v is not None
        })
        generated_text = response.choices[0].message.content
        return generated_text
