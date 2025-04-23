from typing import Any, Literal

from pydantic import BaseModel
from yaml import safe_load
from zipp import Path


class Input(BaseModel):
    directory: str
    extensions: list[Literal[".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt", ".csv", ".md", ".html"]]


class Parameters(BaseModel):
    uri: str | Path
    collection: str
    metadata: str
    schema: str | None = None
    mode: str | None = None


class Storage(BaseModel):
    vectorbase: str
    parameters: Parameters


class Processing(BaseModel):
    tokenizer: str
    chunker: str
    embedding: str
    device: str


class Pipeline(BaseModel):
    name: str
    description: str
    input: Input
    processing: Processing
    storage: Storage


class Pipelines(BaseModel):
    default: Pipeline


class Config(BaseModel):
    llms: dict[str, Any]
    database: dict[str, Any]
    pipeline: Pipelines


if __name__ == "__main__":
    with open("config.yml") as tables_file:
        print(Config(**safe_load(tables_file)))
