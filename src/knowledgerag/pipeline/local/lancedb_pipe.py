from collections.abc import Generator
from itertools import chain
from pathlib import Path
from typing import Any, Callable

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from loguru import logger
from prefect import flow, task
from prefect_dask.task_runners import DaskTaskRunner

from knowledgerag import read_configuration
from knowledgerag.database.lancedb_lib.lancedb_client import LancedbDatabase
from knowledgerag.database.lancedb_lib.lancedb_common import CollectionSettings, LanceDbSettings
from knowledgerag.io.reader.reader import file_reader
from knowledgerag.pipeline.local.ingestion_pipeline import DocumentProcessor
from knowledgerag.utils.profilling import timeit


@task
def fetch_files(path: str, file_types: tuple = (".pdf")) -> Generator:
    """_summary_

    Args:
        path (str): _description_
        file_types (tuple, optional): _description_. Defaults to (".pdf").

    Returns:
        _type_: _description_

    Yields:
        Generator: _description_
    """
    res = file_reader(path=Path(path).resolve(), allowed_file_types=file_types)
    return res


@task
def extract(file_info: Generator) -> list[str]:
    """_summary_

    Args:
        file_info (Generator): _description_

    Returns:
        list[str]: _description_
    """
    f = [i.model_dump()["path"] for i in file_info]
    return f


@task
def get_document(processor: Callable, file_name: str):
    """_summary_

    Args:
        processor (Callable): _description_
        file_name (str): _description_

    Returns:
        _type_: _description_
    """
    return processor(file_name)


@task
def ingestion(
    documents: list[dict],
    db_path: Path | str,
    table_name: str,
    schema: Any,
) -> None:
    """_summary_

    Args:
        documents (list[Path]): _description_
        db_path (Path): _description_
        table_name (str): _description_
        schema (LanceModel): _description_
    """
    client = LancedbDatabase(LanceDbSettings(uri=db_path))
    if table_name not in client.list_collections():
        collection_settings = CollectionSettings(name=table_name, schema=schema, data=documents, mode="create")
        client.create_collection(collection_settings)
    else:
        client.add_records(collection_name=table_name, data=documents)


@task
def index_collection(db_path: str, table_name: str) -> None:
    """_summary_

    Args:
        db_path (str): _description_
        table_name (str): _description_
    """
    db = LancedbDatabase(LanceDbSettings(uri=db_path))
    table = db.client.open_table(table_name)
    indices = [i.name for i in table.list_indices()]
    if "vector_idx" not in indices:
        try:
            table.create_index("cosine")
        except RuntimeError as e:
            logger.error(e)
            pass
    if "text_idx" not in indices:
        try:
            table.create_fts_index("text", use_tantivy=False)
        except RuntimeError as e:
            logger.error(e)
            pass


@flow(task_runner=DaskTaskRunner(), log_prints=True)
def rag_pipe(directory: str, processor: Callable, db_uri: str, db_table_name: str, schema: Any) -> None:
    """_summary_

    Args:
        directory (str): _description_
        processor (Callable): _description_
        db_uri (str): _description_
        db_table_name (str): _description_
        schema (Any): _description_
    """

    dirs = fetch_files.map([
        directory,
    ])
    logger.info(dirs)

    files = list(chain(*extract.map(dirs).result()))

    # metadata ingestion
    # ingestion(files, db_path=db_uri, table_name=f'{db_table_name}_metadata')

    results = []
    for file_name in files:
        res = get_document.submit(processor=processor, file_name=file_name)
        results.append(res)
    for f in results:
        res = f.result()
        ingestion(res, db_path=db_uri, table_name=db_table_name, schema=schema)
    index_collection(db_path=db_uri, table_name=db_table_name)


@timeit
def lancedb_standard_pipeline(directory):
    """_summary_

    Args:
        directory (_type_): _description_

    Returns:
        _type_: _description_
    """
    configuration = read_configuration()
    pipe = configuration["pipeline"]["default"]

    tokenizer = pipe["processing"]["tokenizer"]

    processor = DocumentProcessor(
        tokenizer=tokenizer,
    )

    embedding_model = pipe["processing"]["embedding"]
    db_uri = pipe["storage"]["parameters"]["uri"]
    db_table_name = pipe["storage"]["parameters"]["collection"]
    device = pipe["processing"]["device"]

    provider, model = embedding_model.split("/")
    embedder = get_registry().get(provider).create(name=model, device=device)

    class DbTableHybrid(LanceModel):
        file_name: str
        path: str
        text: str = embedder.SourceField()
        vector: Vector(embedder.ndims()) = embedder.VectorField()
        headings: str
        page_info: str
        content_type: str

    result = rag_pipe(
        directory=directory, processor=processor, db_uri=db_uri, db_table_name=db_table_name, schema=DbTableHybrid
    )

    return result


if __name__ == "__main__":
    import sys

    directory = sys.argv[1]
    lancedb_standard_pipeline(directory)
