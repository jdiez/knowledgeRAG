import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import click
import lancedb
import pandas as pd
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from knowledgerag import read_configuration
from knowledgerag.database.lancedb_lib.lancedb_client import LancedbDatabase
from knowledgerag.database.lancedb_lib.lancedb_common import CollectionSettings, LanceDbSettings
from knowledgerag.io.reader.reader import file_reader, reporting_file_info
from knowledgerag.pipeline.local.ingestion_pipeline import DocumentProcessor


class RAGmetadataLance:
    def __init__(
        self,
        target_database: Path | str,
        metadata_table: str | None = "metadata",
        file_reader: Callable = file_reader,
        reporter: Callable = reporting_file_info,
    ) -> None:
        self.target_database = str(target_database)
        self.metadata_table = metadata_table
        self.file_reader = file_reader
        self.reporting_file_info = reporter
        self.db = lancedb.connect(uri=self.target_database)

    def get_new_files_metadata(self, input_directory: str) -> pd.DataFrame | None:
        """_summary_

        Args:
            input_directory (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        res = asyncio.run(
            self.describe_files(filenames=self.file_reader(input_directory), reporter=self.reporting_file_info)
        )
        result = pd.DataFrame.from_dict([r.model_dump() for r in res])
        result = result if not result.empty else None
        return result

    def get_existing_files_hash(self) -> pd.DataFrame | None:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        try:
            tbl = self.db.open_table(self.metadata_table)
        except ValueError:
            df = None
        else:
            df = tbl.to_pandas()
        return df

    @staticmethod
    def compare_hashes(current: list[str], new: list[str]) -> list[str]:
        """_summary_

        Args:
            current (list[str]): _description_
            new (list[str]): _description_

        Returns:
            list[str]: _description_
        """
        return list(set(new).difference(set(current)))

    @staticmethod
    async def describe_files(filenames, reporter):
        """_summary_

        Args:
            filenames (_type_): _description_
            reporter (_type_): _description_

        Returns:
            _type_: _description_
        """
        loop = asyncio.get_running_loop()
        tasks = []
        with ProcessPoolExecutor() as executor:
            for filename in filenames:
                tasks.append(loop.run_in_executor(executor, reporter, filename))
            return list(await tqdm_asyncio.gather(*tasks))

    def ingest_new(self, data: pd.DataFrame) -> None:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
        """
        table_names = self.db.table_names()
        if self.metadata_table in table_names:
            table = self.db.open_table(self.metadata_table)
            table.add(data)
        else:
            self.db.create_table(name=self.metadata_table, data=data)

    def __call__(self, input_directory: str, ingestion: bool = False) -> pd.DataFrame | None:
        """_summary_

        Args:
            input_directory (str): _description_
            ingestion (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame | None: _description_
        """
        current = self.get_existing_files_hash()
        new = self.get_new_files_metadata(input_directory=input_directory)
        if new is None:
            result = new
        else:
            if current is not None:
                res = self.compare_hashes(current.hash_value.tolist(), new.hash_value.tolist())
                result = new[new.hash_value.isin(res)]
            else:
                result = new
            if ingestion:
                self.ingest_new(result)
        return result


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


def lancedb_standard_pipeline(filenames):
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

    # docling isn't threath safe, and has some parallel processing, so performed sequentially.
    total = len(filenames)
    for n, filename in enumerate(filenames):
        logger.info(f"Processing file {n} of {total}: {filename}.")
        data = processor(filename)
        logger.info(f"Ingesting file: {filename}")
        ingestion(documents=list(data), db_path=db_uri, table_name=db_table_name, schema=DbTableHybrid)
    logger.info(f"Indexing collection: {db_table_name}")
    index_collection(db_path=db_uri, table_name=db_table_name)


@click.command()
@click.option("--data-dir", type=click.Path(), default="/home/jdiez/Downloads/test", help="Path to the data directory")
def main(data_dir: str) -> None:
    configuration = read_configuration()
    pipe = configuration["pipeline"]["default"]
    db = pipe["storage"]["parameters"]["uri"]
    meta = pipe["storage"]["parameters"]["metadata"]
    rp = RAGmetadataLance(target_database=db, metadata_table=meta)
    res = rp(input_directory=data_dir, ingestion=True)
    filenames = res.path.tolist()
    if filenames:
        lancedb_standard_pipeline(filenames)
    else:
        logger.info("Nothing to be ingested.")


if __name__ == "__main__":
    main()
