import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import click
import lancedb
import pandas as pd
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from knowledgerag import read_configuration
from knowledgerag.database.lancedb_lib.lancedb_client import LancedbDatabase
from knowledgerag.database.lancedb_lib.lancedb_common import LanceDbSettings
from knowledgerag.io.reader.reader import file_reader, reporting_file_info
from knowledgerag.pipeline.local.ingestion_pipeline import DocumentProcessor
from knowledgerag.splitting.late_chunking import LateChunkingEmbedding


class RAGmetadataLance:
    def __init__(
        self,
        target_database: Path | str,
        metadata_table: str = "metadata",
        data_table: str = "data",
        file_reader: Callable = file_reader,
        reporter: Callable = reporting_file_info,
    ) -> None:
        self.target_database = str(target_database)
        self.metadata_table = metadata_table
        self.data_table = data_table
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

    def get_existing_files_hash(self, table: str) -> pd.DataFrame | None:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        try:
            tbl = self.db.open_table(table)
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

    def ingest_new(self, data: pd.DataFrame, table_name: str) -> None:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
        """
        table_names = self.db.table_names()
        if table_name in table_names:
            table = self.db.open_table(table_name)
            table.add(data)
        else:
            self.db.create_table(name=table_name, data=data)

    def overwrite(self, data: pd.DataFrame, table_name: str) -> None:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
        """
        try:
            self.db.create_table(name=table_name, data=data, mode="overwrite")
        except Exception as e:
            logger.error(e)

    def sync_data_metadata(self) -> None:
        """Be careful with it ..."""
        metadata_table = self.db.open_table(self.metadata_table).to_pandas()
        data_table = self.db.open_table(self.data_table).to_pandas()
        common_hashes = set(metadata_table.hash_value.tolist()).intersection(set(data_table.hash_value.tolist()))
        metadata_table_data = metadata_table[metadata_table.hash_value.isin(common_hashes)]
        self.overwrite(data=metadata_table_data, table_name=self.metadata_table)
        data_table_data = data_table[data_table.hash_value.isin(common_hashes)]
        self.overwrite(data=data_table_data, table_name=self.data_table)

    def __call__(self, input_directory: str, ingestion: bool = False) -> pd.DataFrame | None:
        """_summary_

        Args:
            input_directory (str): _description_
            ingestion (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame | None: _description_
        """
        current = self.get_existing_files_hash(table=self.metadata_table)
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
                self.ingest_new(data=result, table_name=self.metadata_table)
        return result


def ingestion(
    documents: list[dict],
    db_path: Path | str,
    table_name: str,
    schema: Any | None = None,
) -> None:
    """_summary_

    Args:
        documents (list[Path]): _description_
        db_path (Path): _description_
        table_name (str): _description_
        schema (LanceModel): _description_
    """
    # client = LancedbDatabase(LanceDbSettings(uri=db_path))
    conn = lancedb.connect(db_path)
    # if table_name not in client.list_collections():
    if table_name not in conn.table_names():
        # collection_settings = CollectionSettings(name=table_name, data=documents, mode="create")
        # client.create_collection(collection_settings)
        table = conn.create_table(name=table_name, data=documents)
    else:
        # client.add_records(collection_name=table_name, data=documents)
        table = conn.open_table(table_name)
        table.add(documents)


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

    # embedding_model = pipe["processing"]["embedding"]
    db_uri = pipe["storage"]["parameters"]["uri"]
    db_table_name = pipe["storage"]["parameters"]["collection"]
    # device = pipe["processing"]["device"]

    # provider, model = embedding_model.split("/")
    # embedder = get_registry().get(provider).create(name=model, device=device)

    # class DbTableHybrid(LanceModel):
    #     file_name: str
    #     path: str
    #     text: str = embedder.SourceField()
    #     vector: Vector(embedder.ndims()) = embedder.VectorField()
    #     headings: str
    #     page_info: str
    #     content_type: str

    # docling isn't threath safe, and has some parallel processing, so performed sequentially.
    total = len(filenames)
    lt = LateChunkingEmbedding()
    for n, filename in enumerate(filenames):
        logger.info(f"Processing file {n + 1} of {total}: {filename}.")
        data = processor(filename)
        # processor returns data splitted by document section.
        processed_data = []
        for record in data:
            section = record["text"]
            results = lt(section)
            for result in results:
                result["file_name"] = record["file_name"]
                result["path"] = record["path"]
                result["headings"] = record["headings"]
                result["page_info"] = record["page_info"]
                result["content_type"] = record["content_type"]
                processed_data.append(result)
        logger.info(f"Ingesting file: {filename}")
        # ingestion(documents=list(data), db_path=db_uri, table_name=db_table_name, schema=DbTableHybrid)
        ingestion(documents=processed_data, db_path=db_uri, table_name=db_table_name)
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
