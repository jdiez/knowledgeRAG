import random
from collections.abc import Generator, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import more_itertools
from loguru import logger

from knowledgerag.database.lancedb_lib.lancedb_common import (
    Collection,
    CollectionSettings,
    Database,
    DistanceMetric,
    FtsIndexSettings,
    IndexableElement,
    IndexSettings,
    LanceDbSettings,
    QueryBuilderModel,
    QueryType,
    Record,
    RecordBatchSettings,
    Result,
    ScalarIndexSettings,
    VectorIndexSettings,
    pyarrow_schema_creator,
)
from knowledgerag.utils.profilling import timeit


class LancedbDatabase(Database):
    """Lancedb custom client."""

    @classmethod
    def new(cls, database_settings: LanceDbSettings) -> "LancedbDatabase":
        """Returns a new instance of LanceDbDatabase.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        match database_settings:
            case LanceDbSettings():
                return cls(database_settings)
            case _:
                raise ValueError

    def __init__(self, settings: LanceDbSettings) -> None:
        """Initializes LanceDbDatabase instance.

        Args:
            uri (str): [description]
        """
        self.settings = settings
        self.uri = self.settings.uri

    def __enter__(self) -> "LancedbDatabase":
        """AI is creating summary for __enter__

        Returns:
            [type]: [description]
        """
        self.client = self.__create_client()
        logger.info("Entering LanceDB.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """AI is creating summary for __exit__

        Args:
            exc_type ([type]): [description]
            exc ([type]): [description]
            tb ([type]): [description]
        """
        logger.info("Leaving LanceDB.")

    def __create_client(self) -> Any:
        """AI is creating summary for __create_client

        Returns:
            lancedb.Client: [description]
        """
        client = lancedb.connect(**{k: v for k, v in self.settings.model_dump().items() if v is not None})
        return client

    def _get_config(self) -> dict[str, Any]:
        """AI is creating summary for _get_config

        Returns:
            dict[str, Any]: [description]
        """
        return self.settings.model_dump()

    def create_index(self, collection_name: str, index_settings: IndexSettings) -> None:
        """It creates an index of a given collection.

        Args:
            collection_name (str): [description]
            index_settings (VectorIndexSettings): [description]

        Raises:
            ValueError: [description]
        """
        table = self.client.open_table(collection_name)
        match index_settings:
            case VectorIndexSettings():
                table.create_vector_index(**index_settings.model_dump())
            case ScalarIndexSettings():
                table.create_scalar_index(**index_settings.model_dump())
            case FtsIndexSettings():
                table.create_fts_index(**index_settings.model_dump())
            case _:
                raise ValueError

    def list_indices(self, collection_name: str) -> Iterable | None:
        """List all indices names of a collection.

        Args:
            page_token (str, optional): [description]. Defaults to None.
            limit (int, optional): [description]. Defaults to None.

        Returns:
            Iterable[str]: [description]
        """
        table = self.client.open_table(collection_name)
        return table.list_indices()

    def delete_index(self, collection_name: str, index_name: str) -> None:
        """Delete a collection index by name.

        Args:
            collection_name (str): [description]
            index_name (str): [description]
        """
        table = self.client.open_table(collection_name)
        table.drop_index(name=index_name)

    def query(self, collection_name: str, query: QueryBuilderModel) -> Result:
        """It performs a database query on a given collection using a set of query parameters.

        Args:
            collection_name (str): Name of the collection.
            query (QueryBuilderModel): Set of parameters to perform the requested query.

        Raises:
            TypeError: [description]

        Returns:
            Result: [description]
        """
        table = self.client.open_table(collection_name)
        match query.query_type:
            case QueryType.VECTOR | QueryType.AUTO:
                _result = self.__query_vector(table, query)
            case QueryType.FTS:
                _result = self.__query_fts(table, query)
            case QueryType.HYBRID:
                _result = self.__query_hybrid(table, query)
            case _:
                raise TypeError(f"{query.query_type!s}")
        result = Result(
            query=query.model_dump(),  # Follow Query object structure.
            records=_result.function_result,
            time_elapsed=_result.elapsed_time_str,
            created_at=datetime.now(),
        )
        return result

    @timeit
    def __query_vector(self, table: lancedb.table.Table, query: QueryBuilderModel) -> list[IndexableElement]:
        """Performs a vector query in the database."""
        return (
            table.search(query_type=query.query_type, query=query.query)
            .distance_type(query.distance_metric)
            .limit(query.k)
            .select(query.columns)
            .to_list()
        )

    @timeit
    def __query_fts(self, table: lancedb.table.Table, query: QueryBuilderModel) -> list[IndexableElement]:
        """Performs a full text search."""
        return (
            table.search(query_type=query.query_type, query=query.query)
            .fts_columns(query.fts_columns)
            .distance_type(query.distance_metric)
            .limit(query.k)
            .select(query.columns)
            .to_list()
        )

    @timeit
    def __query_hybrid(self, table: lancedb.table.Table, query: QueryBuilderModel) -> list[IndexableElement]:
        """Performs a hybrid query: vector + text."""
        return (
            table.search(query_type=query.query_type)
            .vector(query.query[0])
            .text(query.query[1])
            # .rerank(normalize=query.rerank.normalize, reranker=query.rerank.reranker)
            .limit(query.k)
            .select(query.columns)
            .to_list()
        )

    def retrieve_records(self, collection_name: str) -> Iterable:
        """ """
        table = self.get_collection(collection_name)
        for record in table.query().to_list():
            yield Record(record)

    def sample(self, collection_name: str, size: int = 10) -> list[Record]:
        """Returns a random sample for the collection.

        Args:
            collection_name (str): [description]
            size (int, optional): [description]. Defaults to 10.

        Returns:
            list[Record]: [description]
        """
        n_records = self.count(collection_name=collection_name)
        index = random.sample(range(0, n_records), size)
        uthr = max(index)
        lthr = min(index)
        records = []
        for n, rec in self.retrieve_records(collection_name=collection_name):
            if n > lthr:
                if n in index:
                    records.append(rec)
                elif n > uthr:
                    break
        return records

    def count(self, collection_name: str) -> int:
        """Counts the number of records of a given collection.

        Returns:
            [type]: [description]
        """
        table = self.client.open_table(collection_name)
        return table.count_rows()

    def health(self) -> bool:
        """Database / connection health.
        Needs to be true for the successful creation of the instance.

        Returns:
            bool: [description]
        """
        return NotImplementedError

    def create_collection(self, collection_settings: CollectionSettings) -> lancedb.table.Table:
        """It creates a collection.

        Args:
            collection_name (str): [description]
            collection_settings (CollectionSettings, optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            lancedb.table.Table: [description]
        """
        match collection_settings:
            case CollectionSettings():
                table = self.client.create_table(**collection_settings.model_dump(by_alias=True))
            case _:
                raise ValueError()
        return table

    def list_collections(self) -> list[str]:
        """List all collection names in the database.

        Returns:
            list[str]: [description]
        """
        return self.client.table_names()

    def get_collection(self, collection_name: str) -> Collection | None:
        """It returns the database collection object.

        Returns:
            [type]: [description]
        """
        return self.client.open_table(collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """It deletes a database collection.

        Args:
            collection_name (str): [description]

        Raises:
            ValueError: [description]
            TypeError: [description]
            TypeError: [description]

        Returns:
            [type]: [description]

        Yields:
            [type]: [description]
        """
        self.client.drop_table(collection_name)

    def delete_all_collections(
        self,
    ) -> None:
        """Equivalent to delete the database."""
        self.client.drop_all_tables()  # same as: self.client.drop_database()

    def _gen_data(self, data: list[IndexableElement], batch_size: int | None = None) -> Generator:
        """AI is creating summary for _gen_data

        Args:
            data (list[IndexableElement]): [description]

        Yields:
            Generator: [description]
        """
        # table.query().
        match batch_size:
            case int():
                for element in more_itertools.chunked(data, n=batch_size):
                    yield element
            case None:
                for element in data:
                    yield [
                        element,
                    ]
            case _:
                raise TypeError()

    def insert_records(
        self,
        collection_name: str,
        record_batch: RecordBatchSettings,
    ) -> None:
        """AI is creating summary for insert_records

        Args:
            collection_name (str): [description]
            record_batch (RecordBatchSettings): [description]
        """
        table = self.client.open_table(collection_name)
        table.add(**record_batch.model_dump())

    def delete_records(self, collection_name: str, query: str | list[int]) -> None:
        """AI is creating summary for delete_records

        Args:
            collection_name (str): [description]
            query (str): [description]
        """
        table = self.client.open_table(collection_name)
        table.delete(query)


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent.parent.absolute() / Path("data")
    lancedb_file = data_path / Path("lance_test.db")

    data = [{"vector": [1.1, 1.2], "lat": 45.5, "long": -122.7}, {"vector": [0.2, 1.8], "lat": 40.1, "long": -74.1}]
    data2 = [
        {"vector": [3.1, 4.1], "text": "Frodo was a happy puppy"},
        {"vector": [5.9, 26.5], "text": "There are several kittens playing"},
    ]
    SCHEMA = pyarrow_schema_creator(data)
    SCHEMA2 = pyarrow_schema_creator(data2)
    dbs = LanceDbSettings(uri=lancedb_file)
    cs = CollectionSettings(
        name="second_table",
        data=data,
        schema=SCHEMA,
        # for these we can asume default values.
        mode="overwrite",
        exists=True,
        on_bad_vectors="error",
        fill_value=0.0,
        storage_options={},
        enable_v2_manifest_paths=None,
    )
    cs2 = CollectionSettings(
        name="third_table",
        data=data2,
        schema=SCHEMA2,
        mode="overwrite",
        exists=True,
        on_bad_vectors="error",
        fill_value=0.0,
        storage_options={},
        enable_v2_manifest_paths=None,
    )

    with LancedbDatabase(dbs) as ldb:
        print("Database config:")
        print(ldb._get_config())
        print(cs)
        res = ldb.create_collection(cs)
        res2 = ldb.create_collection(cs2)
        print(res)
        print(res2)
        print(ldb.count(collection_name="second_table"))
        print(ldb.count(collection_name="third_table"))
        ldb.create_index("third_table", FtsIndexSettings(field_names="text", use_tantivy=False))
        ldb.list_indices(collection_name="third_table")
        table = ldb.get_collection("third_table")
        result = table.search("puppy").limit(10).select(["text"]).to_list()
        print(result)
        vector_query = [5.1, 25.0]
        text_query = "cat puppies"
        query = QueryBuilderModel(
            table="third_table",
            query_type=QueryType.HYBRID,
            vector_column_name="vector",
            query=(vector_query, text_query),
            columns=["vector", "text"],
            k=5,
        )
        result = ldb.query(collection_name="third_table", query=query)
        print(result)
        ldb.insert_records(collection_name="second_table", record_batch=RecordBatchSettings(data=data))
        print(ldb.count(collection_name="second_table"))
        print(ldb.count(collection_name="third_table"))
        print(ldb.list_collections())
        query = QueryBuilderModel(
            table="second_table",
            query_type=QueryType.VECTOR,
            query=[0.3, 1.8],
            vector_column_name="vector",
            distance_metric=DistanceMetric.L2,
            k=3,
            columns=["vector", "lat", "long"],
        )
        print(query)
        print(ldb.query(collection_name="second_table", query=query))
        print(ldb.list_indices(collection_name="second_table"))
