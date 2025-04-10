""" """

import asyncio
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import more_itertools
from ecumeragtest.database.clients.lancedb_client.lancedb_common import (
    CollectionSettings,
    FtsIndexSettings,
    IndexableElement,
    IndexSettings,
    LanceDbSettings,
    QueryBuilderModel,
    QueryType,
    Result,
    ScalarIndexSettings,
    VectorIndexSettings,
    pyarrow_schema_creator,
)
from ecumeragtest.utils.profilling import async_timeit
from loguru import logger


class AsyncLanceDbDatabase:
    """Lancedb asynchronous client for RAGify.∫"""

    @classmethod
    async def new(cls, database_settings: LanceDbSettings) -> "AsyncLanceDbDatabase":
        match database_settings:
            case LanceDbSettings():
                return await cls(database_settings)
            case _:
                raise ValueError

    def __init__(self, settings: LanceDbSettings) -> None:
        """AI is creating summary for __ainit__

        Args:
            settings (LanceDbSettings): [description]
        """
        self.settings = settings

    async def __aenter__(self) -> "AsyncLanceDbDatabase":
        """AI is creating summary for __aenter__

        Returns:
            [type]: [description]
        """
        self.client = await lancedb.connect_async(self.settings.uri)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """AI is creating summary for __aexit__

        Args:
            exc_type ([type]): [description]
            exc ([type]): [description]
            tb ([type]): [description]
        """
        logger.info("Leaving LanceDb")

    async def _get_config(self) -> dict[str, Any]:
        """AI is creating summary for _get_config

        Returns:
            ComponentConfiguration: [description]
        """
        # await asyncio.sleep(0.1)
        return self.settings.model_dump()

    async def list_indices(
        self,
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """AI is creating summary for list_indices

        Args:
            index_name (str): [description]

        Returns:
            list[dict[str, Any]]: [description]
        """
        table = await self.client.open_table(collection_name)
        indices = await table.list_indices()
        return indices

    async def create_index(self, collection_name: str, index_settings: IndexSettings) -> None:
        """It creates an index of a given collection.

        Args:
            collection_name (str): [description]
            index_settings (VectorIndexSettings): [description]

        Raises:
            ValueError: [description]
        """
        table = await self.client.open_table(collection_name)
        match index_settings:
            case VectorIndexSettings():
                await table.create_vector_index(**index_settings.model_dump())
            case ScalarIndexSettings():
                await table.create_scalar_index(**index_settings.model_dump())
            case FtsIndexSettings():
                await table.create_fts_index(**index_settings.model_dump())
            case _:
                raise ValueError

    async def delete_index(self, collection_name: str, index_name: str) -> None:
        """AI is creating summary for delete_index

        Args:
            collection_name (str): [description]
            index_name (str): [description]
        """
        table = await self.client.open_table(collection_name)
        await table.drop_table(index_name)

    def _gen_data(self, data: list[IndexableElement], batch_size: int | None = None) -> Generator:
        """AI is creating summary for _gen_data

        Args:
            data (list[IndexableElement]): [description]
            batch_size (int, optional): [description]. Defaults to None.

        Raises:
            TypeError: [description]

        Yields:
            Generator: [description]
        """
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

    async def index_elements(self, index_name: str, data: list[IndexableElement]) -> None:
        """ """
        # check table exists
        tbl = await self.client.open_table(index_name)
        await tbl.add(data)

    async def query(self, collection_name: str, query: QueryBuilderModel) -> Result:
        """It performs a database query on a given collection using a set of query parameters.

        Args:
            collection_name (str): Name of the collection.
            query (QueryBuilderModel): Set of parameters to perform the requested query.

        Raises:
            TypeError: [description]

        Returns:
            Result: [description]
        """
        table = await self.client.open_table(collection_name)
        match query.query_type:
            case QueryType.VECTOR | QueryType.AUTO:
                _result = await self.__query_vector(table, query)
            case QueryType.FTS:
                _result = await self.__query_fts(table, query)
            case QueryType.HYBRID:
                _result = await self.__query_hybrid(table, query)
            case _:
                raise TypeError(f"{query.query_type!s}")
        result = Result(
            query=query.model_dump(),
            records=_result.function_result,
            time_elapsed=_result.elapsed_time_str,
            created_at=datetime.now(),
        )
        return result

    @async_timeit
    async def __query_vector(self, table: lancedb.table.AsyncTable, query: QueryBuilderModel) -> list[IndexableElement]:
        """Performs a vector query in the database."""
        return (
            await table.search(query_type=query.query_type, query=query.query)
            .distance_type(query.distance_metric)
            .limit(query.k)
            .select(query.columns)
            .to_list()
        )

    @async_timeit
    async def __query_fts(self, table: lancedb.table.AsyncTable, query: QueryBuilderModel) -> list[IndexableElement]:
        """Performs a full text search."""
        return (
            await table.search(query_type=query.query_type, query=query.query)
            .fts_columns(query.fts_columns)
            .distance_type(query.distance_metric)
            .limit(query.k)
            .select(query.columns)
            .to_list()
        )

    @async_timeit
    async def __query_hybrid(self, table: lancedb.table.AsyncTable, query: QueryBuilderModel) -> list[IndexableElement]:
        """Performs a hybrid query: vector + text."""
        return (
            await table.search(query_type=query.query_type)
            .vector(query.query[0])
            .text(query.query[1])
            # .rerank(normalize=query.rerank.normalize, reranker=query.rerank.reranker)
            .limit(query.k)
            .select(query.columns)
            .to_list()
        )

    async def retrieve_records(self) -> Any:
        """ """
        raise NotImplementedError

    async def sample(self) -> Any:
        """ """
        raise NotImplementedError

    async def count(self, collection_name: str) -> int:
        """AI is creating summary for count

        Returns:
            [type]: [description]
        """
        table = await self.client.open_table(collection_name)
        row_count = await table.count_rows()
        return row_count

    async def health(self) -> Any:
        """ """
        raise NotImplementedError

    # collection related
    async def create_collection(self, collection_settings: CollectionSettings) -> None:
        """AI is creating summary for create_collection.

        Args:
            collection_settings (CollectionSettings): [description]

        Returns:
            [type]: [description]
        """
        collection = await self.client.create_table(**collection_settings.model_dump(by_alias=True))
        return collection

    async def list_collections(
        self,
    ) -> list[str]:
        """AI is creating summary for list_collections

        Returns:
            list[str]: [description]
        """
        return await self.client.table_names()

    async def delete_collection(self, collection_name: str) -> None:
        """AI is creating summary for drop_collection

        Args:
            collection_name (str): [description]
        """
        await self.client.drop_table(collection_name)

    async def delete_all_collections(
        self,
    ) -> None:
        """Equivalent to delete the database."""
        await self.client.drop_all_tables()


if __name__ == "__main__":

    async def doctest_example():
        data_path = Path(__file__).parent.parent.parent.parent.absolute() / Path("data")
        lancedb_file = data_path / Path("lance_test_async.db")

        data = [{"vector": [1.1, 1.2], "lat": 45.5, "long": -122.7}, {"vector": [0.2, 1.8], "lat": 40.1, "long": -74.1}]
        SCHEMA = pyarrow_schema_creator(data)
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
        )
        async with AsyncLanceDbDatabase(dbs) as aldb:
            await aldb.create_collection(cs)
            print(await aldb.list_collections())

    asyncio.run(doctest_example())
