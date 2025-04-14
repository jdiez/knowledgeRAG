import enum
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, NewType

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from knowledgerag.utils.wrangling import data_cleaner

ComponentConfiguration = NewType("ComponentConfiguration", dict[str, Any])
IndexableElement = NewType("IndexableElement", dict[str, Any])
Reranker = NewType("Reranker", dict[str, Any])

QueryDataType = list[list[float]] | list[float] | str | tuple | None
ScalarIndexType = Literal["BTREE", "BITMAP", "LABEL_LIST"]


class Collection:
    pass


class DistanceMetric(str, enum.Enum):
    """ """

    L2 = "L2"
    COSINE = "cosine"
    DOT = "dot"
    HAMMING = "hamming"


class QueryType(str, enum.Enum):
    """ """

    VECTOR = "vector"
    FTS = "fts"
    HYBRID = "hybrid"
    AUTO = "auto"


class IndexType(str, enum.Enum):
    """ """

    SCALAR = "scalar"
    VECTOR = "vector"
    FTS = "fts"


class QueryBuilderModel(BaseModel):
    """A query builder that performs hybrid vector and full text search"""

    # table: Annotated[str, Field(description="The table to query.")]
    query_type: Annotated[QueryType, Field(default=QueryType.AUTO, description="")]
    query: Annotated[QueryDataType, Field(description="The query to use.")]
    vector_column_name: Annotated[str, Field(description="")]
    ordering_field_name: Annotated[str | None, Field(default=None, description="")]
    fts_columns: Annotated[str | list[str] | None, Field(default=None, description="")]
    fast_search: Annotated[bool | None, Field(default=False, description="")]
    where: Annotated[str | None, Field(default=None, description="")]
    k: Annotated[int | None, Field(default=3, description="Top k results to return.")]
    distance_metric: Annotated[DistanceMetric | None, Field(default=DistanceMetric.L2, description="")]
    columns: Annotated[list[str] | None, Field(default=None, description="Columns to return in the result.")]


class RerankModel(BaseModel):
    """AI is creating summary for RerankModel"""

    normalize: Annotated[Literal["rank", "score"] | None, Field(default=None, description="")]
    reranker: Annotated[Reranker | None, Field(default=None, description="")]


class Result(BaseModel):
    """ """

    query: dict[str, Any]
    records: list[IndexableElement]
    time_elapsed: str
    created_at: datetime


class CollectionSettings(BaseModel):
    """Setting for Collection creation.

    Args:
        BaseModel ([type]): [description]
    """

    model_config = ConfigDict(alias_generator=data_cleaner)

    name: str
    schema: Annotated[Any, Field(alias="schema", default=None)]
    data: list[IndexableElement] | None
    mode: Literal["create", "overwrite"] | None = None
    exist_ok: bool | None = False
    on_bad_vectors: str | None = "error"
    fill_value: float | None = 0.0
    storage_options: dict[str, str] | None = {}
    # enable_v2_manifest_paths: bool | None = None


# Indices settings for vector, scalar, and full-text search.
class VectorIndexSettings(BaseModel):
    """Settings for Vector Index Creation.

    Args:
        BaseModel ([type]): [description]
    """

    metric: Annotated[
        DistanceMetric | None, Field(default=DistanceMetric.L2)
    ]  # The distance distancemetric to use when creating the index. Valid values are "L2", "cosine", "dot", or "hamming". L2 is euclidean distance. Hamming is available only for binary vectors.
    num_partitions: Annotated[
        int | None, Field(default=256)
    ]  # The number of IVF partitions to use when creating the index. Default is 256.
    num_sub_vectors: Annotated[
        int | None,
        Field(default=96, description="The number of PQ sub-vectors to use when creating the index. Default is 96."),
    ]
    vector_column_name: Annotated[
        str | None,
        Field(
            default="VECTOR_COLUMN_NAME",
            description="The name of the column containing the vectors. Default is VECTOR_COLUMN_NAME.",
        ),
    ]
    replace: Annotated[
        bool | None,
        Field(
            default=True,
            description="If True, replace the existing index with the same name. If False, raise an error if an index with the same name already exists. Default is True.",
        ),
    ]
    accelerator: Annotated[
        str | None,
        Field(
            default=None,
            description="f set, use the given accelerator to create the index. Only support 'cuda' for now.",
        ),
    ]
    index_cache_size: Annotated[
        int | None,
        Field(default=None, description="The size of the index cache in number of entries. Default value is 256."),
    ]
    num_bits: Annotated[
        int | None,
        Field(
            default=8,
            description="The number of bits to encode sub-vectors. Only used with the IVF_PQ index. Only 4 and 8 are supported.",
        ),
    ]


class ScalarIndexSettings(BaseModel):
    """Settings for Scalar Index Creation."""

    column: Annotated[str, Field()]  # The column to be indexed. Must be a boolean, integer, float, or string column.
    replace: Annotated[bool | None, Field(default=True)]  # Replace the existing index if it exists.
    index_type: Annotated[ScalarIndexType | None, Field(default="BTREE")]  # The type of index to create.


class FtsIndexSettings(BaseModel):
    """Settings for Fts Index Creation."""

    field_names: Annotated[
        str | list[str],
        Field(description="The name(s) of the field to index. can be only str if use_tantivy=True for now."),
    ]
    replace: Annotated[
        bool | None,
        Field(
            default=False,
            description="If True, replace the existing index if it exists. Note that this is not yet an atomic operation; the index will be temporarily unavailable while the new index is being created.",
        ),
    ]
    writer_heap_size: Annotated[
        int | None,
        Field(
            default=None,
            descruotion="Only available with use_tantivy=True: 1024 * 1024 * 1024. # Only available with use_tantivy=True",
        ),
    ]
    ordering_field_names: Annotated[
        str | list[str] | None,
        Field(
            default=None,
            description="A list of unsigned type fields to index to optionally order results on at search time. only available with use_tantivy=True",
        ),
    ]
    tokenizer_name: Annotated[
        str | None,
        Field(
            default=None,
            description='The tokenizer to use for the index. Can be "raw", "default" or the 2 letter language code followed by "_stem". So for english it would be "en_stem". For available languages see: https://docs.rs/tantivy/latest/tantivy/tokenizer/enum.Language.html',
        ),
    ]
    use_tantivy: Annotated[
        bool | None,
        Field(
            default=True,
            description="If True, use the legacy full-text search implementation based on tantivy. If False, use the new full-text search implementation based on lance-index.",
        ),
    ]
    with_position: Annotated[
        bool | None, Field(default=True)
    ]  # Only available with use_tantivy=False If False, do not store the positions of the terms in the text. This can reduce the size of the index and improve indexing speed. But it will raise an exception for phrase queries.
    base_tokenizer: Annotated[
        Literal["simple", "whitspace", "raw"],
        Field(default="simple", description="The base tokenizer to use for tokenization. Options are: "),
    ]
    language: Annotated[str, Field(default="English", description="The language to use for tokenization.")]
    max_token_length: Annotated[
        int | None,
        Field(
            default=40, description="The maximum token length to index. Tokens longer than this length will be ignored."
        ),
    ]
    lower_case: Annotated[
        bool | None,
        Field(
            default=True,
            description="# Whether to convert the token to lower case. This makes queries case-insensitive.",
        ),
    ]
    stem: Annotated[
        bool | None,
        Field(
            default=False,
            description='Whether to stem the token. Stemming reduces words to their root form. For example, in English "running" and "runs" would both be reduced to "run".',
        ),
    ]
    remove_stop_words: Annotated[
        bool | None,
        Field(
            default=False,
            description='Whether to remove stop words. Stop words are common words that are often removed from text before indexing. For example, in English "the" and "and".',
        ),
    ]
    ascii_folding: Annotated[
        bool | None,
        Field(
            default=False,
            description='Whether to fold ASCII characters. This converts accented characters to their ASCII equivalent. For example, "café" would be converted to "cafe".',
        ),
    ]


IndexSettings = VectorIndexSettings | ScalarIndexSettings | FtsIndexSettings


class LanceDbSettings(BaseModel):
    """Settings for database instance initialization.

    Args:
        BaseModel ([type]): [description]
    """

    uri: Annotated[str | Path, Field()]  # The uri of the database.
    api_key: Annotated[
        str | None,
        Field(
            default=None,
            description="If presented, connect to LanceDB cloud. "
            "Otherwise, connect to a database on file system or cloud storage. "
            "Can be set via environment variable LANCEDB_API_KEY.",
        ),
    ]
    region: Annotated[str | None, Field(default=None, description="The region to use for LanceDB Cloud.")]
    host_override: Annotated[str | None, Field(default=None, description="The override url for LanceDB Cloud.")]
    read_consistency_interval: Annotated[
        timedelta | None,
        Field(
            default=None,
            description="None  # (For LanceDB OSS only) The interval at which to check for updates to the table from other processes."
            "If None, then consistency is not checked. For performance reasons, this is the default."
            "For strong consistency, set this to zero seconds. Then every read will check for updates from other processes."
            "As a compromise, you can set this to a non-zero timedelta for eventual consistency."
            "If more than that interval has passed since the last check, then the table will be checked for updates."
            "Note: this consistency only applies to read operations. Write operations are always consistent.",
        ),
    ]
    client_config: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Configuration options for the LanceDB Cloud HTTP client. If a dict, then the keys are the attributes of the ClientConfig class. If None, then the default configuration is used.",
        ),
    ]
    storage_options: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description="Additional options for the storage backend. See available options at https://lancedb.github.io/lancedb/guides/storage/.",
        ),
    ]


class RecordBatchSettings(BaseModel):
    """Sett"""

    data: Annotated[list[IndexableElement], Field(description="Data to be inserted")]
    mode: Annotated[Literal["append", "overwrite"] | None, Field(default="append", description="")]
    on_bad_vectors: Annotated[Literal["error", "drop", "fill"] | None, Field(default="error", description="")]
    fill_value: Annotated[float | None, Field(default=None, description="")]


def pyarrow_schema_creator(data: list[dict[str, Any]] | dict[str, Any], metadata: dict[str, str]) -> pa.schema:
    """It creates a new schema from a dict or list of dicts, records.

    Needs to deal and be able to save metadata to file (knowledge base).

    Args:
        data (list[dict[str, Any]]): [description]
        metadata (dict): [description]

    Returns:
        pa.schema: [description]
    """
    model = data[0] if isinstance(data, list) else data
    fields = []
    for k, v in model.items():
        match v:
            case [[float, *_], *_]:
                # match a list of list containing float elements: Matrix
                # fields.append(pa.field(k, pa.list_(pa.float32(), len(v))))
                fields.append(pa.field(k, pa.FloatArray(v)))
            case [float, *_]:
                # match a list containing float elements: Vector
                fields.append(pa.field(k, pa.list_(pa.float32(), len(v))))
            case float():
                fields.append(pa.field(k, pa.float32()))
            case int():
                fields.append(pa.field(k, pa.int16()))
            case str():
                fields.append(pa.field(k, pa.string()))
            case dict():
                # only dict[str, str] or casting the value.
                fields.append(pa.field(k, pa.dictionary(pa.utf8(), pa.utf8())))
            case _:
                raise TypeError(f"Type:{v!s}.")
    table = pa.schema(fields)
    table.schema.metadata = {**table.schema.metadata, **metadata}
    return table
