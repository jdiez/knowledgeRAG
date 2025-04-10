
from docling_haystack.converter import ExportType
from docling_haystack.converter import DoclingConverter
from haystack import Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
)
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from milvus_haystack import MilvusDocumentStore

from docling.chunking import HybridChunker

from lancedb_haystack import LanceDBDocumentStore
from lancedb_haystack import LanceDBEmbeddingRetriever, LanceDBFTSRetriever

HF_TOKEN = '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'    # read
PATHS = ["https://arxiv.org/pdf/2408.09869"]  # Docling Technical Report
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EXPORT_TYPE = ExportType.DOC_CHUNKS
MILVUS_URI = "/home/jdiez/Desktop/docling.db"
LANCE_URI = "/home/jdiez/Desktop/docling_lcdb.db"


document_store = MilvusDocumentStore(
    connection_args={"uri": MILVUS_URI},
    drop_old=True,
    text_field="txt",  # set for preventing conflict with same-name metadata field
)
document_store_ldb = LanceDBDocumentStore(
  database=LANCE_URI,
  table_name="documents",
)
idx_pipe = Pipeline()
idx_pipe.add_component(
    "converter",
    DoclingConverter(
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    ),
)
idx_pipe.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(model=EMBED_MODEL_ID),
)
idx_pipe.add_component("writer", DocumentWriter(document_store=document_store))
if EXPORT_TYPE == ExportType.DOC_CHUNKS:
    idx_pipe.connect("converter", "embedder")
elif EXPORT_TYPE == ExportType.MARKDOWN:
    idx_pipe.add_component(
        "splitter",
        DocumentSplitter(split_by="sentence", split_length=1),
    )
    idx_pipe.connect("converter.documents", "splitter.documents")
    idx_pipe.connect("splitter.documents", "embedder.documents")
else:
    raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")
idx_pipe.connect("embedder", "writer")
idx_pipe.run({"converter": {"paths": PATHS}})

# Switch to Lancedb