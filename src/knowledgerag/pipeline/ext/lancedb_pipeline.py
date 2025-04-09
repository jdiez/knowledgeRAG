"""Pipeline:
- https://github.com/alanmeeson/lancedb-haystack/blob/main/examples/pipeline-usage.ipynb
"""
import pyarrow as pa
from lancedb_haystack import LanceDBDocumentStore
from lancedb_haystack import LanceDBEmbeddingRetriever, LanceDBFTSRetriever

# Declare the metadata fields schema, this lets us filter using it.
# See: https://arrow.apache.org/docs/python/api/datatypes.html
metadata_schema = pa.struct([
  ('title', pa.string()),    
  ('publication_date', pa.timestamp('s')),
  ('page_number', pa.int32()),
  ('topics', pa.list_(pa.string()))
])
# We could get it from pydantic definition with LanceModel mother class.

# Create the DocumentStore
document_store = LanceDBDocumentStore(
  database='my_database', 
  table_name="documents", 
  metadata_schema=metadata_schema, 
  embedding_dims=384    # would come from the embedding model.
)

# Create an embedding retriever
embedding_retriever = LanceDBEmbeddingRetriever(document_store)

# Create a Full Text Search retriever
fts_retriever = LanceDBFTSRetriever(document_store)