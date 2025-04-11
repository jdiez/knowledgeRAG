import json
import logging
from pathlib import Path
from typing import Any, Callable

import lancedb
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from loguru import logger
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

embedder = get_registry().get("sentence-transformers").create(name="all-mpnet-base-v2", device="cpu")
# check instructor embeddings: https://lancedb.github.io/lancedb/embeddings/available_embedding_models/text_embedding_functions/instructor_embedding/


class DbTableHybrid(LanceModel):
    text: str = embedder.SourceField()
    vector: Vector(embedder.ndims()) = embedder.VectorField()
    headings: str
    page: str
    content_type: str


class ChunkMetadata(BaseModel):
    text: str | None = ""
    headings: list[str] | None = []
    page_info: int | None = None
    content_type: str | None = None


class DocumentProcessor:
    def __init__(
        self,
        tokenizer: str = "jinaai/jina-embeddings-v3",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",  # BAAI/bge-m3
        db_uri: str = "/home/jdiez/Downloads/scratch/docling.db",
        db_table_name: str = "document_chunks",
    ) -> None:
        """Initialize document processor with necessary components"""
        # self.api_key = os.getenv("WATSONX_API_KEY", None)
        # self.project_id = os.getenv("WATSONX_PROJECT_ID", None)
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.db_uri = db_uri
        self.db_table_name = db_table_name
        self.db = lancedb.connect(self.db_uri)
        self.setup_document_converter()
        self.setup_ml_components()

    def setup_document_converter(self):
        """Configure document converter with advanced processing capabilities"""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["en"]
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.MPS)

        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                # InputFormat.IMAGE,    # could transform image to text, text embedding and include in the db.
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.CSV,
                InputFormat.MD,
            ],  # whitelist formats, non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline  # , backend=MsWordDocumentBackend
                ),
            },
        )

    def setup_ml_components(self):
        """Initialize embedding model and LLM"""
        self.embed_model = SentenceTransformer(self.embedding_model)

    def extract_chunk_metadata(self, chunk, metadata: BaseModel = ChunkMetadata) -> dict[str, Any]:
        """Extract essential metadata from a chunk"""
        metadata = dict(metadata())
        if "text" in metadata:
            metadata["text"] = chunk.text

        if hasattr(chunk, "meta"):
            # Extract headings
            if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                metadata["headings"] = chunk.meta.headings

            # Extract page information and content type
            if hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if hasattr(item, "label"):
                        metadata["content_type"] = str(item.label)

                    if hasattr(item, "prov") and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, "page_no"):
                                metadata["page_info"] = prov.page_no

        return metadata

    def process_document(self, pdf_path: str) -> Any:
        """Process document and create searchable index with metadata"""
        logger.info(f"Processing document: {pdf_path}")

        # Convert document
        result = self.converter.convert(pdf_path)
        doc = result.document

        # Create chunks using hybrid chunker
        chunker = HybridChunker(tokenizer=self.tokenizer)
        chunks = list(chunker.chunk(doc))
        for _, chunk in enumerate(chunks):
            metadata = self.extract_chunk_metadata(chunk)
            # embeddings = self.embed_model.encode(metadata["text"])
            data_item = {
                # "vector": embeddings,
                "text": metadata["text"],
                "headings": json.dumps(metadata["headings"]),
                "page": metadata["page_info"],
                "content_type": metadata["content_type"],
            }
            yield data_item

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a structured context for the LLM"""
        context_parts = []
        for chunk in chunks:
            # Include section information if available
            try:
                headings = json.loads(chunk["headings"])
                if headings:
                    context_parts.append(f"Section: {' > '.join(headings)}")
            except Exception as e:
                logger.error(e)

            # Add page reference
            if chunk["page"]:
                context_parts.append(f"Page {chunk['page']}:")

            # Add the content
            context_parts.append(chunk["text"])
            context_parts.append("-" * 40)

        return "\n".join(context_parts)

    def query(self, question: str, k: int = 5) -> str:
        """Query the document using semantic search and generate an answer"""
        # Create query embedding and search
        query_embedding = self.embed_model.encode(question)
        results = self.index.search(query_embedding).limit(k)
        chunks = results.to_pandas()

        # Display retrieved chunks with their context
        logger.info(f"\nRelevant chunks for query: '{question}'")
        logger.info("=" * 80)

        # Format chunks for display and LLM
        context = self.format_context(chunks.to_dict("records"))
        logger.info(context)

        # Generate answer using structured context
        prompt = f"""Based on the following excerpts from a document:

                    {context}

                    Please answer this question: {question}

                    Make use of the section information and page numbers in your answer when relevant.
                    """

        return self.llm(prompt)


def basic_lancedb_ingestion_pipeline(
    documents: list[Path | str] | Path | str,
    processor: Callable,
    db_path: Path | str,
    table_name: str,
    schema: LanceModel,
) -> None:
    """_summary_

    Args:
        documents (list[Path]): _description_
        processor (Callable): _description_
        db_path (Path): _description_
        table_name (str): _description_
        schema (LanceModel): _description_
    """
    data = list(processor.process_document(documents))
    conn = lancedb.connect(db_path)
    conn.create_table(table_name, data=data, schema=schema)


def main():
    logging.basicConfig(level=logging.INFO)

    processor = DocumentProcessor()
    pdf_path = "/home/jdiez/Downloads/jamaneurology_johnson_2021_oi_210047_1633018740.39649.pdf"
    db = "/home/jdiez/Downloads/scratch/docling_bis.db"
    table = "example_pdf"
    schema = DbTableHybrid
    basic_lancedb_ingestion_pipeline(
        documents=pdf_path, processor=processor, db_path=db, table_name=table, schema=schema
    )

    # Example query
    # question = "What are the main features of InternLM-XComposer-2.5?"
    # answer = processor.query(question)
    # logger.info("\nAnswer:")
    # logger.info("=" * 80)
    # logger.info(answer)


if __name__ == "__main__":
    main()
