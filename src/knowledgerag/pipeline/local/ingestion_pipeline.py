import json
import logging
from collections.abc import Generator
from functools import partial
from pathlib import Path
from typing import Any, Callable

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

from knowledgerag import read_configuration
from knowledgerag.database.lancedb_lib.lancedb_client import LancedbDatabase
from knowledgerag.database.lancedb_lib.lancedb_common import CollectionSettings, LanceDbSettings


class ChunkMetadata(BaseModel):
    path: str | None = ""
    file_name: str | None = ""
    text: str | None = ""
    headings: str = ""
    page_info: int | None = None
    content_type: str | None = None


class DocumentProcessor:
    def __init__(
        self,
        tokenizer: str,
        chunker: Callable | None = HybridChunker,
        embedding_model: str | None = None,
        device: str | None = "cpu",
        # db_uri: str,
        # db_table_name: str,
    ) -> None:
        """Initialize document processor with necessary components"""
        self.tokenizer = tokenizer
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.device = device
        # self.db_uri = db_uri
        # self.db_table_name = db_table_name
        self.setup_document_converter()

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
            ],  # whitelist formats, from non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline  # , backend=MsWordDocumentBackend
                ),
            },
        )

    def setup_ml_components(self):
        """Initialize embedding model and LLM"""
        if self.embed_model:
            if isinstance(self.embedding_model, str):
                provider, model = self.embedding_model.split("/")
                self.embed_model = get_registry().get(provider).create(name=model, device=self.device)
                # self.embed_model = SentenceTransformer()
            elif isinstance(self.embedding_model, Callable):
                self.embed_model = self.embedding_model
            else:
                raise TypeError()

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
        """Process document and creatfrom e searchable index with metadata"""
        logger.info(f"Processing document: {pdf_path}")
        # Convert document
        result = self.converter.convert(pdf_path)
        # extract further file metadata?
        doc = result.document
        # Create chunks using hybrid chunker
        chunker = self.chunker(tokenizer=self.tokenizer)
        # chunker = self.chunker(tokenizer=self.tokenizer)
        chunks = list(chunker.chunk(doc))
        for _, chunk in enumerate(chunks):
            metadata = self.extract_chunk_metadata(chunk)
            # embeddings = self.embed_model.encode(metadata["text"])
            data_item = ChunkMetadata(
                # "vector": self.embedding_model.encode(metadata['text']),
                file_name=Path(pdf_path).name,
                path=pdf_path,
                text=metadata["text"],
                headings=json.dumps(metadata["headings"]),
                page=metadata["page_info"],
                content_type=metadata["content_type"],
            )
            # validate with pydantic object.
            yield data_item.model_dump()

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
            # further split the chunk?

        return "\n".join(context_parts)

    def __call__(self, file_name: str | Path) -> Generator:
        """It returns a generator list[str, Any] from parsed file."""
        return self.process_document(file_name)


def basic_lancedb_ingestion_pipeline(
    documents: list[Path | str] | Path | str,
    processor: Callable,
    db_path: Path | str,
    table_name: str,
    schema: Any,
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
    collection_settings = CollectionSettings(name=table_name, schema=schema, data=data, mode="create")
    with LancedbDatabase(LanceDbSettings(uri=db_path)) as client:
        if table_name not in client.list_collections():
            client.create_collection(collection_settings)
        else:
            client.add_records(collection_name=table_name, data=data)


def set_up(pipe: dict[str, Any]):
    logging.basicConfig(level=logging.INFO)
    embedding_model = pipe["processing"]["embedding"]
    device = pipe["processing"]["device"]
    tokenizer = pipe["processing"]["tokenizer"]
    db_uri = pipe["storage"]["parameters"]["uri"]
    db_table_name = pipe["storage"]["parameters"]["collection"]

    provider, model = embedding_model.split("/")
    embedder = get_registry().get(provider).create(name=model, device=device)

    class DbTableHybrid(LanceModel):
        file_name: str
        text: str = embedder.SourceField()
        vector: Vector(embedder.ndims()) = embedder.VectorField()
        headings: str
        page: str
        content_type: str

    processor = DocumentProcessor(
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        device=device,
        db_uri=db_uri,
        db_table_name=db_table_name,
    )

    pf = partial(
        basic_lancedb_ingestion_pipeline,
        processor=processor,
        db_path=db_uri,
        table_name=db_table_name,
        schema=DbTableHybrid,
    )

    return pf


if __name__ == "__main__":
    configuration = read_configuration()
    pipe = configuration["pipeline"]["default"]
    pf = set_up(pipe=pipe)
    # from knowledgerag.io.reader.reader import main_file_reader
    # from rich import print_json
    # filetypes = pipe['input']['extensions']
    # print(filetypes)
    # root_path = Path("/home/jdiez/Downloads/test").resolve()
    # for i in list(main_file_reader(root_path, allowed_file_types=filetypes)):
    #     print_json(i.model_dump_json())
    for pdf_path in list(Path("/home/jdiez/Downloads/docs/").glob("*.pdf")):
        print(pf(documents=pdf_path))
    # generate index after ingestion.
    # with LancedbDatabase(LanceDbSettings(uri=db_path)) as client:
    #     table = client.client.open_table(pipe['storage']['parameters']['collection'])
    #     table.create_index('cosine')
    #     table.create_fts_index("text", use_tantivy=False)
