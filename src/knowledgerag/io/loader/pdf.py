"""Module for handling PDF metadata extraction and management."""

import json
from datetime import datetime
from pathlib import Path

from pypdf.errors import EmptyFileError
from rich import print_json

from knowledgerag.models.pdf import PdfMetadata


class FileDoesNotExistError(ValueError):
    """Custom exception for non-existent files."""

    def __init__(self, file_path: Path):
        super().__init__(f"File {file_path} does not exist")


class NotALocalDirectoryError(ValueError):
    """Custom exception for paths that are not directories."""

    def __init__(self, path: Path):
        super().__init__(f"Path {path} is not a directory")


class DirectoryNotFoundError(ValueError):
    """Custom exception for non-existent directories."""

    def __init__(self, directory: Path):
        super().__init__(f"Directory {directory} does not exist")


class NotAPdfFileError(ValueError):
    """Custom exception for files that are not PDFs."""

    def __init__(self, file_path: Path):
        """Initialize the exception with a file path."""
        super().__init__(f"File {file_path} is not a PDF")


def scan_directory(directory: Path, recursive: bool = True) -> list[PdfMetadata]:
    """
    Scan a directory for PDF files and extract their metadata.

    Args:
        directory: Path to the directory to scan
        recursive: Whether to scan subdirectories recursively

    Returns:
        list of PdfMetadata objects for each PDF found
        raise DirectoryNotFoundError(directory)
    Raises:
        ValueError: If the directory does not exist
        raise NotADirectoryError(directory)
    """
    if not directory.exists():
        raise DirectoryNotFoundError(directory)

    if not directory.is_dir():
        raise NotALocalDirectoryError(directory)

    pattern = "**/*.pdf" if recursive else "*.pdf"
    metadata_list = []

    for pdf_path in directory.glob(pattern):
        metadata = extract_pdf_metadata(pdf_path)
        metadata_list.append(metadata)

    return metadata_list


def extract_pdf_metadata(pdf_path: Path) -> PdfMetadata | None:
    """
    Extract metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        raise FileDoesNotExistError(pdf_path)

    Raises:
        raise NotAPdfFileError(pdf_path)
    """
    if not pdf_path.exists():
        raise FileDoesNotExistError(pdf_path)

    if pdf_path.suffix.lower() != ".pdf":
        raise NotAPdfFileError(pdf_path)

    from pypdf import PdfReader

    try:
        reader = PdfReader(pdf_path)
        info = reader.metadata if reader.metadata else {}
    except EmptyFileError as e:
        print(e)
        return None
    else:
        return PdfMetadata(
            filename=pdf_path.name,
            path=pdf_path,
            size_bytes=pdf_path.stat().st_size,
            created_at=datetime.fromtimestamp(pdf_path.stat().st_ctime),
            modified_at=datetime.fromtimestamp(pdf_path.stat().st_mtime),
            pages=len(reader.pages),
            author=info.get("/Author", None),
            title=str(info.get("/Title", None)),
            keywords=[k.strip() for k in info.get("/Keywords", "").split(",")] if info.get("/Keywords") else [],
            # version=reader.pdf_version,
            is_encrypted=reader.is_encrypted,
        )


if __name__ == "__main__":
    import sys

    directory = Path(sys.argv[1]).resolve()
    folder_metadata = scan_directory(directory)
    for fm in folder_metadata:
        if fm is not None:
            print_json(json.dumps(fm.model_dump()))
