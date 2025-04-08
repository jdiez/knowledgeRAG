from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_serializer
from pydantic_core import PlainSerializer


class PdfMetadata(BaseModel):
    """Model representing metadata for a PDF document."""

    model_config = ConfigDict(str_strip_whitespace=True)

    filename: str = Field(description="Name of the PDF file")
    # path: Path = Field(description="Full path to the PDF file")
    path: Annotated[Path, PlainSerializer(lambda x: str(x), return_type=str)]
    size_bytes: int = Field(description="Size of the file in bytes")
    created_at: datetime = Field(description="File creation timestamp")
    modified_at: datetime = Field(description="Last modification timestamp")
    pages: int = Field(description="Number of pages in the PDF")
    author: str | None = Field(default=None, description="Author of the PDF document")
    title: str | None = Field(default=None, description="Title of the PDF document")
    keywords: list[str] = Field(default_factory=list, description="Keywords associated with the document")
    version: str | None = Field(default=None, description="PDF version")
    is_encrypted: bool = Field(default=False, description="Whether the PDF is encrypted")

    @field_serializer("created_at", "modified_at")
    def serialize_dt(self, dt: datetime) -> str:
        """AI is creating summary for serialize_dt

        Args:
            dt (datetime): [description]

        Returns:
            str: [description]
        """
        return dt.strftime("%m/%d/%Y-%H:%M:%S")
