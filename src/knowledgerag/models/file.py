from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field


class FileDescription(BaseModel):
    """It is used to create a file description data structure.
    It contains the file path, file name, owner, group, mode, uid, gid,
    permissions, suffix, sha256, size_kb, metadata date, modified date,
    last access date, query date, pages, author, description, keywords,
    version, and is_encrypted.

    Args:
        BaseModel ([type]): [description]
    """

    path: Annotated[Path, Field(description="File path.")]
    filename: Annotated[str, Field(description="Filename.")]
    owner: Annotated[str, Field(description="File owner.")]
    group: Annotated[str, Field(description="Group.")]
    mode: Annotated[str, Field(description="File mode.")]
    uid: Annotated[int, Field(description="User ID.")]
    gid: Annotated[int, Field(description="Group ID.")]
    permissions: Annotated[str, Field(description="File permission string.")]
    suffix: Annotated[str, Field(description="File suffix.")]
    hash_value: Annotated[str, Field(description="File hashing result.")]
    size_kb: Annotated[str, Field(description="File size in KB.")]
    mmetadata_date: Annotated[datetime, Field(description="Metadata date.")]
    modified_date: Annotated[datetime, Field(description="File modified date.")]
    last_access_date: Annotated[datetime, Field(description="File last access date.")]
    query_date: Annotated[datetime, Field(description="Last query date.")]
    pages: Annotated[int | None, Field(default=None, description="Number of pages.")]
    author: Annotated[str | None, Field(default=None, description="File / document author.")]
    keywords: Annotated[list[str] | None, Field(default=None, descriptions="File keywords.")]
    version: Annotated[str | None, Field(default=None, description="File version.")]
    is_encrypted: Annotated[bool | None, Field(default=False, description="Is file encrypted.")]
