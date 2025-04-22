import hashlib
from collections.abc import Generator
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from loguru import logger

from knowledgerag.models.file import FileDescription  # Adjust the import path as per the actual module structure


class PathError(Exception):
    """Exception raised for custom error in the application."""

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __str__(self):
        return f"{self.path} is not valid."


class DirectoryFileIterator:
    """Iterate over a folder tree and get files by extension.
    It is used to create a file description data structure.
    """

    def __init__(
        self,
        source: str | Path,
        descriptive_function: Callable[
            [
                Path,
            ],
            FileDescription,
        ]
        | None = None,
        allowed_file_types: list[str] | None = None,
    ) -> None:
        """Iterates over a folder tree and get files by extension.

        Args:
            source (str): [description]
            descriptive_function (Callable[[Path, ], BaseModel]): [description]
            allowed_file_types (list[str], optional): [description]. Defaults to None.
        """
        self.source = Path(source).resolve()
        self.__check_root()
        self.allowed_file_types = (
            [i.lower() for i in allowed_file_types] if allowed_file_types is not None else allowed_file_types
        )
        # If not descriptive function it returns file path.
        self.descriptive_function = descriptive_function if descriptive_function else lambda x: x

    def __check_root(self) -> None:
        """_summary_"""
        if self.source.is_file():
            self.source = self.source.parent
        if not self.source.exists():
            raise PathError(self.source)

    def get_file_struct(self, filename: Path) -> FileDescription:
        """Get file description data structure.

        Args:
            filename (str): [description]

        Returns:
            FileDescription: [description]
        """
        description = self.descriptive_function(filename)
        return description

    def get_files(self) -> Generator[FileDescription, None, None]:
        """Creates a file generator.

        Yields:
            Generator: [description]
        """
        for child in self.source.rglob("*"):
            if child.is_file():
                suffix = child.suffix.lower()
                match self.allowed_file_types:
                    case list():
                        if suffix in self.allowed_file_types:
                            yield self.get_file_struct(child)
                    case None:
                        yield self.get_file_struct(child)
                    case _:
                        raise ValueError

    def __iter__(self) -> "DirectoryFileIterator":
        """Return class instance. A generator is created.
        This allows the use of the for loop.

        Yields:
            DirectoryFileIterator: [description]
        """
        self.files = self.get_files()
        return self

    def __next__(self) -> FileDescription:
        """ """
        value = next(self.files)
        return value


def timeConvert(atime: float) -> date:
    """It converts a timestamp to a date.
    It is used to convert the last access date, last modification date,
    and metadata date.

    Args:
        atime (float): [description]

    Returns:
        date: [description]
    """
    dt = atime
    newtime = datetime.fromtimestamp(dt)
    return newtime


def sizeFormat(size: int) -> str:
    """ """
    """AI is creating summary for sizeFormat

    Args:
        size (int): [description]

    Returns:
        str: [description]
    """
    newform = format(size / 1024, ".2f")
    return newform + " kb"


def calculate_checksum_info(filepath: str | Path) -> str:
    """Calculate file information._summary_

    Args:
        filepath (str): [description]

    Returns:
        str: [description]
    """

    try:
        with open(filepath, "rb") as f:
            sha256 = hashlib.sha256()
            while True:
                chunk = f.read(16 * 1024)
                if not chunk:
                    break
                sha256.update(chunk)
    except Exception as e:
        logger.error(f"Error calculating checksum for {filepath}: {e}")

    return sha256.hexdigest()


def reporting_file_info(filepath: str | Path, hashing_function: Callable = calculate_checksum_info) -> FileDescription:
    """AI is creating summary for get_file_info

    Args:
        filepath (str): [description]

    Returns:
        FileDescription: [description]
    """
    a_path = Path(filepath).resolve()
    stats = a_path.stat()
    name = a_path.name
    hash_value = hashing_function(a_path)
    suffix = a_path.suffix
    result = FileDescription(
        path=a_path,
        filename=name,
        owner=a_path.owner(),
        group=a_path.group(),
        mode=oct(stats.st_mode),
        uid=stats.st_uid,
        gid=stats.st_gid,
        permissions=oct(stats.st_mode),
        suffix=suffix,
        hash_value=hash_value,
        file_name=name,
        size_kb=sizeFormat(stats.st_size),
        mmetadata_date=timeConvert(stats.st_ctime),
        modified_date=timeConvert(stats.st_mtime),
        last_access_date=timeConvert(stats.st_atime),
        query_date=datetime.now(),
    )
    return result


def file_reader(
    path: str | Path, allowed_file_types: list[str] | None = None, descriptive_function: Callable | None = None
) -> Generator[FileDescription, None, None]:
    """AI is creating summary for main

    Args:
        path (str): [description]

    Returns:
        FileDescription: [description]
    """
    if allowed_file_types is None:
        allowed_file_types = [".pdf", ".py"]
    f = Path(path).resolve()
    result = DirectoryFileIterator(
        f,
        allowed_file_types=allowed_file_types,
        descriptive_function=descriptive_function,
    ).get_files()
    return result


if __name__ == "__main__":
    """Test the code.
    It is used to test the code.
    """
    from rich import print_json

    root_path = Path(__file__).resolve()
    result = file_reader(root_path)
    for file_rep in result:
        print_json(file_rep.model_dump_json())
