from importlib import import_module
from pathlib import Path

import click
import lancedb
from loguru import logger

from knowledgerag.io.reader.reader import main_file_reader

MODULES = "knowledgerag.models.file"


@click.command()
@click.option("--db-path", type=click.Path(), default="example.db", help="Path to the database")
@click.option("--table-name", type=click.Path(), default="example_table", help="Name of the table")
@click.option("--schema_name", default="FileDescription", help="Schema of the table")
@click.option("--mode", default="overwrite", help="Mode of the table")
@click.option("--data-dir", type=click.Path(), default=None, help="Path to the data directory")
def main_ingestion(db_path: str, table_name: str, schema_name: str, mode: str, data_dir: str) -> None:
    """Ingest data from files into a LanceDB table.

    Args:
        db_path (str): [description]
        table_name (str): [description]
        schema_name (str): [description]
        mode (str): [description]
        data_dir (str): [description]
    """
    schema = getattr(import_module(MODULES), schema_name)
    database_path = Path(db_path).resolve()
    try:
        db = lancedb.connect(database_path)
        table_ = db.create_table(table_name, schema=schema, mode=mode)
    except Exception as e:
        click.echo(f"Database error: {e}")
        return
    else:
        if data_dir is not None:
            try:
                table_.add(list(main_file_reader(Path(data_dir))))
            except lancedb.TableError as e:
                logger.error(f"Table error: {e}")
                return
            else:
                logger.info(f"Data added to table {table_name} successfully.")


if __name__ == "__main__":
    main_ingestion()
