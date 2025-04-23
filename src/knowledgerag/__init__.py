# process configuration
import warnings
from pathlib import Path
from typing import Any

import yaml

warnings.filterwarnings("ignore")
configuration = Path(__file__).parent / Path("config/config.yaml")


def read_configuration(configuration_file: Path = configuration) -> dict[str, Any]:
    """_summary_

    Args:
        configuration_file (Path, optional): _description_. Defaults to configuration.

    Returns:
        dict[str, Any]: _description_
    """
    with open(configuration.resolve()) as file:
        config = yaml.safe_load(file)
    return config


# set environmental variables: ...
