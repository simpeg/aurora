from pathlib import Path

from aurora.config.processing_config import RunConfig
from aurora.config.metadata.processing import Processing

from aurora.config import Processing


def initialize_config(run_config):
    """

    Parameters
    ----------
    processing_cfg: path or str

    Returns :
    -------

    """
    if isinstance(run_config, Path) or isinstance(run_config, str):
        config = RunConfig()
        config.from_json(run_config)
    elif isinstance(run_config, RunConfig):
        config = run_config
        print("ToBeDeprecated")
    elif isinstance(run_config, Processing):
        config = run_config
    else:
        print(f"Unrecognized config of type {type(run_config)}")
        raise Exception
    return config

def initialize_config_object(processing_config):
    """

    Parameters
    ----------
    processing_cfg: path or str

    Returns :
    -------

    """
    if isinstance(processing_config, (Path, str)):
        config = Processing()
        config.from_json(processing_config)
    elif isinstance(processing_config, Processing):
        config = processing_config
    else:
        raise Exception(f"Unrecognized config of type {type(processing_config)}")
    return config