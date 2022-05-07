from pathlib import Path

from aurora.config import Processing


def initialize_config(processing_config):
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