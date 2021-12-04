from pathlib import Path

from aurora.config.processing_config import RunConfig


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
    else:
        print(f"Unrecognized config of type {type(run_config)}")
        raise Exception
    return config