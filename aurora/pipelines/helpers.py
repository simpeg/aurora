from deprecated import deprecated
from pathlib import Path

from aurora.config.processing_config import RunConfig

@deprecated(version="0.0.3", reason="new mt_metadata based config")
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
    else:
        print(f"Unrecognized config of type {type(run_config)}")
        raise Exception
    return config
