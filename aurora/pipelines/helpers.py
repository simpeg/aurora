from pathlib import Path

from mt_metadata.transfer_functions.processing.aurora import Processing


def initialize_config(processing_config):
    """

    Parameters
    ----------
    processing_cfg: path or str
        Either an instance of the processing class or a path to a json file that a
        Processing object is stored in.
    Returns :
    -------
    config: mt_metadata.transfer_functions.processing.aurora.Processing
        Object that contains the processing parameters
    """
    if isinstance(processing_config, (Path, str)):
        config = Processing()
        config.from_json(processing_config)
    elif isinstance(processing_config, Processing):
        config = processing_config
    else:
        raise Exception(f"Unrecognized config of type {type(processing_config)}")
    return config
