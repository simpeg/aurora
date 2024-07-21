"""

    Helper functions for processing pipelines.
    These maybe reorganized later into other modules.

"""

from mt_metadata.transfer_functions.processing.aurora import Processing
from typing import Union
import pathlib


def initialize_config(
    processing_config: Union[Processing, str, pathlib.Path]
) -> Processing:
    """
    Helper function to return an intialized processing config.

    Parameters
    ----------
    processing_cfg: Union[Processing, str, pathlib.Path]
        Either an instance of the processing class or a path to a json file that a
        Processing object is stored in.

    Returns
    -------
    config: mt_metadata.transfer_functions.processing.aurora.Processing
        Object that contains the processing parameters
    """
    if isinstance(processing_config, (pathlib.Path, str)):
        config = Processing()
        config.from_json(processing_config)
    elif isinstance(processing_config, Processing):
        config = processing_config
    else:
        raise TypeError(f"Unrecognized config of type {type(processing_config)}")
    return config
