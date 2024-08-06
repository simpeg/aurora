"""
    This module contains helper functions to control where the parkfield test data
    and test results are stored /accessed.
"""
from aurora.general_helper_functions import DATA_PATH


def make_parkfield_paths() -> dict:
    """
    Makes a dictionary with information about where to store/access PKD test data and results.

    Returns
    -------
    parkfield_paths: dict
        Dict containing paths to "data", "aurora_results", "config", "emtf_results"
    """
    base_path = DATA_PATH.joinpath("parkfield")
    parkfield_paths = {}
    parkfield_paths["data"] = base_path
    parkfield_paths["aurora_results"] = base_path.joinpath("aurora_results")
    parkfield_paths["config"] = base_path.joinpath("config")
    parkfield_paths["emtf_results"] = base_path.joinpath("emtf_results")
    return parkfield_paths


PARKFIELD_PATHS = make_parkfield_paths()
