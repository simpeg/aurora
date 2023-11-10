from aurora.general_helper_functions import DATA_PATH


def make_parkfield_paths():
    base_path = DATA_PATH.joinpath("parkfield")
    parkfield_paths = {}
    parkfield_paths["data"] = base_path
    parkfield_paths["aurora_results"] = base_path.joinpath("aurora_results")
    parkfield_paths["config"] = base_path.joinpath("config")
    parkfield_paths["emtf_results"] = base_path.joinpath("emtf_results")
    return parkfield_paths


PARKFIELD_PATHS = make_parkfield_paths()
