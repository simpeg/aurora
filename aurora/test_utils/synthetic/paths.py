from aurora.general_helper_functions import TEST_PATH

SYNTHETIC_PATH = TEST_PATH.joinpath("synthetic")
CONFIG_PATH = SYNTHETIC_PATH.joinpath("config")
DATA_PATH = SYNTHETIC_PATH.joinpath("data")
EMTF_OUTPUT_PATH = SYNTHETIC_PATH.joinpath("emtf_output")
AURORA_RESULTS_PATH = SYNTHETIC_PATH.joinpath("aurora_results")

AURORA_RESULTS_PATH.mkdir(exist_ok=True)
CONFIG_PATH.mkdir(exist_ok=True)
