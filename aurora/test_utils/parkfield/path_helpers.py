from aurora.general_helper_functions import TEST_PATH

PARKFIELD_PATH = TEST_PATH.joinpath("parkfield")

AURORA_RESULTS_PATH = PARKFIELD_PATH.joinpath("aurora_results")
CONFIG_PATH = PARKFIELD_PATH.joinpath("config")
DATA_PATH = PARKFIELD_PATH.joinpath("data")
EMTF_RESULTS_PATH = PARKFIELD_PATH.joinpath("emtf_results")

# May want to create results and data dir on init
AURORA_RESULTS_PATH.mkdir(exist_ok=True)
DATA_PATH.mkdir(exist_ok=True)
CONFIG_PATH.mkdir(exist_ok=True)
