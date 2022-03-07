from aurora.general_helper_functions import BAND_SETUP_PATH

BANDS_DEFAULT_FILE = BAND_SETUP_PATH.joinpath("bs_test.cfg")
BANDS_256_FILE = BAND_SETUP_PATH.joinpath("bs_256.cfg")



from .metadata import (
    Window, 
    Station, 
    Channel,
    Run,
    Stations,
    Band,
    Decimation, 
    Regression,
    Estimator,
    DecimationLevel,
    Processing,
)

__all__ = [
    "Window",
    "Station",
    "Channel",
    "Run",
    "Stations",
    "Band",
    "Decimation",
    "Regression",
    "Estimator",
    "DecimationLevel",
    "Processing",
    ]
    
 