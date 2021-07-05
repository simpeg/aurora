from aurora.sandbox.processing_config import ProcessingConfig
from filter_helpers import make_coefficient_filter
from pathlib import Path

#<FILTERS>
ACTIVE_FILTERS = []
cf1 = make_coefficient_filter(name="1")
cf10 = make_coefficient_filter(gain=1, name="10")
UNITS = "MT"
ACTIVE_FILTERS = [cf1, cf10]
#</FILTERS>

#<MTH5 CREATION CONFIG>
# def get_mth5_config(test_case_id):
#     if test_case_id=="test1":
#         cfg = {}
#make this an object? or leave as dict?
STATION_01_CFG = {}
STATION_01_CFG["raw_data_path"] = Path(r"test1.asc")
STATION_01_CFG["mth5_path"] = Path(r"test1.h5")
STATION_01_CFG["columns"] = ["hx", "hy", "hz", "ex", "ey"]
#STATION_01_CFG["columns"] = ["hz", "hx", "hy", "ex", "ey"]
STATION_01_CFG["noise_scalar"] = {}
for col in STATION_01_CFG["columns"]:
    STATION_01_CFG["noise_scalar"][col] = 0.0
STATION_01_CFG["filters"] = {}
for col in STATION_01_CFG["columns"]:
    STATION_01_CFG["filters"][col] = []
for col in STATION_01_CFG["columns"]:
    if col in ["ex", "ey"]:
        STATION_01_CFG["filters"][col] = [cf1.name,]
for col in STATION_01_CFG["columns"]:
    if col in ["hx", "hy", "hz"]:
        STATION_01_CFG["filters"][col] = [cf1.name,cf10.name]
STATION_01_CFG["run_id"] = "001"
STATION_01_CFG["station_id"] = "test1"
STATION_01_CFG["sample_rate"] = 1.0

STATION_02_CFG = STATION_01_CFG.copy()
STATION_02_CFG["raw_data_path"] = Path(r"test2.asc")
STATION_02_CFG["mth5_path"] = Path(r"test2.h5")
STATION_02_CFG["station_id"] = "test2"

#</MTH5 CREATION CONFIG>
