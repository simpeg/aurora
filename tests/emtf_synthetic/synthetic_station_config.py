from aurora.sandbox.processing_config import ProcessingConfig
from aurora.time_series.filters.filter_helpers import make_coefficient_filter
from pathlib import Path

#<FILTERS>
ACTIVE_FILTERS = []
unity_coeff_filter = make_coefficient_filter(name="1", gain=1.0)
cf_multipy_10 = make_coefficient_filter(gain=10.0, name="10")
cf_divide_10= make_coefficient_filter(gain=0.1, name="10")
UNITS = "MT"
ACTIVE_FILTERS = [unity_coeff_filter, cf_multipy_10, cf_divide_10]
#</FILTERS>

# class StationConfig(object):
#     def __init__(self, **kwargs):
#         self.raw_data_path = kwargs.get("raw_data_path", None)
#         self.columns = ["hx", "hy", "hz", "ex", "ey"]
#         self.mth5_path = kwargs.get("mth5_path", None)
#
#         #<depends on columns>
#         self.noise_scalar = {}
#         for col in self.columns:
#             self.noise_scalar[col] = 0.0
#
#         #</depends on columns>

#<MTH5 CREATION CONFIG>
#make this an object? or leave as dict?
STATION_01_CFG = {}
STATION_01_CFG["raw_data_path"] = Path("data","test1.asc")
STATION_01_CFG["mth5_path"] = Path("data", "test1.h5")

STATION_01_CFG["columns"] = ["hx", "hy", "hz", "ex", "ey"]
STATION_01_CFG["noise_scalar"] = {}
for col in STATION_01_CFG["columns"]:
    STATION_01_CFG["noise_scalar"][col] = 0.0
STATION_01_CFG["filters"] = {}
# for col in STATION_01_CFG["columns"]:
#     STATION_01_CFG["filters"][col] = []
for col in STATION_01_CFG["columns"]:
    if col in ["ex", "ey"]:
        STATION_01_CFG["filters"][col] = [unity_coeff_filter.name,]
for col in STATION_01_CFG["columns"]:
    if col in ["hx", "hy", "hz"]:
        STATION_01_CFG["filters"][col] = [cf_divide_10.name,cf_multipy_10.name]
STATION_01_CFG["run_id"] = "001"
STATION_01_CFG["station_id"] = "test1"
STATION_01_CFG["sample_rate"] = 1.0

STATION_02_CFG = STATION_01_CFG.copy()
STATION_02_CFG["raw_data_path"] = Path("data","test2.asc")
STATION_02_CFG["mth5_path"] = Path("data", "test2.h5")
STATION_02_CFG["station_id"] = "test2"

#</MTH5 CREATION CONFIG>
