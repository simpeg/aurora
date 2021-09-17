# TODO: Move this into tests/parkfield after factoring out FAP test
from obspy import UTCDateTime
from aurora.sandbox.io_helpers.fdsn_dataset_config import FDSNDatasetConfig

HEXY = ["hx", "hy", "ex", "ey"]  # default components list
# <CREATE TEST CONFIGS>


def make_pkd_test_00_config():
    test_data_set = FDSNDatasetConfig()
    test_data_set.dataset_id = "pkd_test_00"
    test_data_set.network = "BK"
    test_data_set.station = "PKD"
    test_data_set.starttime = UTCDateTime("2004-09-28T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2004-09-28T01:59:59.975000Z")
    # test_data_set.starttime = UTCDateTime("2004-09-28T00:00:00")
    # test_data_set.endtime = UTCDateTime("2004-09-28T23:59:59")
    # test_data_set.channel_codes = "LQ2,LQ3,LT1,LT2"
    test_data_set.channel_codes = "BQ2,BQ3,BT1,BT2"
    test_data_set.description = "2h of PKD data for 2004-09-28 midnight UTC until 0200"
    test_data_set.components_list = ["ex", "ey", "hx", "hy"]
    return test_data_set


def make_test_configs():
    test_data_set_configs = {}

    # <pkd_test_00 Single station>
    test_data_set = make_pkd_test_00_config()
    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    # </pkd_test_00 Single station>

    # <sao_test_00 Single station>
    test_data_set = FDSNDatasetConfig()
    test_data_set.dataset_id = "sao_test_00"
    test_data_set.network = "BK"
    test_data_set.station = "SAO"
    test_data_set.starttime = UTCDateTime("2004-09-28T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2004-09-28T01:59:59.975000Z")
    # test_data_set.starttime = UTCDateTime("2004-09-28T00:00:00")
    # test_data_set.endtime = UTCDateTime("2004-09-28T23:59:59")
    # test_data_set.channel_codes = "LQ2,LQ3,LT1,LT2"
    test_data_set.channel_codes = "BQ2,BQ3,BT1,BT2"
    test_data_set.description = "2h of SAO data for 2004-09-28 midnight UTC until 0200"
    test_data_set.components_list = HEXY
    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    # </sao_test_00 Single station>

    # <pkd_sao_test_00 Remote Reference>
    test_data_set = FDSNDatasetConfig()
    test_data_set.dataset_id = "pkd_sao_test_00"
    test_data_set.network = "BK"
    test_data_set.station = "PKD,SAO"
    test_data_set.starttime = UTCDateTime("2004-09-28T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2004-09-28T01:59:59.975000Z")
    # test_data_set.endtime = UTCDateTime("2004-09-28T00:01:59.999000Z") #small test
    test_data_set.channel_codes = "BQ2,BQ3,BT1,BT2"
    test_data_set.description = (
        "2h of PKD,SAO data for 2004-09-28 midnight UTC until 0200"
    )
    test_data_set.components_list = HEXY
    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    # </sao_test_00 Single station>

    # <cas_nvr_test_00 Remote Reference>
    test_data_set = FDSNDatasetConfig()
    test_data_set.dataset_id = "cas_nvr_test_00"
    test_data_set.network = "ZU"
    test_data_set.station = "CAS04,NVR08"
    # <ORIGINAL>
    # test_data_set.starttime = UTCDateTime("2020-06-02T18:41:43.000000Z")
    # test_data_set.endtime = UTCDateTime("2020-07-13T21:46:12.000000Z")
    # </ORIGINAL>
    test_data_set.starttime = UTCDateTime("2020-06-04T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2020-06-05T00:00:00.000000Z")  # minitest
    # test_data_set.endtime = UTCDateTime("2020-06-24T15:55:46.000000Z")

    # test_data_set.starttime = UTCDateTime("2004-09-28T00:00:00.000000Z")
    # test_data_set.endtime = UTCDateTime("2004-09-28T01:59:59.975000Z")
    # test_data_set.endtime = UTCDateTime("2004-09-28T00:01:59.999000Z") #small test
    test_data_set.channel_codes = None
    test_data_set.description = "earthscope example dataset"
    test_data_set.components_list = HEXY
    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    # </cas_nvr_test_00 Remote Reference>
    return test_data_set_configs


TEST_DATA_SET_CONFIGS = make_test_configs()
