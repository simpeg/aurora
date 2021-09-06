# TODO: Move this into tests/parkfield after factoring out FAP test
from obspy import UTCDateTime
from aurora.sandbox.io_helpers.iris_dataset_config import IRISDatasetConfig

HEXY = ["hx", "hy", "ex", "ey"]  # default components list
# <CREATE TEST CONFIGS>


def make_test_configs():
    test_data_set_configs = {}

    # <pkd_test_00 Single station>
    test_data_set = IRISDatasetConfig()
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
    test_data_set.components_list = HEXY

    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    # </pkd_test_00 Single station>

    # <sao_test_00 Single station>
    test_data_set = IRISDatasetConfig()
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

    return test_data_set_configs


TEST_DATA_SET_CONFIGS = make_test_configs()
