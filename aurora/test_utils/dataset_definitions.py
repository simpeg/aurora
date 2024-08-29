"""
    This module contains methods that are used to define datasets to build from FDSN servers.

    These datasets are in turn used for testing.

"""
from obspy import UTCDateTime
from aurora.sandbox.io_helpers.fdsn_dataset import FDSNDataset


def make_pkdsao_test_00_config(minitest=False) -> FDSNDataset:
    """
    Return a description of a 2h PKD SAO 40Hz dataset from NCEDC.

    Parameters
    ----------
    minitest: bool
        Used for debugging, when set to True, it will just get a minute of data

    Returns
    -------
    test_data_set: aurora.sandbox.io_helpers.fdsn_dataset.FDSNDataset

    """
    test_data_set = FDSNDataset()
    test_data_set.dataset_id = "pkd_sao_test_00"
    test_data_set.network = "BK"
    test_data_set.station = "PKD,SAO"
    test_data_set.starttime = UTCDateTime("2004-09-28T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2004-09-28T01:59:59.975000Z")
    if minitest:
        test_data_set.endtime = UTCDateTime("2004-09-28T00:01:00")  # 1 min
    test_data_set.channel_codes = "BQ2,BQ3,BT1,BT2,BT3"
    test_data_set.description = "2h of PKD,SAO data for 2004-09-28 0000-0200 UTC"
    test_data_set.components_list = ["ex", "ey", "hx", "hy", "hz"]
    return test_data_set


def make_cas04_nvr08_test_00_config() -> FDSNDataset:
    """
    Return a description of a CAS04,NVR08 dataset from IRIS.

    Returns
    -------

    """
    test_data_set = FDSNDataset()
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
    test_data_set.components_list = ["hx", "hy", "ex", "ey"]
    return test_data_set


def make_iak34_test_00_config() -> FDSNDataset:
    """Return a description of a IAK34 dataset from IRIS."""
    test_data_set = FDSNDataset()
    test_data_set.dataset_id = "iak34_test_00"
    test_data_set.network = "EM"
    test_data_set.station = "IAK34"
    # <ORIGINAL>
    # test_data_set.starttime = UTCDateTime("2013-04-25T20:10:08.000000Z")
    # test_data_set.endtime = UTCDateTime("2013-05-13T21:18:53.000000Z")
    # </ORIGINAL>
    test_data_set.starttime = UTCDateTime("2013-04-26T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2013-05-12T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2013-04-27T00:00:00.000000Z")
    test_data_set.channel_codes = None
    test_data_set.description = "earthscope example dataset IAK34"
    test_data_set.components_list = ["hx", "hy", "ex", "ey"]
    return test_data_set


def make_iak34_test_01_config() -> FDSNDataset:
    """Return a description of a IAK34 dataset from IRIS."""
    test_data_set = FDSNDataset()
    test_data_set.dataset_id = "iak34_test_01_long_ss"
    test_data_set.network = "EM"
    test_data_set.station = "IAK34"
    # <ORIGINAL>
    # test_data_set.starttime = UTCDateTime("2013-04-25T20:10:08.000000Z")
    # test_data_set.endtime = UTCDateTime("2013-05-13T21:18:53.000000Z")
    # </ORIGINAL>
    test_data_set.starttime = UTCDateTime("2013-04-26T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2013-05-11T00:00:00.000000Z")
    test_data_set.channel_codes = None
    test_data_set.description = "earthscope example dataset IAK34"
    test_data_set.components_list = ["hx", "hy", "ex", "ey"]
    return test_data_set


def make_iak34_test_02_config() -> FDSNDataset:
    """Return a description of a IAK34 dataset from IRIS."""
    test_data_set = FDSNDataset()
    test_data_set.dataset_id = "iak34_test_02_long_rr"
    test_data_set.network = "EM"
    test_data_set.station = "IAK34,NEK33"
    # <ORIGINAL>
    # test_data_set.starttime = UTCDateTime("2013-04-25T20:10:08.000000Z")
    # test_data_set.endtime = UTCDateTime("2013-05-13T21:18:53.000000Z")
    # </ORIGINAL>
    test_data_set.starttime = UTCDateTime("2013-04-26T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2013-05-10T00:00:00.000000Z")
    test_data_set.channel_codes = None
    test_data_set.description = "earthscope example dataset IAK34"
    test_data_set.components_list = ["hx", "hy", "ex", "ey"]
    return test_data_set


def make_iak34_test_03_config() -> FDSNDataset:
    """Return a description of a IAK34 dataset from IRIS."""
    test_data_set = FDSNDataset()
    test_data_set.dataset_id = "iak34_test_03_long_rr"
    test_data_set.network = "EM"
    test_data_set.station = "IAK34,NEK33"
    # <ORIGINAL>
    # test_data_set.starttime = UTCDateTime("2013-04-25T20:10:08.000000Z")
    # test_data_set.endtime = UTCDateTime("2013-05-13T21:18:53.000000Z")
    # </ORIGINAL>
    test_data_set.starttime = UTCDateTime("2013-05-15T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2013-05-26T00:00:00.000000Z")
    test_data_set.channel_codes = None
    test_data_set.description = "earthscope example dataset IAK34"
    test_data_set.components_list = ["hx", "hy", "ex", "ey"]
    return test_data_set


def make_iak34_test_04_config() -> FDSNDataset:
    """Return a description of a IAK34 dataset from IRIS."""
    test_data_set = FDSNDataset()
    test_data_set.dataset_id = "iak34_test_04_rr"
    test_data_set.network = "EM"
    test_data_set.station = "IAK34,NEN34"  # NEK33
    # <ORIGINAL>
    # test_data_set.starttime = UTCDateTime("2013-04-25T20:10:08.000000Z")
    # test_data_set.endtime = UTCDateTime("2013-05-13T21:18:53.000000Z")
    # </ORIGINAL>
    test_data_set.starttime = UTCDateTime("2013-04-28T00:00:00.000000Z")
    test_data_set.endtime = UTCDateTime("2013-04-29T00:00:00.000000Z")
    test_data_set.channel_codes = None
    test_data_set.description = "earthscope example dataset IAK34"
    test_data_set.components_list = ["hx", "hy", "ex", "ey"]
    return test_data_set


def make_test_configs() -> dict:
    """Make all the test dataset configs and put them in a dict"""
    test_data_set_configs = {}

    # pkd_sao_test_00 Remote Reference
    test_data_set = make_pkdsao_test_00_config()
    test_data_set_configs[test_data_set.dataset_id] = test_data_set

    # cas_nvr_test_00 Remote Reference
    test_data_set = make_cas04_nvr08_test_00_config()
    test_data_set_configs[test_data_set.dataset_id] = test_data_set

    # IAK34SS
    test_data_set = make_iak34_test_00_config()
    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    test_data_set = make_iak34_test_01_config()
    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    test_data_set = make_iak34_test_02_config()
    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    test_data_set = make_iak34_test_03_config()
    test_data_set_configs[test_data_set.dataset_id] = test_data_set
    test_data_set = make_iak34_test_04_config()
    test_data_set_configs[test_data_set.dataset_id] = test_data_set

    return test_data_set_configs


TEST_DATA_SET_CONFIGS = make_test_configs()
