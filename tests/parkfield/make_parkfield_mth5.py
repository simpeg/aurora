"""
Relates to github aurora issue #17
Ingest the Parkfield data make mth5 to use as the interface for tests
"""

from obspy import UTCDateTime
from obspy.clients import fdsn

from aurora.general_helper_functions import TEST_PATH
from aurora.sandbox.io_helpers.make_dataset_configs import TEST_DATA_SET_CONFIGS
from aurora.sandbox.mth5_helpers import initialize_mth5
from aurora.sandbox.mth5_helpers import test_can_read_back_data
from mth5.timeseries import RunTS
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment

DATA_PATH = TEST_PATH.joinpath("parkfield", "data")
DATA_PATH.mkdir(exist_ok=True)
FDSN_CHANNEL_MAP = {}
FDSN_CHANNEL_MAP["BQ2"] = "BQ1"
FDSN_CHANNEL_MAP["BQ3"] = "BQ2"
FDSN_CHANNEL_MAP["BT1"] = "BF1"
FDSN_CHANNEL_MAP["BT2"] = "BF2"
FDSN_CHANNEL_MAP["BT3"] = "BF3"

DATA_SOURCE = "IRIS"
# DATA_SOURCE = "NCEDC"
# def make_channels_fdsn_compliant(streams):


def create_from_iris(dataset_id):
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    inventory = dataset_config.get_inventory_from_iris(
        ensure_inventory_stages_are_named=True
    )
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)
    run_metadata = experiment.surveys[0].stations[0].runs[0]
    target_folder = DATA_PATH
    h5_path = target_folder.joinpath(f"{dataset_config.dataset_id}.h5")
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)
    print(
        f"station_id = {dataset_config.station}"
    )  # station_id in mth5_obj.station_list
    print(f"network_id = {dataset_config.network}")
    print(f"channel_ids = {dataset_config.channel_codes}")

    station_group = mth5_obj.get_station(dataset_config.station)

    client = fdsn.Client(DATA_SOURCE)
    streams = client.get_waveforms(
        dataset_config.network,
        dataset_config.station,
        None,
        dataset_config.channel_codes,
        dataset_config.starttime,
        dataset_config.endtime,
    )
    # <REASSIGN NON-CONVENTIONAL CHANNEL LABELS (Q2, Q3, T1, T2)>
    for stream in streams:
        stream.stats["channel"] = FDSN_CHANNEL_MAP[stream.stats["channel"]]
    # </REASSIGN NON-CONVENTIONAL CHANNEL LABELS (Q2, Q3, T1, T2)>

    # <This block is called often - should be a method>
    start_times = sorted(list(set([tr.stats.starttime.isoformat() for tr in streams])))
    end_times = sorted(list(set([tr.stats.endtime.isoformat() for tr in streams])))
    run_stream = streams.slice(UTCDateTime(start_times[0]), UTCDateTime(end_times[-1]))
    # </This block is called often - should be a method>
    run_ts_obj = RunTS()
    run_ts_obj.from_obspy_stream(run_stream, run_metadata)
    run_id = "001"
    run_ts_obj.run_metadata.id = run_id
    run_group = station_group.add_run(run_id)
    run_group.from_runts(run_ts_obj)
    mth5_obj.close_mth5()

    return


def test_make_parkfield_mth5():
    dataset_id = "pkd_test_00"
    create_from_iris(dataset_id)
    h5_path = DATA_PATH.joinpath(f"{dataset_id}.h5")
    test_can_read_back_data(h5_path, "PKD", "001")


# <TEST FAILS BECAUSE DATA NOT AVAILABLE FROM IRIS -- NEED TO REQUEST FROM NCEDC>
# def test_make_hollister_mth5():
#     dataset_id = "sao_test_00"
#     create_from_iris(dataset_id)
#     h5_path = DATA_PATH.joinpath(f"{dataset_id}.h5")
#     test_can_read_back_data(h5_path, "SAO", "001")
# </TEST FAILS BECAUSE DATA NOT AVAILABLE FROM IRIS -- NEED TO REQUEST FROM NCEDC>


def main():
    test_make_parkfield_mth5()
    # test_make_hollister_mth5()


if __name__ == "__main__":
    main()
