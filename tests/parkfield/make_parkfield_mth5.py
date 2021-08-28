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


def create_from_iris(dataset_id):
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    inventory = dataset_config.get_inventory_from_iris(
        ensure_inventory_stages_are_named=True
    )
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)
    # experiment = get_experiment_from_obspy_inventory(inventory)
    run_metadata = experiment.surveys[0].stations[0].runs[0]
    target_folder = DATA_PATH
    h5_path = target_folder.joinpath(f"{dataset_config.dataset_id}.h5")
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)

    start_time = UTCDateTime("2004-09-28T00:00:00.000000Z")
    end_time = UTCDateTime("2004-09-28T01:59:59.975000Z")
    station_id = "PKD"  # station_id in mth5_obj.station_list
    network_id = "BK"
    channel_ids = "BT1,BT2,BQ2,BQ3"
    station_group = mth5_obj.get_station(station_id)

    client = fdsn.Client("IRIS")
    streams = client.get_waveforms(
        network_id, station_id, None, channel_ids, start_time, end_time
    )
    # <REASSIGN NON-CONVENTIONAL CHANNEL LABELS (Q2, Q3, T1, T2)>
    streams[0].stats["channel"] = "BQ1"
    streams[1].stats["channel"] = "BQ2"
    streams[2].stats["channel"] = "BF1"
    streams[3].stats["channel"] = "BF2"
    # </REASSIGN NON-CONVENTIONAL CHANNEL LABELS (Q2, Q3, T1, T2)>

    start_times = sorted(list(set([tr.stats.starttime.isoformat() for tr in streams])))
    end_times = sorted(list(set([tr.stats.endtime.isoformat() for tr in streams])))
    run_stream = streams.slice(UTCDateTime(start_times[0]), UTCDateTime(end_times[-1]))
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


def main():
    test_make_parkfield_mth5()


if __name__ == "__main__":
    main()
