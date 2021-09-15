"""
Relates to github aurora issue #17
Ingest the Parkfield data make mth5 to use as the interface for tests
"""

from obspy import UTCDateTime
from obspy.clients import fdsn

from aurora.general_helper_functions import TEST_PATH
from aurora.sandbox.io_helpers.make_dataset_configs import TEST_DATA_SET_CONFIGS
from aurora.pipelines.helpers import initialize_mth5
from aurora.pipelines.helpers import read_back_data
from mth5.timeseries import RunTS
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from aurora.time_series.filters.filter_helpers import MT2SI_ELECTRIC_FIELD_FILTER

DATA_PATH = TEST_PATH.joinpath("parkfield", "data")
DATA_PATH.mkdir(exist_ok=True)
FDSN_CHANNEL_MAP = {}
FDSN_CHANNEL_MAP["BQ2"] = "BQ1"
FDSN_CHANNEL_MAP["BQ3"] = "BQ2"
FDSN_CHANNEL_MAP["BT1"] = "BF1"
FDSN_CHANNEL_MAP["BT2"] = "BF2"
FDSN_CHANNEL_MAP["BT3"] = "BF3"

# def make_channels_fdsn_compliant(streams):


def create_from_server(dataset_id, data_source="IRIS"):
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    inventory = dataset_config.get_inventory_from_iris(
        base_url=data_source, ensure_inventory_stages_are_named=True
    )
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)
    print(
        f"ADD MT2SI_ELECTRIC_FIELD_FILTER to electric channels for parkfield here"
        f" {MT2SI_ELECTRIC_FIELD_FILTER} "
    )
    # survey = experiment.surveys[0]
    # survey.filters["MT2SI Electric Field"] = MT2SI_ELECTRIC_FIELD_FILTER
    # survey.stations[0].runs[0].channels[0].filter.update(
    #    MT2SI_ELECTRIC_FIELD_FILTER)
    # survey.stations[0].runs[0].channels[1].filter.update(
    # MT2SI_ELECTRIC_FIELD_FILTER)
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

    client = fdsn.Client(data_source)
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
        # #<UNITS HACK>
        # if stream.stats["channel"][1] == "Q":
        #     print("WARNING - HANDLE INCORRECT UNITS TRANSLATE V/m to mV/km")
        #     stream.data *= 1000000
        # #</UNITS HACK>
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


def create_from_server_multistation(dataset_id, data_source="IRIS"):
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    # dataset_config.station="PKD"#debuging
    inventory = dataset_config.get_inventory_from_iris(
        ensure_inventory_stages_are_named=True,
        base_url=data_source,
    )
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)

    target_folder = DATA_PATH
    h5_path = target_folder.joinpath(f"{dataset_config.dataset_id}.h5")
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)
    print(
        f"station_id = {dataset_config.station}"
    )  # station_id in mth5_obj.station_list
    print(f"network_id = {dataset_config.network}")
    print(f"channel_ids = {dataset_config.channel_codes}")

    client = fdsn.Client(data_source)
    streams = client.get_waveforms(
        dataset_config.network,
        dataset_config.station,
        None,
        dataset_config.channel_codes,
        dataset_config.starttime,
        dataset_config.endtime,
    )
    # <REASSIGN NON-CONVENTIONAL CHANNEL LABELS (Q2, Q3, T1, T2)>
    # <TIME ALIGN WORKAROUND>
    import datetime

    # CLOCK_ZERO = UTCDateTime(2004, 9, 28, 0, 0, 0.021467)
    # CLOCK_ZERO = dataset_config.starttime
    for stream in streams:
        stream.stats["channel"] = FDSN_CHANNEL_MAP[stream.stats["channel"]]
        # station_channel = f"STA:{stream.stats['station']} CH
        # {stream.stats['channel']}"
        print(
            f"{stream.stats['station']}  {stream.stats['channel']} N="
            f"{len(stream.data)}  startime {stream.stats.starttime}"
        )
        dt_seconds = stream.stats.starttime - dataset_config.starttime
        print(f"dt_seconds {dt_seconds}")
        dt = datetime.timedelta(seconds=dt_seconds)
        print(f"dt = {dt}")
        stream.stats.starttime = stream.stats.starttime - dt

        # stream.stats.endtime =- dt
    # </TIME ALIGN WORKAROUND>
    # </REASSIGN NON-CONVENTIONAL CHANNEL LABELS (Q2, Q3, T1, T2)>

    # BREAK STREAMS UP BY STATION?

    # <This block is called often - should be a method>
    start_times = sorted(list(set([tr.stats.starttime.isoformat() for tr in streams])))
    end_times = sorted(list(set([tr.stats.endtime.isoformat() for tr in streams])))
    run_stream = streams.slice(UTCDateTime(start_times[0]), UTCDateTime(end_times[-1]))
    import obspy

    streams_dict = {}
    streams_dict["PKD"] = obspy.core.Stream(run_stream.traces[:4])
    streams_dict["SAO"] = obspy.core.Stream(run_stream.traces[4:])
    # BREAK STREAMS UP BY STATION?
    # </This block is called often - should be a method>
    station_groups = {}
    station_groups["PKD"] = mth5_obj.get_station("PKD")
    station_groups["SAO"] = mth5_obj.get_station("SAO")
    run_id = "001"
    for i_station, station in enumerate(mth5_obj.station_list):
        run_metadata = experiment.surveys[0].stations[i_station].runs[0]
        run_ts_obj = RunTS()
        run_ts_obj.from_obspy_stream(streams_dict[station], run_metadata)
        run_ts_obj.run_metadata.id = run_id
        run_group = station_groups[station].add_run(run_id)
        run_group.from_runts(run_ts_obj)
    mth5_obj.close_mth5()

    return


def test_make_parkfield_mth5():
    dataset_id = "pkd_test_00"
    create_from_server(dataset_id, data_source="IRIS")
    h5_path = DATA_PATH.joinpath(f"{dataset_id}.h5")
    read_back_data(h5_path, "PKD", "001")


def test_make_hollister_mth5():
    dataset_id = "sao_test_00"
    create_from_server(dataset_id, data_source="NCEDC")
    h5_path = DATA_PATH.joinpath(f"{dataset_id}.h5")
    run, runts = read_back_data(h5_path, "SAO", "001")
    print("hello")


def test_make_parkfield_hollister_mth5():
    dataset_id = "pkd_sao_test_00"
    create_from_server_multistation(dataset_id, data_source="NCEDC")
    h5_path = DATA_PATH.joinpath(f"{dataset_id}.h5")
    pkd_run_obj, pkd_run_ts = read_back_data(h5_path, "PKD", "001")
    sao_run_obj, sao_run_ts = read_back_data(h5_path, "SAO", "001")
    print(pkd_run_obj)
    print(sao_run_obj)
    print("OK")


# <TEST FAILS BECAUSE DATA NOT AVAILABLE FROM IRIS -- NEED TO REQUEST FROM NCEDC>
# def test_make_hollister_mth5():
#     dataset_id = "sao_test_00"
#     create_from_iris(dataset_id)
#     h5_path = DATA_PATH.joinpath(f"{dataset_id}.h5")
#     read_back_data(h5_path, "SAO", "001")
# </TEST FAILS BECAUSE DATA NOT AVAILABLE FROM IRIS -- NEED TO REQUEST FROM NCEDC>


def main():
    test_make_parkfield_mth5()
    # test_make_hollister_mth5()
    test_make_parkfield_hollister_mth5()


#    test_make_parkfield_mth5()
# test_make_hollister_mth5()


if __name__ == "__main__":
    main()
