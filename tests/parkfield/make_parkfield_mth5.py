# Relates to github aurora issue #17
# Ingest the Parkfield data and use mth5 as the interface for this task
from pathlib import Path

import copy

from mth5.timeseries import RunTS

from aurora.general_helper_functions import DATA_PATH
from aurora.sandbox.io_helpers.inventory_review import scan_inventory_for_nonconformity
from aurora.sandbox.io_helpers.make_dataset_configs import TEST_DATA_SET_CONFIGS
from aurora.sandbox.io_helpers.test_data import HEXY
from aurora.sandbox.mth5_helpers import cast_run_to_run_ts
from aurora.sandbox.mth5_helpers import check_run_channels_have_expected_properties
from aurora.sandbox.mth5_helpers import get_experiment_from_obspy_inventory
from aurora.sandbox.mth5_helpers import initialize_mth5
from aurora.sandbox.mth5_helpers import mth5_from_iris_database
#from aurora.sandbox.mth5_helpers import test_runts_from_xml




def create_from_iris(dataset_id):
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    inventory = dataset_config.get_inventory_from_iris(ensure_inventory_stages_are_named=True)
    experiment = get_experiment_from_obspy_inventory(inventory)
    run_metadata = experiment.surveys[0].stations[0].runs[0]
    #experiment.surveys[0].stations[0].runs[0].update_time_period()
    target_folder = Path()
    h5_path = target_folder.joinpath(f"{dataset_config.dataset_id}.h5")
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)

    #Make a dummy run_ts obj?
    #probably easiest to simply cast the data as an obspy stream
    import datetime
    from obspy import UTCDateTime
    from obspy.clients import fdsn
    from obspy.core import Stream
    from obspy.core import Trace
    start_time = UTCDateTime("2004-09-28T00:00:00.000000Z")
    end_time = UTCDateTime("2004-09-28T01:59:59.975000Z")
    station_id = "PKD" #station_id in mth5_obj.station_list
    station_group = mth5_obj.get_station(station_id)

    client = fdsn.Client("IRIS")
    streams = client.get_waveforms("BK", station_id, None, "BT1,BT2,BQ2,BQ3",
                                   start_time,
                                   end_time)
    #<FIX NON-CONVENTIONAL CHANNEL LABELS>
    streams[0].stats["channel"] = "BQ1"
    streams[1].stats["channel"] = "BQ2"
    streams[2].stats["channel"] = "BF1"
    streams[3].stats["channel"] = "BF2"
    #</FIX NON-CONVENTIONAL CHANNEL LABELS>
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


def create_from_local_data():
    from aurora.sandbox.io_helpers.test_data import get_example_array_list
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    inventory = dataset_config.get_inventory_from_iris(ensure_inventory_stages_are_named=True)
    #</20210713>
    array_list = get_example_array_list(components_list=HEXY,
                                        load_actual=True,
                                        station_id=station_id)
    header = {}
    header['sampling_rate'] = 40.0;
    header['starttime'] = start_time
    header['endtime'] = end_time
    header["station"] = station_id
    header["network"] = "BK"
    #unused headers:
    #'calib','location', 'channel']
    #hx_header = copy.deepcopy(header)
    header["channel"] = "BF1"
    hx = Trace(data=array_list[0].ts, header=header)
    header["channel"] = "BF2"
    hy = Trace(data=array_list[1].ts, header=header)
    header["channel"] = "BQ1"
    ex = Trace(data=array_list[2].ts, header=header)
    header["channel"] = "BQ2"
    ey = Trace(data=array_list[3].ts, header=header)
    run_stream = Stream(traces=[hx, hy, ex, ey])
    #
    run_id = "001"
    run_ts_obj = RunTS()
    run_ts_obj.from_obspy_stream(run_stream, run_metadata)
    #PROBLEM THE RUN_TS is 2h long but the run is unde
    run_ts_obj.run_metadata.id = run_id
    run_group = station_group.add_run(run_id)
    run_group.from_runts(run_ts_obj)

    mth5_obj.close_mth5()
    # # #
    # # #
    # # #<COPIED FROM IRIS DMC EXAMPLE>
    # # station_group = mth5_obj.get_station(station)
    # #
    # # # runs can be split into channels with similar start times and sample rates
    # # start_times = sorted(
    # #     list(set([tr.stats.starttime.isoformat() for tr in streams])))
    # # end_times = sorted(
    # #     list(set([tr.stats.endtime.isoformat() for tr in streams])))
    # #
    # # for index, times in enumerate(zip(start_times, end_times), 1):
    # #     run_id = f"{index:03}"
    # #     run_stream = streams.slice(UTCDateTime(times[0]), UTCDateTime(times[1]))
    # #     run_ts_obj = RunTS()
    # #     # need to add run metadata because in the stationxml the channel metadata
    # #     # is only one entry for all similar channels regardless of their duration
    # #     # so we need to make sure that propagates to the MTH5.
    # #     run_ts_obj.from_obspy_stream(run_stream, run_metadata)
    # #     run_ts_obj.run_metadata.id = run_id
    # #     run_group = station_group.add_run(run_id)
    # #     run_group.from_runts(run_ts_obj)
    # #
    # # if to_stationxml:
    # #     new_inv = translator.mt_to_xml(m.to_experiment(),
    # #                                    stationxml_fn=local_path.joinpath(
    # #                                        f"{station}_from_mth5.xml"))
    # #
    # # if not interact:
    # #     m.close_mth5()
    # # #</COPIED FROM IRIS DMC EXAMPLE>
    # # #
    # # #
    # # #mth5_obj = mth5_from_iris_database(dataset_config)
    # # #filters are accessed like this:
    # # #mth5_obj.filters_group.filter_dict
    # # run_count = 1
    # # for station_id in mth5_obj.station_list:
    # #     print(station_id)
    # #     run_id = str(run_count).zfill(3)
    # #     print(f"run_id {run_id}")
    # #     run_obj = mth5_obj.get_run(station_id, str(run_count).zfill(3))
    # #     print("OK you have created the run")
    # #     print("next step it to check the run is ok")
    # #
    # #     print("once the data are there, then you can need to save the h5")
    # #     print("follow the example in the synthetic data maker to do this")
    # #     #print(experiment.surveys[0].stations[0].runs[0])
    # #     check_run_channels_have_expected_properties(run_obj)
    # #
    # #     array_list = get_example_array_list(components_list=HEXY,
    # #                                     load_actual=True,
    # #                                     station_id=station_id)
    # #     #HERE IS WHERE WE WILL NEED TO ADD FILTERS
    # #     #run_obj has filters right now
    # #
    # #     runts_obj = cast_run_to_run_ts(run_obj, array_list=array_list)
    # #     # run_obj still has filters
    # #     station_group = mth5_obj.add_station(station_id)
    # #     run_group = station_group.add_run(run_id)
    # #     # run_group still has filters
    # #     #this action here is removing the filters ... we need to add the data
    # #     # to the
    # #     run_group.from_runts(runts_obj)
    # #     # # add filters
    # #     # for fltr in ACTIVE_FILTERS:
    # #     #     cf_group = m.filters_group.add_filter(fltr)
    # #     # make an MTH5
    # # mth5_obj.close_mth5()
    #
    # pass

def test_can_read_back_data():
    from mth5.mth5 import MTH5
    run_id = "001"
    processing_config = {}
    processing_config["mth5_path"] = "pkd_test_00.h5"
    processing_config["local_station_id"] = "PKD"
    config = processing_config
    m = MTH5()
    m.open_mth5(config["mth5_path"], mode="r")
    local_run_obj = m.get_run(config["local_station_id"], run_id)
    local_run_ts = local_run_obj.to_runts()
    print("hi")

def main():
    dataset_id = "pkd_test_00"
    create_from_iris(dataset_id)
    create_from_local_data(dataset_id)
    test_can_read_back_data()

if __name__ == '__main__':
    main()
