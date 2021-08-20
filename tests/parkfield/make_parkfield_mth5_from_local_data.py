from pathlib import Path


from obspy import UTCDateTime
from obspy.core import Stream
from obspy.core import Trace

from aurora.sandbox.io_helpers.make_dataset_configs import TEST_DATA_SET_CONFIGS
from aurora.sandbox.mth5_helpers import get_experiment_from_obspy_inventory
from aurora.sandbox.mth5_helpers import initialize_mth5
from aurora.sandbox.mth5_helpers import test_can_read_back_data
from mth5.timeseries import RunTS

def create_from_local_data(dataset_id):
    """
    TODO:
    This test could be made to depend on mth5_test_data and the imports from
    mth5_test_data can be placed within this function.
    In fact, this function could be migrated to mth5_test_data entirely,
    removing that dependency from aurora but leaving the code available for
    development and to act as an example.
    Parameters
    ----------
    dataset_id

    Returns
    -------

    """
    from aurora.sandbox.io_helpers.test_data import get_example_array_list

    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    inventory = dataset_config.get_inventory_from_iris(ensure_inventory_stages_are_named=True)
    experiment = get_experiment_from_obspy_inventory(inventory)
    run_metadata = experiment.surveys[0].stations[0].runs[0]
    station_id = "PKD"
    start_time = UTCDateTime("2004-09-28T00:00:00.000000Z")
    end_time = UTCDateTime("2004-09-28T01:59:59.975000Z")
    target_folder = Path()
    h5_path = target_folder.joinpath(f"ascii_{dataset_config.dataset_id}.h5")
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)
    station_group = mth5_obj.get_station(station_id)
    #</20210713>
    components_list = ["hx", "hy", "ex", "ey"]
    array_list = get_example_array_list(components_list=components_list,
                                        load_actual=True,
                                        station_id=station_id)
    header = {}
    header['sampling_rate'] = 40.0;
    header['starttime'] = start_time
    header['endtime'] = end_time
    header["station"] = station_id
    header["network"] = "BK"
    #unused headers:['calib','location']

    header["channel"] = "BF1"
    hx = Trace(data=array_list[0].ts, header=header)
    header["channel"] = "BF2"
    hy = Trace(data=array_list[1].ts, header=header)
    header["channel"] = "BQ1"
    ex = Trace(data=array_list[2].ts, header=header)
    header["channel"] = "BQ2"
    ey = Trace(data=array_list[3].ts, header=header)
    run_stream = Stream(traces=[hx, hy, ex, ey])

    run_id = "001"
    run_ts_obj = RunTS()
    run_ts_obj.from_obspy_stream(run_stream, run_metadata)
    run_ts_obj.run_metadata.id = run_id
    run_group = station_group.add_run(run_id)
    run_group.from_runts(run_ts_obj)

    mth5_obj.close_mth5()



def main():
    dataset_id = "pkd_test_00"
    create_from_local_data(dataset_id)
    test_can_read_back_data("pkd_test_00.h5", "PKD", "001")

if __name__ == '__main__':
    main()
