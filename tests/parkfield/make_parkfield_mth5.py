# Relates to github aurora issue #17
# Ingest the Parkfield data and use mth5 as the interface for this task


from aurora.general_helper_functions import DATA_PATH
from aurora.sandbox.io_helpers.make_dataset_configs import TEST_DATA_SET_CONFIGS
from aurora.sandbox.io_helpers.test_data import get_example_array_list
from aurora.sandbox.io_helpers.test_data import HEXY
from aurora.sandbox.mth5_helpers import cast_run_to_run_ts
from aurora.sandbox.mth5_helpers import check_run_channels_have_expected_properties
from aurora.sandbox.mth5_helpers import mth5_from_iris_database
#from aurora.sandbox.mth5_helpers import test_runts_from_xml




def test_create(dataset_id):
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    mth5_obj = mth5_from_iris_database(dataset_config)
    #filters are accessed like this:
    #mth5_obj.filters_group.filter_dict
    run_count = 1
    for station_id in mth5_obj.station_list:
        print(station_id)
        run_id = str(run_count).zfill(3)
        print(f"run_id {run_id}")
        run_obj = mth5_obj.get_run(station_id, str(run_count).zfill(3))
        print("OK you have created the run")
        print("next step it to check the run is ok")

        print("once the data are there, then you can need to save the h5")
        print("follow the example in the synthetic data maker to do this")
        #print(experiment.surveys[0].stations[0].runs[0])
        check_run_channels_have_expected_properties(run_obj)

        array_list = get_example_array_list(components_list=HEXY,
                                        load_actual=True,
                                        station_id=station_id)
        #HERE IS WHERE WE WILL NEED TO ADD FILTERS
        runts_obj = cast_run_to_run_ts(run_obj, array_list=array_list)

        station_group = mth5_obj.add_station(station_id)
        run_group = station_group.add_run(run_id)
        run_group.from_runts(runts_obj)
        # # add filters
        # for fltr in ACTIVE_FILTERS:
        #     cf_group = m.filters_group.add_filter(fltr)
        # make an MTH5
    mth5_obj.close_mth5()

    pass

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
    test_create(dataset_id)
    test_can_read_back_data()

if __name__ == '__main__':
    main()
