"""
Relates to github aurora issue #17
Ingest the Parkfield data make mth5 to use as the interface for tests

2021-09-17: Modifying create methods to use FDSNDatasetConfig as input rather than
dataset_id
"""
from aurora.sandbox.io_helpers.make_dataset_configs import TEST_DATA_SET_CONFIGS
from aurora.sandbox.io_helpers.make_mth5_helpers import create_from_server_multistation
from aurora.pipelines.helpers import read_back_data

from helpers import DATA_PATH

# create_from_server DEPRECATED 2021-09-18
# def create_from_server(dataset_config, data_source="IRIS", target_folder=Path()):
#
#     # <GET EXPERIMENT>
#     inventory = dataset_config.get_inventory_from_client(
#         base_url=data_source, ensure_inventory_stages_are_named=True
#     )
#     translator = XMLInventoryMTExperiment()
#     experiment = translator.xml_to_mt(inventory_object=inventory)
#     # </GET EXPERIMENT>
#
#     # <TRIAGE ONE-OFF ISSUE WITH UNITS>
#     experiment = triage_mt_units_electric_field(experiment)
#     # </TRIAGE ONE-OFF ISSUE WITH UNITS>
#
#     # <INITIALIZE MTH5>
#     h5_path = target_folder.joinpath(dataset_config.h5_filebase)
#     mth5_obj = initialize_mth5(h5_path)
#     mth5_obj.from_experiment(experiment)
#     # </INITIALIZE MTH5>
#     dataset_config.describe()
#
#     # <GET DATA STREAMS>
#     streams = dataset_config.get_data_via_fdsn_client(data_source=data_source)
#     streams = make_channel_labels_fdsn_compliant(streams)
#     run_stream = trim_streams_to_acquisition_run(streams)
#     # </GET DATA STREAMS>
#
#     # <THIS NEEDS TO BE WRAPPED IN MULTI STATION /MULTI RUN LOGIC>
#     station_group = mth5_obj.get_station(dataset_config.station)
#     run_metadata = experiment.surveys[0].stations[0].runs[0]
#     run_ts_obj = RunTS()
#     run_ts_obj.from_obspy_stream(run_stream, run_metadata)
#     run_id = "001"
#     run_ts_obj.run_metadata.id = run_id
#     run_group = station_group.add_run(run_id)
#     run_group.from_runts(run_ts_obj)
#     # </THIS NEEDS TO BE WRAPPED IN MULTI STATION /MULTI RUN LOGIC>
#     mth5_obj.close_mth5()
#
#     return


def test_make_parkfield_mth5():
    dataset_id = "pkd_test_00"
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    create_from_server_multistation(
        dataset_config,
        data_source="IRIS",
        target_folder=DATA_PATH,
        triage_units="V/m to mV/km",
    )
    h5_path = DATA_PATH.joinpath(dataset_config.h5_filebase)
    read_back_data(h5_path, "PKD", "001")


def test_make_parkfield_hollister_mth5():
    dataset_id = "pkd_sao_test_00"
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    create_from_server_multistation(
        dataset_config,
        data_source="NCEDC",
        target_folder=DATA_PATH,
        triage_units="V/m to mV/km",
    )
    h5_path = DATA_PATH.joinpath(dataset_config.h5_filebase)
    pkd_run_obj, pkd_run_ts = read_back_data(h5_path, "PKD", "001")
    sao_run_obj, sao_run_ts = read_back_data(h5_path, "SAO", "001")
    print(pkd_run_obj)
    print(sao_run_obj)
    print("OK")


# def test_make_hollister_mth5():
#     dataset_id = "sao_test_00"
#     create_from_server(dataset_id, data_source="NCEDC")
#     h5_path = DATA_PATH.joinpath(f"{dataset_id}.h5")
#     run, runts = read_back_data(h5_path, "SAO", "001")
#     print("hello")
#
#

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


if __name__ == "__main__":
    main()
