"""
20210520: This is a copy of aurora_driver.py which is going to be overwritten.
Most of the tests and tools are associated with MTH5 helper stuffs so moved
to mth5_helpers.py for now.  Needs a clean up.

2021-09-17: I have started commenting out blocks of code here that are not used anymore.
This future of this module is not certain.  Most of these tests will be moved to
mt_metadata and mth5, however, there is one application I would like to support here:
Karl would like to create tools in aurora to work and develop offline.  This means
1. Pulling station_xml and storing locally as xml and then reloading on demand to
create inventory, --> experiement, --> mth5
2. Using rover to access datasets into a local archive and then populating mth5 from
the data.  **is this a solution for NCEDC?? Does ROVER work with NCEDC??

"""
import numpy as np
from pathlib import Path

from aurora.test_utils.dataset_definitions import TEST_DATA_SET_CONFIGS

from aurora.sandbox.io_helpers.testing_data import get_example_data
from aurora.sandbox.xml_sandbox import describe_inventory_stages
from mth5.utils.helpers import initialize_mth5
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment

HEXY = ["hx", "hy", "ex", "ey"]  # default components list
xml_path = Path("/home/kkappler/software/irismt/mt_metadata/data/xml")
magnetic_xml_template = xml_path.joinpath("mtml_magnetometer_example.xml")
electric_xml_template = xml_path.joinpath("mtml_electrode_example.xml")
fap_xml_example = ""


def get_experiment_from_obspy_inventory(inventory):
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)
    return experiment


def check_run_channels_have_expected_properties(run):
    """
    Just some sanity check that we can access filters
    Parameters
    ----------
    run

    Returns
    -------

    """
    print(run.channel_summary)
    hx = run.get_channel("hx")
    print(hx.channel_response_filter)
    print(hx.channel_response_filter.filters_list)
    print(hx.channel_response_filter.complex_response(np.arange(3) + 1))
    ex = run.get_channel("ex")
    print(ex.channel_response_filter)
    print(ex.channel_response_filter.filters_list)
    print(ex.channel_response_filter.complex_response(np.arange(3) + 1))
    return


def run_obj_from_mth5(mth5_obj):
    """
    one off method showing how we create runs in an mth5 object

    Parameters
    ----------
    mth5_obj

    Returns
    -------

    """
    if "REW09" in mth5_obj.station_list:  # old test
        run_obj = mth5_obj.get_run("REW09", "a")
    elif "PKD" in mth5_obj.station_list:  # pkd test
        run_obj = mth5_obj.get_run("PKD", "001")  # this run is created here
        check_run_channels_have_expected_properties(run_obj)
    else:
        print("skipping creation of run ")
        raise Exception

    return run_obj


def mth5_run_from_experiment(station_id, experiment, h5_path=None):
    """

    Parameters
    ----------
    station_id
    experiment
    h5_path

    Returns
    -------

    """
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)
    run_obj = run_obj_from_mth5(mth5_obj)
    return run_obj


def cast_run_to_run_ts(run, array_list=None, station_id=None):
    """
    basically embed data into a run_ts object.
    array_list = get_example_array_list(components_list=HEXY,
                                        load_actual=True,
                                        station_id="PKD")
    Parameters
    ----------
    run
    array_list
    station_id

    Returns
    -------

    """
    runts_object = run.to_runts()
    if array_list:
        runts_object.set_dataset(array_list)
    if station_id:
        runts_object.station_metadata.id = station_id
    return runts_object


# </LOAD SOME DATA FROM A SINGLE STATION>


def set_driver_parameters():
    driver_parameters = {}
    driver_parameters["create_xml"] = True
    #    driver_parameters["test_filter_control"] = False  # failing because cannot
    # find/read xml
    driver_parameters["run_ts_from_xml_01"] = True
    driver_parameters["run_ts_from_xml_02"] = False  # Fail: no access to pkd.xml
    driver_parameters["initialize_data"] = True
    return driver_parameters


def test_can_access_fap_filters():
    """
    Need a test in mt_metadata that does this
    Returns
    -------

    """

    from aurora.sandbox.io_helpers.fdsn_dataset_config import FDSNDatasetConfig

    test_data_set = FDSNDatasetConfig()
    test_data_set.dataset_id = "fap_test"
    test_data_set.network = "EM"
    test_data_set.station = "FL001"
    test_data_set.starttime = None
    test_data_set.endtime = None
    test_data_set.channel_codes = "MFN"
    test_data_set.description = "test of a fap xml"

    fap_inventory = test_data_set.get_inventory_from_client(
        ensure_inventory_stages_are_named=True
    )
    describe_inventory_stages(fap_inventory)
    # <HERE IS THE SPOT TO DROP TRACE TO REVIEW INGEST OF FAP>
    experiment = get_experiment_from_obspy_inventory(fap_inventory)
    filters_dict = experiment.surveys[0].filters
    print(filters_dict)
    fap = filters_dict["mfn_0"]
    print(fap)
    num_filters = len(experiment.surveys[0].filters)
    print(f"num_filters {num_filters}")
    return


def main():
    # test_can_access_fap_filters()

    driver_parameters = set_driver_parameters()
    # <CREATE METADATA XML>
    if driver_parameters["create_xml"]:
        dataset_ids = [
            "pkd_test_00",
            "sao_test_00",
        ]
        for dataset_id in dataset_ids:
            test_dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
            inventory = test_dataset_config.get_inventory_from_client(
                ensure_inventory_stages_are_named=False
            )
            experiment = get_experiment_from_obspy_inventory(inventory)
            print(experiment)

    # </CREATE METADATA XML>


    # <TEST RunTS FROM XML>
    if driver_parameters["run_ts_from_xml_02"]:
        dataset_id = "pkd_test_00"
        test_dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
        pkd_xml = test_dataset_config.get_station_xml_filename()
        experiment = get_experiment_from_xml_path(pkd_xml)
        run_obj = mth5_run_from_experiment("PKD", experiment)
        runts_obj = cast_run_to_run_ts(run_obj, station_id="PKD")
        print(runts_obj)
    # </TEST RunTS FROM XML>

    # <INITIALIZE DATA>
    if driver_parameters["initialize_data"]:
        try:
            pkd_mvts = get_example_data(station_id="PKD", component_station_label=True)
            sao_mvts = get_example_data(station_id="SAO", component_station_label=True)
            pkd = pkd_mvts.dataset
            sao = sao_mvts.dataset
            pkd.update(sao)
        except FileNotFoundError:
            print("failing to find: "
                  "aurora/tests/parkfield/data/pkd_sao_2004_272_00-02.h5 ")
            print("I put a copy of that data here:")
            print("https://drive.google.com/drive/folders/1WFhhLrt5wSlw4FaAFkfb58allNdGjwck")

    # </INITIALIZE DATA>
    print("try to combine these runs")


if __name__ == "__main__":
    main()
    print("Fin")
