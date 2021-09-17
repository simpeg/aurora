"""

20210520: This is a copy of aurora_driver.py which is going to be overwritten.
Most of the tests and tools are associated with MTH5 helper stuffs so moved
to mth5_helpers.py for now.  Needs a clean up.
"""
import datetime
import numpy as np
from pathlib import Path

from aurora.pipelines.helpers import initialize_mth5
from aurora.sandbox.io_helpers.make_dataset_configs import TEST_DATA_SET_CONFIGS
from aurora.sandbox.io_helpers.test_data import get_example_array_list
from aurora.sandbox.io_helpers.test_data import get_example_data
from aurora.sandbox.xml_sandbox import describe_inventory_stages
from mt_metadata.timeseries import Experiment
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment

HEXY = ["hx", "hy", "ex", "ey"]  # default components list
xml_path = Path("/home/kkappler/software/irismt/mt_metadata/data/xml")
magnetic_xml_template = xml_path.joinpath("mtml_magnetometer_example.xml")
electric_xml_template = xml_path.joinpath("mtml_electrode_example.xml")
fap_xml_example = ""


def align_streams(streams, clock_start):
    """
    This is a hack around to handle data that are asynchronously sampled.
    It should not be used in general.  It is only appropriate for datasets that have
    been tested with it.
    PKD, SAO only at this point.

    Parameters
    ----------
    streams : iterable of types obspy.core.stream.Stream
    clock_start : obspy UTCDateTime
        this is a reference time that we set the first sample to be

    Returns
    -------

    """
    for stream in streams:
        print(
            f"{stream.stats['station']}  {stream.stats['channel']} N="
            f"{len(stream.data)}  startime {stream.stats.starttime}"
        )
        dt_seconds = stream.stats.starttime - clock_start
        print(f"dt_seconds {dt_seconds}")
        dt = datetime.timedelta(seconds=dt_seconds)
        print(f"dt = {dt}")
        stream.stats.starttime = stream.stats.starttime - dt
    return streams


def mth5_from_iris_database(dataset_config, load_data=True, target_folder=Path()):
    """
    This can work in a way that uses data, or just initializes the mth5

    Parameters
    ----------
    metadata_config:

    Returns
    -------

    """
    inventory = dataset_config.get_inventory_from_iris(
        ensure_inventory_stages_are_named=True
    )
    experiment = get_experiment_from_obspy_inventory(inventory)

    # make an MTH5
    h5_path = target_folder.joinpath(f"{dataset_config.dataset_id}.h5")
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)

    return mth5_obj


def test_runts_from_xml(dataset_id, runts_obj=False):
    """
    THIS METHOD SHOULD BE DEPRECATED ONCE THE PARKFIELD EXAMPLE TEST IS RUNNING
    This means after github issues #33 and #34 are closed.
    This function is an example of mth5 creation.  It is a separate topic from
    aurora pipeline.  This is an Element#1 aspect of the proposal.

    We base this on a dataset_id but really it needs a dataset config,
    so probably taking a config as an input would be more modular.

    The flow here is to get the inventory object from the iris database using
    :param dataset_id:
    :param runts_obj:
    :return:
    """
    dataset_id = "pkd_test_00"
    #
    test_dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    inventory = test_dataset_config.get_inventory_from_iris(
        ensure_inventory_stages_are_named=True
    )
    experiment = get_experiment_from_obspy_inventory(inventory)

    experiment.surveys[0].filters["fir_fs2d5_2000.0"]
    experiment.surveys[0].filters["fir_fs2d5_200.0"].decimation_input_sample_rate
    h5_path = Path("PKD.h5")
    run_obj = mth5_run_from_experiment("PKD", experiment, h5_path=h5_path)

    if runts_obj:
        array_list = get_example_array_list(
            components_list=HEXY, load_actual=True, station_id="PKD"
        )
        runts_obj = cast_run_to_run_ts(run_obj, array_list=array_list)
    return experiment, run_obj, runts_obj


# <GET EXPERIMENT>
def get_experiment_from_xml_path(xml):
    xml_path = Path(xml)
    experiment = Experiment()
    experiment.from_xml(fn=xml_path)
    print(experiment, type(experiment))
    return experiment


def get_experiment_from_obspy_inventory(inventory):
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)
    return experiment


# </GET EXPERIMENT>


def get_filters_dict_from_experiment(experiment, verbose=False):
    """
    MTH5 HELPER
    Only takes the zero'th survey, we will need to index surveys eventually
    Parameters
    ----------
    experiment
    verbose

    Returns
    -------

    """
    surveys = experiment.surveys
    survey = surveys[0]
    survey_filters = survey.filters
    if verbose:
        print(experiment, type(experiment))
        print("Survey Filters", survey.filters)
        filter_keys = list(survey_filters.keys())
        print("FIlter keys", filter_keys)
        for filter_key in filter_keys:
            print(filter_key, survey_filters[filter_key])
    return survey_filters


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


def filter_control_example(xml_path=None):
    """
    This has two stages:
    1. reads an xml
    2. casts to experiement
    3. does filter tests.
    The filter tests all belong in MTH5 Helpers.
    Loads an xml file and casts it to experiment.  Iterates over the filter objects to
    confirm that these all registered properly and are accessible.  Evaluates
    each filter at a few frequencies to confirm response function works

    ToDo: Access "single_station_mt.xml" from metadata repository
    Parameters
    ----------
    xml_path

    Returns
    -------

    """
    if xml_path is None:
        from aurora.general_helper_functions import MT_METADATA_DATA

        xml_path = MT_METADATA_DATA.joinpath("stationxml", "mtml_single_station.xml")
    experiment = get_experiment_from_xml_path(xml_path)
    filter_dict = get_filters_dict_from_experiment(experiment)
    frq = np.arange(5) + 1.2
    filter_keys = list(filter_dict.keys())
    for key in filter_keys:
        my_filter = filter_dict[key]
        response = my_filter.complex_response(frq)
        print(f"{key} response", response)

    for key in filter_dict.keys():
        print(f"key = {key}")
    print("OK")


def test_filter_control():
    print("move this from driver")
    pass


# </LOAD SOME DATA FROM A SINGLE STATION>


def set_driver_parameters():
    driver_parameters = {}
    driver_parameters["create_xml"] = True
    driver_parameters["test_filter_control"] = False  # failing because cannot
    # find/read xml
    driver_parameters["run_ts_from_xml_01"] = True
    driver_parameters["run_ts_from_xml_02"] = False  # Fail: no access to pkd.xml
    driver_parameters["initialize_data"] = True
    return driver_parameters


def test_can_access_fap_filters():
    from aurora.sandbox.io_helpers.fdsn_dataset_config import FDSNDatasetConfig

    test_data_set = FDSNDatasetConfig()
    test_data_set.dataset_id = "fap_test"
    test_data_set.network = "EM"
    test_data_set.station = "FL001"
    test_data_set.starttime = None
    test_data_set.endtime = None
    test_data_set.channel_codes = "MFN"
    test_data_set.description = "test of a fap xml"

    fap_inventory = test_data_set.get_inventory_from_iris(
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
            inventory = test_dataset_config.get_inventory_from_iris(
                ensure_inventory_stages_are_named=False
            )
            experiment = get_experiment_from_obspy_inventory(inventory)
            print(experiment)

    # </CREATE METADATA XML>

    # <TEST FILTER CONTROL>
    if driver_parameters["test_filter_control"]:
        filter_control_example()  # KeyError: 'survey' -20210828
        dataset_id = "pkd_test_00"
        test_dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
        xml_path = test_dataset_config.get_station_xml_filename()
        filter_control_example(xml_path=xml_path)
    # </TEST FILTER CONTROL>

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
        pkd_mvts = get_example_data(station_id="PKD", component_station_label=True)
        sao_mvts = get_example_data(station_id="SAO", component_station_label=True)
        pkd = pkd_mvts.dataset
        sao = sao_mvts.dataset
        pkd.update(sao)
    # </INITIALIZE DATA>
    print("try to combine these runs")


if __name__ == "__main__":
    main()
    print("Fin")
