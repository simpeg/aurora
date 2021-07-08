"""
20210511: This script is intended to run an example version of end-to-end processing.
        #import xml.etree.ElementTree as ET
        #tree = ET.parse(xml_path)
        # mt_root_element = tree.getroot()
        # mt_experiment = Experiment()
        # mt_experiment.from_xml(mt_root_element)


TODO: MTH5 updated so that channel now returns a channel response
The question is how to propagate the response information to Attributes RunTS

20210520: This is a copy of aurora_driver.py which is going to be overwritten.  Most of the tests and tools
are associated with MTH5 helper stuffs so moved to mth5_helpers.py for now.  Needs a clean up.
"""

import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr

from aurora.sandbox.io_helpers.test_data import get_example_array_list
from aurora.sandbox.io_helpers.test_data import get_example_data
from aurora.sandbox.io_helpers.test_data import TEST_DATA_SET_CONFIGS
from aurora.sandbox.xml_sandbox import describe_inventory_stages
from mt_metadata.timeseries import Experiment
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mt_metadata.utils import STATIONXML_02
from mth5.mth5 import MTH5
from mth5.timeseries.channel_ts import ChannelTS
from mth5.timeseries.run_ts import RunTS

HEXY = ['hx','hy','ex','ey'] #default components list
xml_path = Path("/home/kkappler/software/irismt/mt_metadata/data/xml")
magnetic_xml_template = xml_path.joinpath("mtml_magnetometer_example.xml")
electric_xml_template = xml_path.joinpath("mtml_electrode_example.xml")
single_station_xml_template = STATIONXML_02 # Fails for "no survey key"
fap_xml_example = ""

#single_station_xml_template = Path("single_station_mt.xml")
def test_runts_from_xml(dataset_id, runts_obj=False):
    """
    This function is an example of mth5 creation.  It is a separate topic from
    aurora pipeline.  This is an Element#1 aspect of the proposal.
    :param dataset_id:
    :param runts_obj:
    :return:
    """
    dataset_id = "pkd_test_00"
    #
    test_dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    inventory = test_dataset_config.get_inventory_from_iris(ensure_inventory_stages_are_named=True)
    experiment = get_experiment_from_obspy_inventory(inventory)

    experiment.surveys[0].filters["fir_fs2d5_2000.0"]
    experiment.surveys[0].filters[
        "fir_fs2d5_200.0"].decimation_input_sample_rate
    test_dataset_config.save_xml(experiment)
    h5_path = Path("PKD.h5")
    run_obj = embed_experiment_into_run("PKD", experiment, h5_path=h5_path)

    if runts_obj:
        array_list = get_example_array_list(components_list=HEXY,
                                            load_actual=True,
                                            station_id="PKD")
        runts_obj = cast_run_to_run_ts(run_obj, array_list=array_list)
    return experiment, run_obj, runts_obj

#<GET EXPERIMENT>
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



def get_inventory_from_test_data_config(dataset_id):
    """

    Parameters
    ----------
    dataset_id: dataset_id = "pkd_test_00"

    Returns
    -------

    """
    from iris_mt_scratch.sandbox.io_helpers.test_data import TEST_DATA_SET_CONFIGS
    test_dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    inventory = test_dataset_config.get_inventory_from_iris(ensure_inventory_stages_are_named=True)
    return inventory




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
    hx = run.get_channel('hx')
    print(hx.channel_response_filter)
    print(hx.channel_response_filter.filters_list)
    print(hx.channel_response_filter.complex_response(np.arange(3) + 1))
    ex = run.get_channel('ex')
    print(ex.channel_response_filter)
    print(ex.channel_response_filter.filters_list)
    print(ex.channel_response_filter.complex_response(np.arange(3) + 1))
    return

def test_experiment_from_station_xml():
    """
    This test passes but when we use the hack of setting magnetic to "T" instead of "F" in
    fdsn_tools.py it fails for no code "F"
    Returns
    -------

    """
    from mt_metadata.utils import STATIONXML_02
    single_station_xml_template = STATIONXML_02  # Fails for "no survey key"
    #single_station_xml_template = Path("single_station_mt.xml")
    translator = XMLInventoryMTExperiment()
    mt_experiment = translator.xml_to_mt(stationxml_fn=STATIONXML_02)
    return


def embed_experiment_into_run(station_id, experiment, h5_path=None):
    """
    2021-05-12: Trying to initialize RunTS class from xml metadata.

    THis function served two purposes
    1. it was a proving ground for fiddling around with runs and xmls.
    2. It is specifically used by the driver for loading runs from PKD
    or SAO.

    It should therefore be factored into some general run stuffs and

    This will give us a single station run for now


    Tried several ways to manually assign run properties
    Here are some sample commands I may need this week.
    #ch = get_channel("hx", station_id="PKD", load_actual=True)
    #hx.from_channel_ts(ch,how="data")
    # run_01.metadata.sample_rate = 40.0
    # run_01.metadata.time_period.start = datetime.datetime(2004,9,28,0,0,0)
    # run_01.metadata.time_period.end = datetime.datetime(2004, 9, 28, 2, 0, 0)
    #run_01.station_metadata = "PKD"
    #run_01.write_metadata()
    #?run_01.from_channel_ts()


    Parameters
    ----------
    direct_from_xml

    Returns  type(run_obj)
    -------
    type(run_obj)



    TODO: @Jared: can we make mth5_obj.open_mth5(str(h5_path), "w")
    work with Path() object rather than str(path)?
    """
    if h5_path is None:
        h5_path = Path("test.h5")
    else:
        h5_path = Path(h5_path)
    if h5_path.exists():
        h5_path.unlink()
    mth5_obj = MTH5()
    mth5_obj.open_mth5(str(h5_path), "w")
    mth5_obj.from_experiment(experiment)

    if "REW09" in mth5_obj.station_list: #old test
        run_obj = mth5_obj.get_run("REW09", "a")
    elif "PKD" in mth5_obj.station_list: #pkd test
        run_obj = mth5_obj.get_run("PKD", "001") #this run is created here
        print(experiment.surveys[0].stations[0].runs[0])
        check_run_channels_have_expected_properties(run_obj)
    else:
        print("skipping creation of run ")
        raise Exception

    return run_obj




def cast_run_to_run_ts(run, array_list=None, station_id=None):
    """
    add to mth5 helpers?
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
        print("Not working with STATIONXML_02-- FDSN Tide hack")
        xml_path = Path("single_station_mt.xml")
        #xml_path = STATIONXML_02
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



def test_filter_stages():
    """
    Sanity check to look at each stage of the filters.  Just want to look at their spectra for now,
    input/output units should be added also, but the belongs in MTH5 or mt_metadata
    Returns
    -------

    """
    pass


def test_filter_control():
    print("move this from driver")
    pass

#</LOAD SOME DATA FROM A SINGLE STATION>

def set_driver_parameters():
    driver_parameters = {}
    driver_parameters["create_xml"] = True
    driver_parameters["test_filter_control"] = True
    driver_parameters["run_ts_from_xml_01"] = True
    driver_parameters["run_ts_from_xml_02"] = True
    driver_parameters["run_ts_from_xml_03"] = True
    driver_parameters["initialize_data"] = True
    return driver_parameters

def test_can_access_fap_filters():
    fap_inventory = get_inventory_from_test_data_config("fap_test")
    describe_inventory_stages(fap_inventory)
    #<HERE IS THE SPOT TO DROP TRACE TO REVIEW INGEST OF FAP>
    experiment = get_experiment_from_obspy_inventory(fap_inventory)
    filters_dict = experiment.surveys[0].filters
    print(filters_dict)
    fap = filters_dict['mfn_0']
    num_filters = len(experiment.surveys[0].filters)
    # if num_filters ==2:
    #     print("probably not correct yet")


def main():
    #test_experiment_from_station_xml()
    #test_can_access_fap_filters()

    driver_parameters = set_driver_parameters()
    #<CREATE METADATA XML>
    if driver_parameters["create_xml"]:
        dataset_ids = ["pkd_test_00", "sao_test_00",]
        for dataset_id in dataset_ids:
            test_dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
            inventory = test_dataset_config.get_inventory_from_iris(ensure_inventory_stages_are_named=False)
            experiment = get_experiment_from_obspy_inventory(inventory)
            test_dataset_config.save_xml(experiment)#, tag="20210522")

    #</CREATE METADATA XML>

    #<TEST FILTER CONTROL>
    if driver_parameters["test_filter_control"]:
        filter_control_example()
        dataset_id = "pkd_test_00"
        test_dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
        xml_path = test_dataset_config.get_station_xml_filename()
        filter_control_example(xml_path=xml_path)
    #</TEST FILTER CONTROL>



    #<TEST RunTS FROM XML>
    #method 1 is in aurora driver
        # <METHOD2>
    if driver_parameters["run_ts_from_xml_02"]:
        dataset_id = "pkd_test_00"
        test_dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
        pkd_xml = test_dataset_config.get_station_xml_filename()
        experiment = get_experiment_from_xml_path(pkd_xml)
        run_obj = embed_experiment_into_run("PKD", experiment)
        runts_obj = cast_run_to_run_ts(run_obj, station_id="PKD")
        # </METHOD2>

        # <METHOD3>
    if driver_parameters["run_ts_from_xml_03"]:
        try:
            experiment = get_experiment_from_xml_path(single_station_xml_template)
            run_obj = embed_experiment_into_run("REW09", experiment)
            runts_obj = cast_run_to_run_ts(run_obj)
        except KeyError:
            print("FDSN TIDE HACK CAUSING EXCEPTION")
        # </METHOD3>
    #</TEST RunTS FROM XML>

    #<INITIALIZE DATA>
    if driver_parameters["initialize_data"]:
        pkd_mvts = get_example_data(station_id="PKD", component_station_label=True)
        sao_mvts = get_example_data(station_id="SAO", component_station_label=True)
        pkd = pkd_mvts.dataset
        sao = sao_mvts.dataset
        pkd.update(sao)
    #</INITIALIZE DATA>
    print("try to combine these runs")


if __name__ == "__main__":
    main()
    print("Fin")
