import obspy
from obspy.clients.fdsn.header import FDSNNoServiceException
from pathlib import Path

from aurora.sandbox.obspy_helpers import align_streams
from aurora.sandbox.obspy_helpers import make_channel_labels_fdsn_compliant
from aurora.sandbox.obspy_helpers import trim_streams_to_acquisition_run
from aurora.sandbox.triage_metadata import triage_missing_coil_hollister
from aurora.sandbox.triage_metadata import triage_mt_units_electric_field
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.utils.helpers import initialize_mth5
from mth5.timeseries import RunTS


def create_from_server_multistation(
    dataset_config,
    data_source="IRIS",
    target_folder=Path(),
    run_id="001",
    force_align_streams=True,
    triage_units=None,
    triage_missing_coil=False,
    **kwargs
):
    """

    Parameters
    ----------
    dataset_config
    data_source
    target_folder
    run_id : string
        This is a temporary workaround. A more robust program that assigns run
        numbers, and/or gets run labels from StationXML is needed

    Returns
    -------

    """

    # <GET EXPERIMENT>
    try:
        inventory = dataset_config.get_inventory_from_client(
            ensure_inventory_stages_are_named=True,
            base_url=data_source,
        )
    except FDSNNoServiceException:
        raise IOError("NCEDC is Down, cannot build data")

    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)
    # </GET EXPERIMENT>

    # <TRIAGE ONE-OFF ISSUE WITH UNITS>
    if triage_units:
        if triage_units == "V/m to mV/km":
            experiment = triage_mt_units_electric_field(experiment)
    if triage_missing_coil:
        experiment = triage_missing_coil_hollister(experiment)

    # </TRIAGE ONE-OFF ISSUE WITH UNITS>

    # <INITIALIZE MTH5>
    h5_path = target_folder.joinpath(dataset_config.h5_filebase)
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)
    # </INITIALIZE MTH5>
    dataset_config.describe()

    # <GET DATA STREAMS>
    streams = dataset_config.get_data_via_fdsn_client(data_source=data_source)
    streams = make_channel_labels_fdsn_compliant(streams)
    if force_align_streams:
        print("WARNING: ALIGN STREAMS NOT ROBUSTLY TESTED")
        streams = align_streams(streams, dataset_config.starttime)
    streams = trim_streams_to_acquisition_run(streams)
    # </GET DATA STREAMS>

    streams_dict = {}
    station_groups = {}
    # NEED TO ITERATE OVER RUNS HERE - THIS IS NOT ROBUST
    for i_station, station_id in enumerate(mth5_obj.station_list):
        station_traces = [
            tr for tr in streams.traces if tr.stats.station == station_id
        ]
        streams_dict[station_id] = obspy.core.Stream(station_traces)
        station_groups[station_id] = mth5_obj.get_station(station_id)

        run_metadata = experiment.surveys[0].stations[i_station].runs[0]
        run_ts_obj = RunTS()
        run_ts_obj.from_obspy_stream(streams_dict[station_id], run_metadata)
        run_ts_obj.run_metadata.id = run_id
        run_group = station_groups[station_id].add_run(run_id)
        run_group.from_runts(run_ts_obj)
    mth5_obj.close_mth5()

    return
