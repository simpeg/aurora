import obspy
from obspy.clients.fdsn.header import FDSNException
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
    fdsn_dataset,
    target_folder=Path(),
    run_id="001",
    force_align_streams=True,
    triage_units=None,
    triage_missing_coil=False,
    **kwargs,
):
    """

    Parameters
    ----------
    fdsn_dataset: aurora.sandbox.io_helpers.fdsn_dataset.FDSNDataset
    target_folder
    run_id : string
        This is a temporary workaround. A more robust program that assigns run
        numbers, and/or gets run labels from StationXML is needed

    Returns
    -------

    """

    # Get Experiement
    try:
        inventory = fdsn_dataset.get_inventory(ensure_inventory_stages_are_named=True)
    # if inventory is None:
    #     print("Inventory Access Failed - NCEDC may be down")
    #     raise TypeError("None returned instead of Inventory")
    #     return None

    except Exception as e:  # FDSNException:
        print(f"Exception {e}")
        # raise ValueError("NCEDC is Down, cannot build data")
        return
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)

    # TRIAGE ONE-OFF ISSUE WITH UNITS
    if triage_units:
        if triage_units == "V/m to mV/km":
            experiment = triage_mt_units_electric_field(experiment)
    if triage_missing_coil:
        experiment = triage_missing_coil_hollister(experiment)

    # INITIALIZE MTH5
    h5_path = target_folder.joinpath(fdsn_dataset.h5_filebase)
    mth5_obj = initialize_mth5(h5_path)
    mth5_obj.from_experiment(experiment)

    fdsn_dataset.describe()

    # GET DATA STREAMS
    streams = fdsn_dataset.get_data_via_fdsn_client()
    streams = make_channel_labels_fdsn_compliant(streams)
    if force_align_streams:
        print("WARNING: ALIGN STREAMS NOT ROBUSTLY TESTED")
        streams = align_streams(streams, fdsn_dataset.starttime)
    streams = trim_streams_to_acquisition_run(streams)

    streams_dict = {}
    station_groups = {}
    # Iterate over stations, packing runs with data (not robust)
    for i_station, station_id in enumerate(mth5_obj.station_list):
        station_traces = [tr for tr in streams.traces if tr.stats.station == station_id]
        streams_dict[station_id] = obspy.core.Stream(station_traces)
        station_groups[station_id] = mth5_obj.get_station(station_id)
        run_metadata = experiment.surveys[0].stations[i_station].runs[0]
        run_metadata.id = run_id
        run_ts_obj = RunTS()
        run_ts_obj.from_obspy_stream(streams_dict[station_id], run_metadata)
        run_group = station_groups[station_id].add_run(run_id)
        run_group.from_runts(run_ts_obj)
    mth5_obj.close_mth5()
    return h5_path
