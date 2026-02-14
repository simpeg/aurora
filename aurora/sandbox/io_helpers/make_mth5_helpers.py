"""
    This module contains helper functions for making mth5 from FDSN clients.

"""
import pathlib
from pathlib import Path
from typing import Optional, Union

import obspy
from loguru import logger
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.timeseries import RunTS
from mth5.utils.helpers import initialize_mth5

from aurora.sandbox.obspy_helpers import (
    align_streams,
    make_channel_labels_fdsn_compliant,
    trim_streams_to_common_timestamps,
)
from aurora.sandbox.triage_metadata import (
    triage_missing_coil_hollister,
    triage_mt_units_electric_field,
    triage_mt_units_magnetic_field,
)


def create_from_server_multistation(
    fdsn_dataset,
    target_folder: Optional[pathlib.Path] = Path(),
    run_id: Optional[str] = "001",
    force_align_streams: Optional[bool] = True,
    triage_units: Optional[Union[list, None]] = None,
    triage_missing_coil: Optional[bool] = False,
) -> pathlib.Path:
    """

    This function builds an MTH5 file from FDSN client. The input dataset is described by fdsn_dataset.

    Parameters
    ----------
    fdsn_dataset: aurora.sandbox.io_helpers.fdsn_dataset.FDSNDataset
        Description of the dataset to create
    target_folder: Optional[pathlib.Path]
        The folder to create the dataset (mth5 file)
    run_id: str
        Label for the run
    force_align_streams: bool
        If True, the streams will be aligned if they are offset
    triage_units: list or None
        elements of the list should be in ["V/m to mV/km", "T to nT" ]
        These values in the list will result in an additional filter being added to
        the electric or magnetic field channels.
    triage_missing_coil: bool

    Returns
    -------
    h5_path: pathlib.Path
        The path to the mth5 that was built.
    """

    # Get Experiement
    try:
        inventory = fdsn_dataset.get_inventory(ensure_inventory_stages_are_named=True)
    # if inventory is None:
    #     print("Inventory Access Failed - NCEDC may be down")
    #     raise TypeError("None returned instead of Inventory")
    #     return None

    except Exception as e:  # FDSNException:
        logger.error(f"Exception {e}")
        # raise ValueError("NCEDC is Down, cannot build data")
        return
    translator = XMLInventoryMTExperiment()
    experiment = translator.xml_to_mt(inventory_object=inventory)

    # TRIAGE ONE-OFF ISSUE WITH UNITS
    if triage_units is not None:
        if "V/m to mV/km" in triage_units:
            experiment = triage_mt_units_electric_field(experiment)
        if "T to nT" in triage_units:
            experiment = triage_mt_units_magnetic_field(experiment)
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
        logger.warning("WARNING: ALIGN STREAMS NOT ROBUSTLY TESTED")
        streams = align_streams(streams, fdsn_dataset.starttime)
    streams = trim_streams_to_common_timestamps(streams)

    streams_dict = {}
    station_groups = {}
    # Iterate over stations, packing runs with data (not robust)
    for i_station, station_id in enumerate(mth5_obj.station_list):
        station_traces = [tr for tr in streams.traces if tr.stats.station == station_id]
        streams_dict[station_id] = obspy.core.Stream(station_traces)
        station_groups[station_id] = mth5_obj.get_station(station_id)
        run_metadata = experiment.surveys[0].stations[i_station].runs[0]
        run_metadata.id = (
            run_id  #  This seems to get ignored by the call to from_obspy_stream below
        )
        run_ts_obj = RunTS()
        run_ts_obj.from_obspy_stream(streams_dict[station_id], run_metadata)
        run_ts_obj.run_metadata.id = run_id  # Force setting run id
        run_group = station_groups[station_id].add_run(run_id)
        run_group.from_runts(run_ts_obj)
    mth5_obj.close_mth5()
    return h5_path
