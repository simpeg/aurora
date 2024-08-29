"""
    This module contains various helper functions that were used to fix errors in metadata.
"""

from mt_metadata.timeseries import Experiment
from mt_metadata.timeseries.filters.helper_functions import MT2SI_ELECTRIC_FIELD_FILTER
from mt_metadata.timeseries.filters.helper_functions import MT2SI_MAGNETIC_FIELD_FILTER
from loguru import logger
import mth5.groups


def triage_mt_units_electric_field(experiment: Experiment) -> Experiment:
    """
    Updates an mth5 experiment with filter information

    One-off example of adding a filter to an mth5 in the case where the electric
    field data are given in V/m, but they were expected in mV/km.  This adds the
    correct filter to the metadata so that the calibrated data have units of
    mV/km.

    Parameters
    ----------
    experiment: mt_metadata.timeseries.Experiment
        mt_metadata representation of station XML

    Returns
    -------
    experiment: mt_metadata.timeseries.Experiment
        Updated experiment with additional filter to make electric field into MT Units (mV/km)

    """
    logger.info(
        f"Add MT2SI_ELECTRIC_FIELD_FILTER to electric channels for parkfield here"
        f" {MT2SI_ELECTRIC_FIELD_FILTER} "
    )
    filter_name = MT2SI_ELECTRIC_FIELD_FILTER.name
    survey = experiment.surveys[0]
    survey.filters[filter_name] = MT2SI_ELECTRIC_FIELD_FILTER
    stations = survey.stations
    for station in stations:
        channels = station.runs[0].channels
        for channel in channels:
            if channel.component[0] == "e":
                channel.filter.name.insert(0, filter_name)
                channel.filter.applied.insert(0, True)
    return experiment


def triage_mt_units_magnetic_field(experiment: Experiment) -> Experiment:
    """
    Updates an mth5 experiment with filter information

    One-off example of adding a filter to an mth5 in the case where the magnetic
    field data are given in T, but they were expected in nT.  This adds the
    correct filter to the metadata so that the calibrated data have units of nT.

    Parameters
    ----------
    experiment: mt_metadata.timeseries.Experiment
        mt_metadata representation of station XML

    Returns
    -------
    experiment: mt_metadata.timeseries.Experiment
        Updated experiment with additional filter to make magnetic field into MT Units (nT)

    """
    logger.info(
        f"Add MT2SI_MAGNETIC_FIELD_FILTER to magnetic channels for parkfield"
        f" {MT2SI_MAGNETIC_FIELD_FILTER} "
    )
    filter_name = MT2SI_MAGNETIC_FIELD_FILTER.name
    survey = experiment.surveys[0]
    survey.filters[filter_name] = MT2SI_MAGNETIC_FIELD_FILTER
    stations = survey.stations
    for station in stations:
        channels = station.runs[0].channels
        for channel in channels:
            if channel.component[0] == "h":
                channel.filter.name.insert(0, filter_name)
                channel.filter.applied.insert(0, True)
    return experiment


def triage_missing_coil_hollister(experiment: Experiment) -> Experiment:
    """
    Fixes missing metadata for Hollister station

    One-off for hollister missing hy metadata in NCEDC archive.

    Parameters
    ----------
    experiment

    Returns
    -------

    """
    msg = "missing coil not fixed in metadata"
    raise NotImplementedError(msg)
    # survey = experiment.surveys[0]
    # stations = survey.stations
    # for station in stations:
    #     if station.id == "SAO":
    #         runs = station.runs
    #         for run in runs:
    #             channels = run.channels
    #             logger.info(channels)
    #             for channel in channels:
    #                 logger.info(channel.id)
    #
    #     # station = stations[i_station]
    #     runs = station.runs[0]
    #     logger.info("help")


def triage_run_id(expected_run_id: str, run_obj: mth5.groups.RunGroup) -> None:
    """
    Fixes metadata from an old version of MTH5.

    This situation was encountered during the Musgraves processing for 2023 HPC workshop.
    The MTH5 files being used were from a previous era, and the run_object metadata did not
    contain the expected value for run_id.

    Parameters
    ----------
    expected_run_id: string
        The expected name of the run
    run_obj: mth5.groups.run.RunGroup
        The run object that should have correct name.

    """
    try:
        assert expected_run_id == run_obj.metadata.id
    except AssertionError:
        logger.warning("WARNING Run ID in dataset_df does not match run_obj")
        logger.warning("WARNING Forcing run metadata to match dataset_df")
        run_obj.metadata.id = expected_run_id
    return
