from mt_metadata.timeseries.filters.helper_functions import MT2SI_ELECTRIC_FIELD_FILTER
from mt_metadata.timeseries.filters.helper_functions import MT2SI_MAGNETIC_FIELD_FILTER

from loguru import logger


def triage_mt_units_electric_field(experiment):
    """
    One-off example of adding a filter to an mth5 in the case where the electric
    field data are given in V/m, but they were expected in mV/km.  This adds the
    correct filter to the metadata so that the calibrated data have units of
    mV/km.
     Parameters
    ----------
    experiment ;

    Returns
    -------

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


def triage_mt_units_magnetic_field(experiment):
    """
    One-off example of adding a filter to an mth5 in the case where the electric
    field data are given in V/m, but they were expected in mV/km.  This adds the
    correct filter to the metadata so that the calibrated data have units of
    mV/km.
     Parameters
    ----------
    experiment ;

    Returns
    -------

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


def triage_missing_coil_hollister(experiment):
    """
    One off for hollister missing hy metadata for no reason I can tell
    Parameters
    ----------
    experiment

    Returns
    -------

    """
    survey = experiment.surveys[0]
    stations = survey.stations
    for station in stations:
        if station.id == "SAO":
            runs = station.runs
            for run in runs:
                channels = run.channels
                logger.info(channels)
                for channel in channels:
                    logger.info(channel.id)

        # station = stations[i_station]
        runs = station.runs[0]
        logger.info("help")
