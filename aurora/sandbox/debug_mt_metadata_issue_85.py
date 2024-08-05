"""
    This module contains a test that should be moved into mt_metadata
    TODO: Make this a test in mt_metadata
"""
from mt_metadata.timeseries.location import Location
from mth5.mth5 import MTH5
from loguru import logger


def test_can_add_location():
    """
    20220624: This mini test is being factored out of the normal tests.
    Want to assign location.  The location lands in the station_group but not in the
    run_group.
    Reported this conundrum https://github.com/kujaku11/mt_metadata/issues/85

    When I close and reopen the file, I cannot find the latitude I set anywhere.

    Returns
    -------

    """
    # make an MTH5
    m = MTH5(file_version="0.1.0")
    m.open_mth5("location_test.h5", mode="w")
    station_group = m.add_station("eureka")

    location = Location()
    location.latitude = 17.996
    station_group.metadata.location = location
    # setting latitude as above does not wind up in run, but is in station_group

    run_group = station_group.add_run("001")
    run_group.station_group.metadata.location = location
    # setting latitude as above line does not wind up in the run either"

    logger.info("Why don't the following values agree??")
    logger.info(f"station group {station_group.metadata.location.latitude}")
    logger.info(f"Run Group {run_group.station_group.metadata.location.latitude}")
    m.close_mth5()

    logger.info("Reopen the file and check if update was done on close()")
    m.open_mth5("location_test.h5", mode="r")
    eureka = m.get_station("eureka")
    logger.info(f"station group {eureka.metadata.location.latitude}")
    run_001 = eureka.get_run("001")
    logger.info(f"Run Group {run_001.station_group.metadata.location.latitude}")
    m.close_mth5()
    return


def main():
    test_can_add_location()


if __name__ == "__main__":
    main()
