"""
Relates to github aurora issue #17
Ingest the Parkfield data make mth5 to use as the interface for tests

2021-09-17: Modifying create methods to use FDSNDatasetConfig as input rather than
dataset_id
"""
from aurora.test_utils.dataset_definitions import TEST_DATA_SET_CONFIGS
from mth5.utils.helpers import read_back_data
from mth5.helpers import close_open_files
from aurora.sandbox.io_helpers.make_mth5_helpers import (
    create_from_server_multistation,
)
from aurora.test_utils.parkfield.path_helpers import DATA_PATH

PKDSAO_DATA_SOURCE = "https://service.ncedc.org/"
# Add alternate source: "NCEDC",


def make_pkdsao_mth5(dataset_id):
    """

    Parameters
    ----------
    dataset_id: str
        Either "pkd_test_00" or "pkd_sao_test_00".  Specifies the h5 to build,
        single station or remote reference

    """
    close_open_files()
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    create_from_server_multistation(
        dataset_config,
        data_source=PKDSAO_DATA_SOURCE,
        target_folder=DATA_PATH,
        triage_units="V/m to mV/km",
    )
    h5_path = DATA_PATH.joinpath(dataset_config.h5_filebase)
    for station in dataset_config.station.split(","):
        read_back_data(h5_path, station, "001")


def main():
    dataset_ids = ["pkd_test_00", "pkd_sao_test_00"]
    for dataset_id in dataset_ids:
        make_pkdsao_mth5(dataset_id)


if __name__ == "__main__":
    main()
