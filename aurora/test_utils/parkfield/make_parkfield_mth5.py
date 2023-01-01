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

PKDSAO_DATA_SOURCE = "https://service.ncedc.org/"  # "NCEDC"


def make_parkfield_mth5():
    close_open_files()
    makeparkfield_h5_path = DATA_PATH.joinpath("pkd_test_00.h5")
    dataset_id = "pkd_test_00"
    print(f" 1 makeparkfield_h5_path.exists() {makeparkfield_h5_path.exists()}")
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    create_from_server_multistation(
        dataset_config,
        data_source=PKDSAO_DATA_SOURCE,
        target_folder=DATA_PATH,
        triage_units="V/m to mV/km",
    )
    print(f" 2 makeparkfield_h5_path.exists() {makeparkfield_h5_path.exists()}")
    h5_path = DATA_PATH.joinpath(dataset_config.h5_filebase)
    print(f" 3 makeparkfield_h5_path.exists() {makeparkfield_h5_path.exists()}")
    read_back_data(h5_path, "PKD", "001")
    print(f" 4 makeparkfield_h5_path.exists() {makeparkfield_h5_path.exists()}")


def make_parkfield_hollister_mth5():
    close_open_files()

    dataset_id = "pkd_sao_test_00"
    dataset_config = TEST_DATA_SET_CONFIGS[dataset_id]
    create_from_server_multistation(
        dataset_config,
        data_source=PKDSAO_DATA_SOURCE,
        target_folder=DATA_PATH,
        triage_units="V/m to mV/km",
    )
    h5_path = DATA_PATH.joinpath(dataset_config.h5_filebase)
    pkd_result = read_back_data(h5_path, "PKD", "001")
    sao_result = read_back_data(h5_path, "SAO", "001")
    print(pkd_result)
    print(sao_result)


# def test_make_hollister_mth5():
#     dataset_id = "sao_test_00"
#     create_from_server(dataset_id, data_source="NCEDC")
#     h5_path = DATA_PATH.joinpath(f"{dataset_id}.h5")
#     run, runts = read_back_data(h5_path, "SAO", "001")
#     print("hello")


def main():
    make_parkfield_mth5()
    # test_make_hollister_mth5()
    make_parkfield_hollister_mth5()


if __name__ == "__main__":
    main()
