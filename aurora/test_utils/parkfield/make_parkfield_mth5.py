"""
Create Parkfield / Hollister mth5 to use as test data

"""
from aurora.test_utils.dataset_definitions import TEST_DATA_SET_CONFIGS
from mth5.utils.helpers import read_back_data
from mth5.helpers import close_open_files
from aurora.sandbox.io_helpers.make_mth5_helpers import create_from_server_multistation
from aurora.test_utils.parkfield.path_helpers import DATA_PATH

PKDSAO_DATA_SOURCES = ["NCEDC", "https://service.ncedc.org/"]


def select_data_source():
    from obspy.clients.fdsn import Client

    ok = False
    while not ok:
        for data_source in PKDSAO_DATA_SOURCES:
            try:
                Client(base_url=data_source, force_redirect=True)
                ok = True
            except:
                print(f"Data source {data_source} not initializing")
    if not ok:
        print("No data sources for Parkfield / Hollister initializing")
        print("NCEDC probably down")
        raise ValueError
    else:
        return data_source


def make_pkdsao_mth5(dataset_id):
    """

    Parameters
    ----------
    dataset_id: str
        Either "pkd_test_00" or "pkd_sao_test_00".  Specifies the h5 to build,
        single station or remote reference

    """
    close_open_files()
    fdsn_dataset = TEST_DATA_SET_CONFIGS[dataset_id]
    fdsn_dataset.data_source = select_data_source()  # PKDSAO_DATA_SOURCE
    fdsn_dataset.initialize_client()
    create_from_server_multistation(
        fdsn_dataset,
        target_folder=DATA_PATH,
        triage_units="V/m to mV/km",
    )
    h5_path = DATA_PATH.joinpath(fdsn_dataset.h5_filebase)
    for station in fdsn_dataset.station.split(","):
        print(station)
        read_back_data(h5_path, station, "001")


def main():
    dataset_ids = ["pkd_test_00", "pkd_sao_test_00"]
    for dataset_id in dataset_ids:
        make_pkdsao_mth5(dataset_id)


if __name__ == "__main__":
    main()
