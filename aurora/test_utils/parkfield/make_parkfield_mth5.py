"""
Create Parkfield / Hollister mth5 to use as test data

"""
from aurora.test_utils.dataset_definitions import TEST_DATA_SET_CONFIGS
from mth5.utils.helpers import read_back_data
from mth5.helpers import close_open_files
from aurora.sandbox.io_helpers.make_mth5_helpers import create_from_server_multistation
from aurora.test_utils.parkfield.path_helpers import PARKFIELD_PATHS

DATA_SOURCES = ["NCEDC", "https://service.ncedc.org/"]
DATASET_ID = "pkd_sao_test_00"
FDSN_DATASET = TEST_DATA_SET_CONFIGS[DATASET_ID]


def select_data_source():
    from obspy.clients.fdsn import Client

    ok = False
    while not ok:
        for data_source in DATA_SOURCES:
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


def make_pkdsao_mth5(fdsn_dataset):
    """ """
    close_open_files()
    fdsn_dataset.data_source = select_data_source()
    fdsn_dataset.initialize_client()
    h5_path = create_from_server_multistation(
        fdsn_dataset,
        target_folder=PARKFIELD_PATHS["data"],
        triage_units="V/m to mV/km",
    )

    for station in fdsn_dataset.station.split(","):
        print(station)
        read_back_data(h5_path, station, "001")
    return h5_path


def ensure_h5_exists():
    """

    Returns
    -------

    """

    h5_path = PARKFIELD_PATHS["data"].joinpath(FDSN_DATASET.h5_filebase)
    if h5_path.exists():
        return h5_path

    try:
        h5_path = make_pkdsao_mth5(FDSN_DATASET)
        return h5_path
    except Exception as e:
        print(f"Encountered {e} Exception - make_pkdsao_mth5 failed")
        print("Check data server connection")
        raise IOError


def main():
    make_pkdsao_mth5()


if __name__ == "__main__":
    main()
