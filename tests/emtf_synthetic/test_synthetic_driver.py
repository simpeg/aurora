from make_mth5_from_asc import create_mth5_synthetic_file
from make_mth5_from_asc import create_mth5_synthetic_file_for_array
from test_processing import process_sythetic_mth5_single_station
from test_processing import process_sythetic_mth5_remote_reference
from synthetic_station_config import STATION_01_CFG
from synthetic_station_config import STATION_02_CFG
from synthetic_station_config import ACTIVE_FILTERS



def test_create_mth5():
    create_mth5_synthetic_file(STATION_01_CFG, plot=False)
    create_mth5_synthetic_file(STATION_02_CFG)
    create_mth5_synthetic_file_for_array([STATION_01_CFG, STATION_02_CFG])

def test_process_mth5():
    test1_cfg = "test1_processing_config.json"
    process_sythetic_mth5_single_station(test1_cfg, STATION_01_CFG["run_id"])
    test2_cfg = "test2_processing_config.json"
    process_sythetic_mth5_single_station(test2_cfg, STATION_02_CFG["run_id"])
    process_sythetic_mth5_remote_reference(STATION_01_CFG, STATION_02_CFG)


def main():
    test_create_mth5()
    test_process_mth5()

if __name__ == '__main__':
    main()
