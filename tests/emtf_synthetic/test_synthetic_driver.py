from make_mth5_from_asc import create_mth5_synthetic_file
from make_mth5_from_asc import create_mth5_synthetic_file_for_array
from test_processing import process_sythetic_mth5_single_station
from test_processing import process_sythetic_mth5_remote_reference
from synthetic_station_config import STATION_01_CFG
from synthetic_station_config import STATION_02_CFG
from synthetic_station_config import ACTIVE_FILTERS






def main():
    create_mth5_synthetic_file(STATION_01_CFG, plot=False)
    create_mth5_synthetic_file(STATION_02_CFG)
    create_mth5_synthetic_file_for_array([STATION_01_CFG, STATION_02_CFG])
    process_sythetic_mth5_single_station(STATION_01_CFG)
    process_sythetic_mth5_single_station(STATION_02_CFG)
    process_sythetic_mth5_remote_reference(STATION_01_CFG, STATION_02_CFG)

if __name__ == '__main__':
    main()
