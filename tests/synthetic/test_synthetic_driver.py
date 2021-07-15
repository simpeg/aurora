from pathlib import Path

from aurora.pipelines.process_mth5 import process_mth5_decimation_level
from aurora.sandbox.plot_helpers import plot_tf_obj
from make_mth5_from_asc import create_mth5_synthetic_file
from make_mth5_from_asc import create_mth5_synthetic_file_for_array
from make_processing_configs import create_config_for_test_case
from synthetic_station_config import STATION_01_CFG
from synthetic_station_config import STATION_02_CFG
from synthetic_station_config import ACTIVE_FILTERS



def test_create_mth5():
    create_mth5_synthetic_file(STATION_01_CFG, plot=False)
    create_mth5_synthetic_file(STATION_02_CFG)
    create_mth5_synthetic_file_for_array([STATION_01_CFG, STATION_02_CFG])

def test_create_processing_configs():
    create_config_for_test_case("test1")
    create_config_for_test_case("test2")
    create_config_for_test_case("test12rr")

def test_synthetic_1():
    test_cfg = Path("config", "test1_processing_config.json")
    tf_obj = process_mth5_decimation_level(test_cfg, STATION_01_CFG["run_id"])
    plot_tf_obj(tf_obj)

def test_synthetic_2():
    test_cfg = test_cfg = Path("config", "test2_processing_config.json")
    tf_obj = process_mth5_decimation_level(test_cfg, STATION_02_CFG["run_id"])
    plot_tf_obj(tf_obj)

def test_synthetic_rr12():
    test_cfg = test_cfg = Path("config", "test12rr_processing_config.json")
    tf_obj = process_mth5_decimation_level(test_cfg, STATION_01_CFG["run_id"])
    plot_tf_obj(tf_obj)


def test_process_mth5():
    test_synthetic_1()
    test_synthetic_2()
    test_synthetic_rr12()


def main():
    test_create_mth5()
    test_create_processing_configs()
    test_process_mth5()

if __name__ == '__main__':
    main()
