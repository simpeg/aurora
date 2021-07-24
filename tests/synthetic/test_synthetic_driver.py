from pathlib import Path

from aurora.pipelines.process_mth5 import process_mth5_run
from aurora.sandbox.plot_helpers import plot_tf_obj
from make_mth5_from_asc import create_mth5_synthetic_file
from make_mth5_from_asc import create_mth5_synthetic_file_for_array
from make_processing_configs import create_run_config_for_test_case
from synthetic_station_config import STATION_01_CFG
from synthetic_station_config import STATION_02_CFG
from synthetic_station_config import ACTIVE_FILTERS



def test_create_mth5():
    create_mth5_synthetic_file(STATION_01_CFG, plot=False)
    create_mth5_synthetic_file(STATION_02_CFG)
    create_mth5_synthetic_file_for_array([STATION_01_CFG, STATION_02_CFG])


def test_create_run_configs():
    create_run_config_for_test_case("test1")
    create_run_config_for_test_case("test2")
    create_run_config_for_test_case("test12rr")


def process_synthetic_1():
    test_config = Path("config", "test1_run_config.json")
    z_file_path = Path("test1_aurora.zss")
    z_file_path = z_file_path.absolute()
    run_id = "001"
    tf_collection = process_mth5_run(test_config, run_id, units="MT",
                                     show_plot=True,
                                     z_file_path=z_file_path)
    #z_file_path = Path("test1_aurora.zss")
    #tf_collection.write_emtf_z_file("test1_aurora.zss")
    print("RETURN TF OBJ AND PLOT ENMASSE ONCE MULTIDEC IS RUNNING")

def process_synthetic_2():
    test_config = Path("config", "test2_run_config.json")
    run_id = "001"
    process_mth5_run(test_config, run_id, units="MT")

def process_synthetic_rr12():
    test_config = Path("config", "test12rr_run_config.json")
    run_id = STATION_01_CFG["run_id"]
    process_mth5_run(test_config, run_id, units="MT")


def test_process_mth5():
    #create_mth5_synthetic_file(STATION_01_CFG, plot=False)
    create_run_config_for_test_case("test1")
    process_synthetic_1()
    #process_synthetic_2()
    #process_synthetic_rr12()

def main():
    #test_create_mth5()
    #test_create_run_configs()
    test_process_mth5()

if __name__ == '__main__':
    main()
