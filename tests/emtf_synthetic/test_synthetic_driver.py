from pathlib import Path
from make_mth5_from_asc import create_mth5_synthetic_file
from make_mth5_from_asc import create_mth5_synthetic_file_for_array
from aurora.transfer_function.rho_plot import RhoPlot
from aurora.pipelines.process_mth5 import process_mth5_decimation_level
from synthetic_station_config import STATION_01_CFG
from synthetic_station_config import STATION_02_CFG
from synthetic_station_config import ACTIVE_FILTERS
from test_create_processing_configs import create_config_for_test_case


def plot_tf_obj(tf_obj):
    import matplotlib.pyplot as plt
    plotter = RhoPlot(tf_obj)
    fig, axs = plt.subplots(nrows=2)
    plotter.rho_sub_plot(axs[0])
    plotter.phase_sub_plot(axs[1])
    plt.show()


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
    print("GET PLOTTER FROM MTpy")
    plot_tf_obj(tf_obj)

def test_synthetic_2():
    test_cfg = test_cfg = Path("config", "test2_processing_config.json")
    tf_obj = process_mth5_decimation_level(test_cfg, STATION_02_CFG["run_id"])
    print("GET PLOTTER FROM MTpy")
    plot_tf_obj(tf_obj)

def test_synthetic_rr12():
    test_cfg = test_cfg = Path("config", "test12rr_processing_config.json")
    tf_obj = process_mth5_decimation_level(test_cfg, STATION_01_CFG["run_id"])
    print("GET PLOTTER FROM MTpy")
    plot_tf_obj(tf_obj)


def test_process_mth5():
    test_synthetic_1()
    test_synthetic_2()
    test_synthetic_rr12()


def main():
    #test_create_mth5()
    test_create_processing_configs()
    test_process_mth5()

if __name__ == '__main__':
    main()
