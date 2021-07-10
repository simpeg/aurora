from make_mth5_from_asc import create_mth5_synthetic_file
from make_mth5_from_asc import create_mth5_synthetic_file_for_array
from aurora.transfer_function.rho_plot import RhoPlot
from aurora.pipelines.process_mth5 import process_mth5_decimation_level
#from test_processing import process_sythetic_mth5_single_station
from test_processing import process_sythetic_mth5_remote_reference
from synthetic_station_config import STATION_01_CFG
from synthetic_station_config import STATION_02_CFG
from synthetic_station_config import ACTIVE_FILTERS


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

def test_process_mth5():
    test1_cfg = "test1_processing_config.json"
    tf_obj = process_mth5_decimation_level(test1_cfg, STATION_01_CFG["run_id"])
    print("GET PLOTTER FROM MTpy")
    plot_tf_obj(tf_obj)

    test2_cfg = "test2_processing_config.json"
    process_mth5_decimation_level(test2_cfg, STATION_02_CFG["run_id"])
    plot_tf_obj(tf_obj)
    process_sythetic_mth5_remote_reference(STATION_01_CFG, STATION_02_CFG)


def main():
#    test_create_mth5()
    test_process_mth5()

if __name__ == '__main__':
    main()
