"""
One off method to help read in transfer function dumps provided by Gary from some of
the matlab tests.

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


from aurora.config.metadata.processing import Processing
from aurora.config.emtf_band_setup import BANDS_256_29_FILE
from aurora.general_helper_functions import TEST_PATH
from aurora.sandbox.io_helpers.emtf_band_setup import EMTFBandSetupFile
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.transfer_function.emtf_z_file_helpers import clip_bands_from_z_file
from aurora.transfer_function.emtf_z_file_helpers import get_default_orientation_block
from aurora.transfer_function.plot.rho_phi_helpers import plot_phi
from aurora.transfer_function.plot.rho_phi_helpers import plot_rho
from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)


def test_matlab_zfile_reader(case_id="IAK34ss", make_plot=False):
    """

    Parameters
    ----------
    case_id: string
        one of ["IAK34ss", "synthetic"]
        currently only IAK34ss is supported
    make_plot: bool
        Set to True when debugging

    Takes a stored matlab file: "IAK34_struct_zss.mat", reads it in, and packs
    a z-file from it.

    There is a stored version of the z-file from July 2022 (archived in git in Sept
    2022) that we assert equality with to pass the test.

    ToDo: There is another stored mat file:"TS1zss20210831.mat"
    this appears to be synthetic station test1, processed with BANDS_256_29_FILE
    The output z-file from this can be compared to test1_aurora_matlab.zss.
    The results should agree approximately.


    """
    test_dir_path = TEST_PATH.joinpath("io")
    bs_file = BANDS_256_29_FILE
    if case_id == "synthetic":
        n_periods_clip = 3  # for synthetic case
        estimator_engine = "RME"
        local_station_id = "test1"
        remote_station_id = ""
        input_channels = ["hx", "hy"]
        output_channels = [
            "hz",
            "ex",
            "ey",
        ]
        reference_channels = []
        z_mat = "TS1zss20210831.mat"
        archived_z_file_path = None
        z_file_path = "from_matlab.zss"
    elif case_id == "IAK34ss":
        n_periods_clip = 3
        estimator_engine = "RME"
        local_station_id = "IAK34"
        remote_station_id = ""
        input_channels = ["hx", "hy"]
        output_channels = [
            "hz",
            "ex",
            "ey",
        ]
        reference_channels = []
        z_mat = test_dir_path.joinpath("IAK34_struct_zss.mat")
        archived_z_file_path = test_dir_path.joinpath("archived_from_matlab.zss")
        z_file_path = test_dir_path.joinpath("from_matlab.zss")

    orientation_strs = get_default_orientation_block()

    sample_rate = 1.0
    tf_dict = {}

    p = Processing()
    p.stations.local.id = local_station_id
    if remote_station_id:
        p.stations.remote.id = remote_station_id
    emtf_band_setup = EMTFBandSetupFile(filepath=bs_file, sample_rate=sample_rate)
    num_samples_window = 256
    decimation_factors = [1, 4, 4, 4]
    band_edges = emtf_band_setup.compute_band_edges(
        decimation_factors=decimation_factors,
        num_samples_window=4 * [num_samples_window],
    )
    p.assign_bands(band_edges, sample_rate, decimation_factors, 4 * num_samples_window)
    for i_dec in range(4):
        p.decimations[i_dec].decimation.sample_rate = sample_rate
        p.decimations[i_dec].window.num_samples = num_samples_window
        p.decimations[i_dec].estimator.engine = estimator_engine
        p.decimations[i_dec].input_channels = input_channels
        p.decimations[i_dec].output_channels = output_channels
        p.decimations[i_dec].reference_channels = reference_channels

        tf_obj = p.make_tf_level(i_dec)
        tf_dict[i_dec] = tf_obj
        sample_rate /= 4.0

    tmp = sio.loadmat(z_mat)
    if case_id == "synthetic":
        stuff = tmp["temp"][0][0].tolist()
    elif case_id == "IAK34ss":
        stuff = tmp["TFstruct"][0][0].tolist()
    TF = stuff[4]
    periods = stuff[5]
    cov_ss = stuff[7]
    cov_nn = stuff[8]
    n_data = stuff[9]
    R2 = stuff[10]
    # stderr = stuff[12]
    # freq = stuff[14]

    nan_cov_nn = []
    for i in range(len(periods)):
        if np.isnan(cov_nn[:, :, i]).any():
            nan_cov_nn.append(i)
            print(f"NAN {i}")

    if case_id == "synthetic":
        cov_nn[:, :, 28] = cov_nn[:, :, 27]
    elif case_id == "IAK34ss":
        for i in range(12):
            cov_nn[:, :, i] = cov_nn[:, :, 12]

    # FIX NAN / INF

    tf_dict[0].tf.data = TF[:, :, :11]
    tf_dict[0].cov_nn.data = cov_nn[:, :, :11]
    tf_dict[0].cov_ss_inv.data = cov_ss[:, :, :11]
    tf_dict[0].num_segments.data = n_data[:, :11]
    tf_dict[0].R2.data = R2[:, :11]

    tf_dict[1].tf.data = TF[:, :, 11:17]
    tf_dict[1].cov_nn.data = cov_nn[:, :, 11:17]
    tf_dict[1].cov_ss_inv.data = cov_ss[:, :, 11:17]
    tf_dict[1].num_segments.data = n_data[:, 11:17]
    tf_dict[1].R2.data = R2[:, 11:17]

    tf_dict[2].tf.data = TF[:, :, 17:23]
    tf_dict[2].cov_nn.data = cov_nn[:, :, 17:23]
    tf_dict[2].cov_ss_inv.data = cov_ss[:, :, 17:23]
    tf_dict[2].num_segments.data = n_data[:, 17:23]
    tf_dict[2].R2.data = R2[:, 17:23]

    tf_dict[3].tf.data = TF[:, :, 23:]
    tf_dict[3].cov_nn.data = cov_nn[:, :, 23:]
    tf_dict[3].cov_ss_inv.data = cov_ss[:, :, 23:]
    tf_dict[3].num_segments.data = n_data[:, 23:]
    tf_dict[3].R2.data = R2[:, 23:]

    for i_dec in range(4):
        tf_dict[i_dec].tf.data = tf_dict[i_dec].tf.data

    tfc = TransferFunctionCollection(
        header=tf_obj.tf_header, tf_dict=tf_dict, processing_config=p
    )
    tfc.write_emtf_z_file(z_file_path, orientation_strs=orientation_strs)

    if n_periods_clip:
        clip_bands_from_z_file(z_file_path, n_periods_clip, n_sensors=5)

    zfile = read_z_file(z_file_path)
    archived_zfile = read_z_file(archived_z_file_path)

    zfile.apparent_resistivity(angle=0)
    archived_zfile.apparent_resistivity(angle=0)
    assert (zfile.rxy == archived_zfile.rxy).all()
    assert (zfile.ryx == archived_zfile.ryx).all()
    assert (zfile.pxy == archived_zfile.pxy).all()
    assert (zfile.pyx == archived_zfile.pyx).all()

    if make_plot:
        scl = 1.0
        fig, axs = plt.subplots(nrows=2, figsize=(11, 8.5), dpi=300, sharex=True)
        markersize = 1
        plot_rho(
            axs[0],
            zfile.periods,
            zfile.rxy * scl,
            label="rxy",
            markersize=markersize,
            color="red",
        )
        plot_rho(
            axs[0],
            zfile.periods,
            zfile.ryx * scl,
            label="ryx",
            markersize=markersize,
            color="blue",
        )
        axs[0].legend()
        plot_phi(
            axs[1],
            zfile.periods,
            zfile.pxy,
            label="pxy",
            markersize=markersize,
            color="red",
        )
        plot_phi(
            axs[1],
            zfile.periods,
            zfile.pyx,
            label="pyx",
            markersize=markersize,
            color="blue",
        )
        axs[0].set_ylim(1, 1000)
        axs[0].set_xlim(1, 10000)
        plt.show()
    print("success!")
