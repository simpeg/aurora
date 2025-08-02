"""
One off method to help read in transfer function dumps provided by Gary from some of
the matlab tests.

"""

import numpy as np
import pandas as pd
import scipy.io as sio

from aurora.config.metadata.processing import Processing
from aurora.config.emtf_band_setup import BANDS_256_29_FILE
from aurora.general_helper_functions import get_test_path
from aurora.pipelines.transfer_function_kernel import TransferFunctionKernel
from aurora.sandbox.io_helpers.emtf_band_setup import EMTFBandSetupFile
from aurora.transfer_function.emtf_z_file_helpers import clip_bands_from_z_file

from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from mt_metadata.timeseries.survey import Survey
from mt_metadata.transfer_functions.core import TF
from loguru import logger

from mth5.processing import KernelDataset

TEST_PATH = get_test_path()


def read_matlab_z_file(case_id, z_mat):
    tmp = sio.loadmat(z_mat)
    if case_id == "synthetic":
        tmp = tmp["temp"][0][0].tolist()
    elif case_id == "IAK34ss":
        tmp = tmp["TFstruct"][0][0].tolist()
    return tmp


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

    # 1. Define processing parameters based on which of the two tests are to be run
    if case_id == "synthetic":
        band_setup_file = BANDS_256_29_FILE
        field_data_sample_rate = 1.0
        num_samples_window = 256
        decimation_factors = [1, 4, 4, 4]
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
        matlab_z_file = "TS1zss20210831.mat"
        archived_z_file_path = None
        z_file_path = "from_matlab.zss"
    elif case_id == "IAK34ss":
        band_setup_file = BANDS_256_29_FILE
        field_data_sample_rate = 1.0
        num_samples_window = 256
        decimation_factors = [1, 4, 4, 4]
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
        matlab_z_file = test_dir_path.joinpath("IAK34_struct_zss.mat")
        archived_z_file_path = test_dir_path.joinpath("archived_from_matlab.zss")
        z_file_path = test_dir_path.joinpath("from_matlab.zss")

    # 2. Create an aurora processing config
    p = Processing()
    p.stations.local.id = local_station_id
    if remote_station_id:
        p.stations.remote.id = remote_station_id
    emtf_band_setup = EMTFBandSetupFile(
        filepath=band_setup_file, sample_rate=field_data_sample_rate
    )
    band_edges = emtf_band_setup.compute_band_edges(
        decimation_factors=decimation_factors,
        num_samples_window=4 * [num_samples_window],
    )
    p.assign_bands(
        band_edges,
        field_data_sample_rate,
        decimation_factors,
        num_samples_window,
    )

    # 3. populate decimation levels of processing config
    tf_dict = {}
    sample_rate = field_data_sample_rate
    for i_dec in range(4):
        p.decimations[i_dec].decimation.sample_rate = sample_rate
        p.decimations[i_dec].stft.window.num_samples = num_samples_window
        p.decimations[i_dec].estimator.engine = estimator_engine
        p.decimations[i_dec].input_channels = input_channels
        p.decimations[i_dec].output_channels = output_channels
        p.decimations[i_dec].reference_channels = reference_channels

        tffz_obj = p.make_tf_level(i_dec)
        tf_dict[i_dec] = tffz_obj
        sample_rate /= 4.0

    # 4. Read in the matlab_z object
    tmp = read_matlab_z_file(case_id, matlab_z_file)
    matlab_tf = tmp[4]
    periods = tmp[5]
    cov_ss = tmp[7]
    cov_nn = tmp[8]
    n_data = tmp[9]
    R2 = tmp[10]
    # stderr = tmp[12]
    # freq = tmp[14]

    # 4a. FIX NAN / INF
    nan_cov_nn = []
    for i in range(len(periods)):
        if np.isnan(cov_nn[:, :, i]).any():
            nan_cov_nn.append(i)
            logger.info(f"NAN {i}")

    if case_id == "synthetic":
        cov_nn[:, :, 28] = cov_nn[:, :, 27]
    elif case_id == "IAK34ss":
        for i in range(12):
            cov_nn[:, :, i] = cov_nn[:, :, 12]

    # 5. Pack the TF dicts with data values
    tf_dict[0].tf.data = matlab_tf[:, :, :11]
    tf_dict[0].cov_nn.data = cov_nn[:, :, :11]
    tf_dict[0].cov_ss_inv.data = cov_ss[:, :, :11]
    tf_dict[0].num_segments.data = n_data[:, :11]
    tf_dict[0].R2.data = R2[:, :11]

    tf_dict[1].tf.data = matlab_tf[:, :, 11:17]
    tf_dict[1].cov_nn.data = cov_nn[:, :, 11:17]
    tf_dict[1].cov_ss_inv.data = cov_ss[:, :, 11:17]
    tf_dict[1].num_segments.data = n_data[:, 11:17]
    tf_dict[1].R2.data = R2[:, 11:17]

    tf_dict[2].tf.data = matlab_tf[:, :, 17:23]
    tf_dict[2].cov_nn.data = cov_nn[:, :, 17:23]
    tf_dict[2].cov_ss_inv.data = cov_ss[:, :, 17:23]
    tf_dict[2].num_segments.data = n_data[:, 17:23]
    tf_dict[2].R2.data = R2[:, 17:23]

    tf_dict[3].tf.data = matlab_tf[:, :, 23:]
    tf_dict[3].cov_nn.data = cov_nn[:, :, 23:]
    tf_dict[3].cov_ss_inv.data = cov_ss[:, :, 23:]
    tf_dict[3].num_segments.data = n_data[:, 23:]
    tf_dict[3].R2.data = R2[:, 23:]

    # 6. Create TF Collection, and TFKernel
    tfc = TransferFunctionCollection(tf_dict=tf_dict, processing_config=p)
    kd = KernelDataset(local_station_id=local_station_id)
    survey_metadata = Survey()
    kd.survey_metadata["0"] = survey_metadata
    kd_df_dict = {
        "remote": [False],
        "station": [local_station_id],
        "processing_type": ["matlab EMTF"],
        "survey": ["0"],
        "run": ["a"],
        "start": ["1980-01-01T00:00:00"],
        "end": ["1980-01-02T00:00:00"],
    }
    kd_df = pd.DataFrame(data=kd_df_dict)
    kd.df = kd_df
    tfk = TransferFunctionKernel(dataset=kd, config=p)

    # 7. Create mt_metadata TF object and write z-file
    tf_obj = tfk.export_tf_collection(tfc)
    tf_obj.station_metadata.id = local_station_id
    tf_obj.station_metadata.runs[0].channels["ey"].measurement_azimuth = 90.0
    tf_obj.station_metadata.runs[0].channels["hy"].measurement_azimuth = 90.0
    tf_obj.write(z_file_path)

    # 7a. clip last few periods (matlab object and z-objects provided were not identical)
    if n_periods_clip:
        clip_bands_from_z_file(z_file_path, n_periods_clip, n_sensors=5)

    # 8. Compare z-file written by TF object to the archived version
    tf_obj_from_z_file = TF()
    tf_obj_from_z_file.from_zmm(z_file_path)
    archived_tf = TF()
    archived_tf.from_zmm(archived_z_file_path)
    assert np.isclose(
        tf_obj_from_z_file.transfer_function.data,
        archived_tf.transfer_function.data,
        rtol=1e-3,
    ).all()

    logger.info("success!")
