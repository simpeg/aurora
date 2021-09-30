"""
One off method to help read in transfer function dumps provided by Gary from some of
the matlab tests.

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from aurora.config.decimation_level_config import DecimationLevelConfig
from aurora.general_helper_functions import BAND_SETUP_PATH
from aurora.sandbox.io_helpers.zfile_murphy import read_z_file
from aurora.time_series.frequency_band import FrequencyBands
from aurora.transfer_function.emtf_z_file_helpers import clip_bands_from_z_file
from aurora.transfer_function.emtf_z_file_helpers import get_default_orientation_block
from aurora.transfer_function.plot.rho_phi_helpers import plot_phi
from aurora.transfer_function.plot.rho_phi_helpers import plot_rho
from aurora.transfer_function.transfer_function_header import TransferFunctionHeader
from aurora.transfer_function.transfer_function_collection import (
    TransferFunctionCollection,
)
from aurora.transfer_function.TTFZ import TTFZ

CASE = "IAK34ss"  # synthetic"
bs_file = BAND_SETUP_PATH.joinpath("bs_256.cfg")
if CASE == "synthetic":
    n_periods_clip = 3  # for synthetic case
    z_mat = "TS1zss20210831.mat"
elif CASE == "IAK34ss":
    n_periods_clip = 3
    z_mat = "IAK34_struct_zss.mat"

orientation_strs = get_default_orientation_block()

frequency_bands = FrequencyBands()
sample_rate = 1.0
tf_dict = {}

for i_dec in range(4):
    frequency_bands = FrequencyBands()
    frequency_bands.from_emtf_band_setup(
        filepath=bs_file,
        sampling_rate=sample_rate,
        decimation_level=i_dec + 1,
        num_samples_window=256,
    )
    transfer_function_header = TransferFunctionHeader(
        processing_scheme="RME",
        local_station_id="test1",
        reference_station_id="",
        input_channels=["hx", "hy"],
        output_channels=[
            "hz",
            "ex",
            "ey",
        ],
        reference_channels=[],
    )
    tf_obj = TTFZ(transfer_function_header, frequency_bands)
    config = DecimationLevelConfig()
    config.sample_rate = sample_rate
    config.num_samples_window = 256
    tf_obj.processing_config = config
    tf_dict[i_dec] = tf_obj

    sample_rate /= 4.0

tmp = sio.loadmat(z_mat)
if CASE == "synthetic":
    stuff = tmp["temp"][0][0].tolist()
elif CASE == "IAK34ss":
    stuff = tmp["TFstruct"][0][0].tolist()
TF = stuff[4]
periods = stuff[5]
cov_ss = stuff[7]
cov_nn = stuff[8]
n_data = stuff[9]
R2 = stuff[10]
stderr = stuff[12]
freq = stuff[14]

nan_cov_nn = []
for i in range(len(periods)):
    if np.isnan(cov_nn[:, :, i]).any():
        nan_cov_nn.append(i)
        print(f"NAN {i}")


if CASE == "synthetic":
    cov_nn[:, :, 28] = cov_nn[:, :, 27]
elif CASE == "IAK34ss":
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

tfc = TransferFunctionCollection(header=tf_obj.tf_header, tf_dict=tf_dict)
z_file_path = "from_matlab.zss"
tfc.write_emtf_z_file(z_file_path, orientation_strs=orientation_strs)

if n_periods_clip:
    clip_bands_from_z_file(z_file_path, n_periods_clip, n_sensors=5)

zfile = read_z_file(z_file_path)

zfile.apparent_resistivity(angle=0)

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
    axs[1], zfile.periods, zfile.pxy, label="pxy", markersize=markersize, color="red"
)
plot_phi(
    axs[1], zfile.periods, zfile.pyx, label="pyx", markersize=markersize, color="blue"
)
axs[0].set_ylim(1, 1000)
axs[0].set_xlim(1, 10000)
plt.show()
print("success!")
