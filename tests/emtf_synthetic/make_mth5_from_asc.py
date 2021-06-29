# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:03:21 2021

@author: jpeacock
"""

import numpy as np
from pathlib import Path
from random import seed
import pandas as pd
from mth5.timeseries import ChannelTS, RunTS
from mth5.mth5 import MTH5

from filter_helpers import make_coefficient_filter
seed(0)
#<FILTERS>
ACTIVE_FILTERS = []
cf1 = make_coefficient_filter(name="1")
cf10 = make_coefficient_filter(gain=100, name="10")
UNITS = "SI"
ACTIVE_FILTERS = [cf1, cf10]
#</FILTERS>
#<GLOBAL CFG>
#make this an object? or leave as dict?
STATION_01_CFG = {}
STATION_01_CFG["ascii_data_path"] = Path(r"test1.asc")
STATION_01_CFG["columns"] = ["hx", "hy", "hz", "ex", "ey"]
STATION_01_CFG["noise_scalar"] = {}
for col in STATION_01_CFG["columns"]:
    STATION_01_CFG["noise_scalar"][col] = 0.0
STATION_01_CFG["filters"] = {}
for col in STATION_01_CFG["columns"]:
    STATION_01_CFG["filters"][col] = []
for col in STATION_01_CFG["columns"]:
    if col in ["ex", "ey"]:
        STATION_01_CFG["filters"][col] = [cf1.name,]
for col in STATION_01_CFG["columns"]:
    if col in ["hx", "hy", "hz"]:
        STATION_01_CFG["filters"][col] = [cf1.name,cf10.name]
STATION_01_CFG["run_id"] = "001"
STATION_01_CFG["station_id"] = "mt001"
STATION_01_CFG["sample_rate"] = 1.0

STATION_02_CFG = STATION_01_CFG.copy()
STATION_02_CFG["ascii_data_path"] = Path(r"test2.asc")
STATION_02_CFG["station_id"] = "mt002"



parent_data_path = STATION_01_CFG["ascii_data_path"].parent
SYNTHETIC_DATA = parent_data_path.joinpath("emtf_synthetic.h5")
#</GLOBAL CFG>



# # make filters:
# ACTIVE_FILTERS = []
# cf1 = make_coefficient_filter(name="1")
# cf10 = make_coefficient_filter(gain=100, name="10")
# UNITS = "SI"
# ACTIVE_FILTERS = [cf1, cf10]
# COLUMNS = ["hx", "hy", "hz", "ex", "ey"]
def create_mth5_synthetic_file(plot=False):
    df = pd.read_csv(STATION_01_CFG["ascii_data_path"],
                     names=STATION_01_CFG["columns"], sep="\s+")
    for col in STATION_01_CFG["columns"]:
        df[col] += STATION_01_CFG["noise_scalar"][col]*np.random.randn(len(df))

    # loop over stations and make them ChannelTS objects
    # we need to add a tag in the channels
    # so that when you call a run it will get all the filters with it.
    ch_list = []
    for col in df.columns:
        data = df[col].values
        # meta_dict = {"component": col,
        #              "sample_rate": SAMPLE_RATE,
        #              "filter.name":[cf1.name, cf10.name]}
        if col in ["ex", "ey"]:
            meta_dict = {"component": col,
                         "sample_rate": STATION_01_CFG["sample_rate"],
                         "filter.name": STATION_01_CFG["filters"][col],
                        }
            chts = ChannelTS(channel_type="electric", data=data,
                             channel_metadata=meta_dict)
            # add metadata to the channel here
            chts.channel_metadata.dipole_length = 50


        elif col in ["hx", "hy", "hz"]:
            meta_dict = {"component": col,
                         "sample_rate": STATION_01_CFG["sample_rate"],
                         "filter.name": STATION_01_CFG["filters"][col],
                        }
            chts = ChannelTS(channel_type="magnetic", data=data,
                             channel_metadata=meta_dict)

        ch_list.append(chts)

    # make a RunTS object
    runts = RunTS(array_list=ch_list)

    # add in metadata
    runts.station_metadata.id = STATION_01_CFG["station_id"]
    runts.run_metadata.id = STATION_01_CFG["run_id"]

    # plot the data
    if plot:
        runts.plot()

    #survey = Survey()

    # make an MTH5
    m = MTH5()
    m.open_mth5(SYNTHETIC_DATA, mode="w")
    station_group = m.add_station(STATION_01_CFG["station_id"])
    run_group = station_group.add_run(STATION_01_CFG["run_id"])
    run_group.from_runts(runts)
    #add filters
    for fltr in ACTIVE_FILTERS:
        cf_group = m.filters_group.add_filter(fltr)

    m.close_mth5()


def read_the_sythetic_mth5():
    import matplotlib.pyplot as plt
    from aurora.general_helper_functions import SANDBOX
    from aurora.time_series.windowing_scheme import WindowingScheme
    from aurora.time_series.frequency_band import FrequencyBands
    from aurora.time_series.frequency_band_helpers import extract_band
    from aurora.transfer_function.iter_control import IterControl
    from aurora.transfer_function.transfer_function_header import \
        TransferFunctionHeader
    from aurora.transfer_function.TTFZ import TTFZ
    from aurora.transfer_function.TRME import TRME
    from aurora.transfer_function.TRegression import RegressionEstimator

    #<CONFIG>
    BAND_SETUP = "EMTF"
    BAND_SETUP_FILE = SANDBOX.joinpath("bs_256.cfg")
    NUM_SAMPLES_WINDOW = 256
    MAX_NUMBER_OF_ITERATIONS = 10
    TF_LOCAL_SITE = "PKD      "  # This comes from mth5/mt_metadata aurora#18
    TF_REMOTE_SITE = None  # "SAO"   #This comes from mth5/mt_metadata aurora#18
    TF_PROCESSING_SCHEME = "RME"  # ""RME" #"OLS","RME", #required
    TF_INPUT_CHANNELS = ["hx", "hy"]  # optional, default ["hx", "hy"]
    TF_OUTPUT_CHANNELS = ["ex", "ey"]  # optional, default ["ex", "ey", "hz"]
    TF_REFERENCE_CHANNELS = None  # optional, default ["hx", "hy"],
    UNITS = "SI"
    SAMPLE_RATE = STATION_01_CFG["sample_rate"]
    #<CONFIG>
    m = MTH5()
    m.open_mth5(SYNTHETIC_DATA, mode="a")
    #m.survey_group.
    run_obj = m.get_run(STATION_01_CFG["station_id"],
                        STATION_01_CFG["run_id"])
    runts = run_obj.to_runts()
    ds = runts.dataset
    windowing_scheme = WindowingScheme(taper_family="hamming",
                                        num_samples_window=NUM_SAMPLES_WINDOW,
                                        num_samples_overlap=192,
                                        sampling_rate=runts.sample_rate)
    windowed_obj = windowing_scheme.apply_sliding_window(runts.dataset)
    print("windowed_obj", windowed_obj)
    tapered_obj = windowing_scheme.apply_taper(windowed_obj)
    print("tapered_obj", tapered_obj)
    stft_obj = windowing_scheme.apply_fft(tapered_obj)
    print("stft_obj", stft_obj)
    #<CALIBRATE>
    for channel_id in stft_obj.keys():
        mth5_channel = run_obj.get_channel(channel_id)
        channel_filter = mth5_channel.channel_response_filter
        calibration_response = channel_filter.complex_response(
        stft_obj.frequency.data)

    # if UNITS == "SI":
    #     if channel_id[0].lower() == 'h':
    #         calibration_response /= 1e-9  # SI Units
        stft_obj[channel_id].data /= calibration_response
    #</CALIBRATE>

    stft_obj_xrda = stft_obj.to_array("channel")
#     frequencies = stft_obj.frequency.data[1:]
#     # print(f"Lower Bound:{frequencies[0]}, Upper bound:{frequencies[-1]}")
#
    frequency_bands = FrequencyBands()
    if BAND_SETUP == "EMTF":
        frequency_bands.from_emtf_band_setup(filepath=BAND_SETUP_FILE,
            sampling_rate = SAMPLE_RATE,
            decimation_level = 1,
            num_samples_window = NUM_SAMPLES_WINDOW)
    elif BAND_SETUP == "XXXX":
        print("TODO:Write a method to choose lower and upper bounds, "
              "and number of bands to split it into")

    transfer_function_header = TransferFunctionHeader(
                    processing_scheme = TF_PROCESSING_SCHEME,
                    local_site = TF_LOCAL_SITE,
                    remote_site = TF_REMOTE_SITE,
                    input_channels = TF_INPUT_CHANNELS,
                    output_channels = TF_OUTPUT_CHANNELS,
                    reference_channels = TF_REFERENCE_CHANNELS)
    transfer_function_obj = TTFZ(transfer_function_header,
    frequency_bands.number_of_bands)

    for i_band in range(frequency_bands.number_of_bands):
        band = frequency_bands.band(i_band)
        band_dataarray = extract_band(band, stft_obj_xrda)
        band_dataset = band_dataarray.to_dataset("channel")
        X = band_dataset[TF_INPUT_CHANNELS]
        Y = band_dataset[TF_OUTPUT_CHANNELS]
        if TF_PROCESSING_SCHEME == "OLS":
            regression_estimator = RegressionEstimator(X=X, Y=Y)
            Z = regression_estimator.estimate_ols()
            print(f"{TF_PROCESSING_SCHEME}, \n {Z}")
        elif TF_PROCESSING_SCHEME=="RME":
            iter_control = IterControl(max_number_of_iterations=MAX_NUMBER_OF_ITERATIONS)
            regression_estimator = TRME(X=X, Y=Y, iter_control=iter_control)
            Z = regression_estimator.estimate()
            print(f"RME{TF_PROCESSING_SCHEME}, \n {Z}")
        else:
            print(f"processing_scheme {TF_PROCESSING_SCHEME} not supported")
            print(f"processing_scheme must be one of OLS, RME "
            f"not supported")
            raise Exception

        transfer_function_obj.set_tf(i_band, regression_estimator, band.center_period)

    transfer_function_obj.apparent_resistivity(units=UNITS)
    from aurora.transfer_function.rho_plot import RhoPlot

    plotter = RhoPlot(transfer_function_obj)
    fig, axs = plt.subplots(nrows=2)
    plotter.rho_sub_plot(axs[0])
    plotter.phase_sub_plot(axs[1])
    # plotter.rho_plot2()
    # plotter.phase_plot()
    plt.show()
    print(transfer_function_obj.rho.shape)
    print(transfer_function_obj.rho[0])
    print(transfer_function_obj.rho[-1])
    print("OK")



def main():
    create_mth5_synthetic_file()
    read_the_sythetic_mth5()

if __name__ == '__main__':
    main()

