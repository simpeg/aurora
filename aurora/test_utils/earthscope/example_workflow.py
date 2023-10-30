# Required imports for the program. 
import xml.etree.ElementTree as ET
from html import unescape

from aurora.config import BANDS_DEFAULT_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary

from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
from mth5.mth5 import MTH5
from mth5.clients import FDSN, MakeMTH5
from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML

from matplotlib import pyplot as plt
from aurora.transfer_function.kernel_dataset import KernelDataset

import sys
import glob



## Set variables
print("Setting Variables")
doInitializeMTH5 = True
doOpenMTH5 = True
doReadSPUD = True
doGetData = True
doRunAurora = True
doAddSPUD = True
doComparison = True

for arg in sys.argv:
    if arg.lower().startswith('--network='):
        network = arg.split('=')[1]
    elif arg.lower().startswith('--station='):
        station = arg.split('=')[1]
    elif arg.lower().startswith('--initialize_mth5='):
        doInitializeMTH5 = arg.split('=')[1] == 'True'
    elif arg.lower().startswith('--open_mth5='):
        doOpenMTH5 = arg.split('=')[1]  == 'True'
    elif arg.lower().startswith('--add_spud='):
        doAddSPUD = arg.split('=')[1] == 'True'
    elif arg.lower().startswith('--run_aurora='):
        doRunAurora = arg.split('=')[1] == 'True'

    elif arg.lower().startswith('--only_spud'):
        doAddSPUD = True
        doInitializeMTH5 = False
        doOpenMTH5 = False
        doRunAurora = False
    


default_path = Path().cwd()

file_base = f'{network}_{station}'

spud_dir = '../spud_xml/'
spud_filenames = glob.glob(f'{spud_dir}{file_base}_*.xml')
if len(spud_filenames) > 1:
    print(f'{file_base} has more than one possible file: {spud_filenames}')
    quit()
else:
    try:
        spud_filename = spud_filenames[0]
    except:
        print(f"Could not find xml matching {spud_dir}{file_base}_*.xml")
        quit()

print(f'Using SPUD xml file: {spud_filename}')

# These dates are always essentially 'forever' - just the whole station history
startdate = '1970-01-01 00:00:00'
enddate = dt.datetime.now()




## Initialize a MakeMTH5 object
if doInitializeMTH5:
    print("Creating MTH5 object")
    mth5_object = MTH5(file_version="0.2.0")


if doReadSPUD:
    print("Adding TF from SPUD")
    print(f'-> Opening File: {spud_filename}')
    spud_tf = TF(spud_filename)
    spud_tf.read_tf_file()
    spud_tf.tf_id = f'{station}_spud'

    # # TODO: get remote reference station for processing Aurora
    # ## HOW TO KNOW IF REMOTE STATION IS SAME NETWORK?
    # remote_references = spud_tf.station_metadata.get_attr_from_name('transfer_function.remote_references')
    # remotes = list()
    # for remote_station in remote_references:
    #     if not len(remote_station.split('-')) > 1:
    #         if remote_station != station:
    #             remotes.append(remote_station)

    # print(remote_references)
    # print(remotes)





    
    


if doGetData:
    fdsn_object = FDSN(mth5_version='0.2.0')
    fdsn_object.client = "IRIS"

    ## Make the data inquiry as a DataFrame
    print(station)
    request_list = [[network, station, '', '*', startdate, enddate]]
    
    # try:
    #     for remote_station in remotes:
    #         request_list.append([network, remote_station, '', '*', startdate, enddate])

    # except:
    #     pass

    print(request_list)


    request_df =  pd.DataFrame(request_list, columns=fdsn_object.request_columns)

    print(f'Request List:\n{request_df}')

    # make_mth5_object = MakeMTH5(mth5_version='0.1.0', interact=False)
    # mth5_filename = make_mth5_object.from_fdsn_client(request_df, client="IRIS")

    print("Making mth5 from fdsn client")

    mth5_filename = fdsn_object.make_mth5_from_fdsn_client(request_df, interact=False)

    # mth5_filename = '8P_REV06_NVS12_CAV08_RET03_RER03.h5'
    # open file already created
    print(f"Opening mth5_object from {mth5_filename}")
    
    mth5_object.open_mth5(mth5_filename)





if doRunAurora:
    # # run aurora on the mth5_object
    print("Running AURORA")
    mth5_run_summary = RunSummary()
    h5_path = default_path.joinpath(mth5_filename)
    mth5_run_summary.from_mth5s([h5_path,])
    run_summary = mth5_run_summary.clone()
    print(run_summary.df)

    coverage_short_list_columns = ['station_id', 'run_id', 'start', 'end', ]
    kernel_dataset = KernelDataset()

    # print(type(remotes[0]), remotes[0])
    # kernel_dataset.from_run_summary(run_summary, station, remotes[0])
    kernel_dataset.from_run_summary(run_summary, station)

    kernel_dataset.drop_runs_shorter_than(15000)
    print(kernel_dataset.df[coverage_short_list_columns])

    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(kernel_dataset,   
                                        emtf_band_file=BANDS_DEFAULT_FILE,)

    for decimation in config.decimations:
        decimation.estimator.engine = "RME"

    show_plot = False
    aurora_tf = process_mth5(config,
                        kernel_dataset,
                        units="MT",
                        show_plot=show_plot,
                        z_file_path=None,
                    )

    aurora_tf.tf_id = f'{station}_aurora'
    print(f'Result of process_mth5:\n{aurora_tf}')
    tf_group_01 = mth5_object.add_transfer_function(aurora_tf)


if doAddSPUD:
    tf_group_02 = mth5_object.add_transfer_function(spud_tf)

    print(f'tf_group_01\n{tf_group_02}')

    est = tf_group_02.get_estimate("transfer_function")
    print(f'TEMP: est {est}')

    tf_group_02.has_estimate("covariance")
    print(f'TEMP: tf_group_01.has_estimate("covariance") {tf_group_02.has_estimate("covariance")}')


if doComparison:
    # Compare the two -first order, first attempt
    print((spud_tf.impedance - aurora_tf.impedance).std())

