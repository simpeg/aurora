"""
Now that we have a big mth5 (from 01_make_cas04_mth5.py)
we can show examples of processing single and multistation.

Desired Examples:

1. Process Single Station
2. Process Single Station with runlist
3. Process Remote Reference
3. Process Remote Reference with runlist

Note 1: Functionality of RunSummary()
1. User can see all possible ways of processing the data
(possibly one list per station in run_summary)
2. User can get a list of local_station options
3. User can select local_station
-this can trigger a reduction of runs to only those that are from the local staion
and simultaneous runs at other stations
4. Given a local station, a list of possible reference stations can be generated
5. Given a remote reference station, a list of all relevent runs, truncated to
maximize coverage of the local station runs is generated
6. Given such a "restricted run list", runs can be dropped
7. Time interval endpoints can be changed


N.B. Rotations are not being applied when TFs are compared.
To get good comparison in the past, I manually changed the orientations and tilts
from:
number of channels   5   number of frequencies   25
 orientations and tilts of each channel
    1    13.20     0.00 CAS  Hx
    2   103.20     0.00 CAS  Hy
    3     0.00    90.00 CAS  Hz
    4    13.20     0.00 CAS  Ex
    5   103.20     0.00 CAS  Ey

To:
orientations and tilts of each channel
    1     0.00     0.00 CAS  Hx
    2    90.00     0.00 CAS  Hy
    3     0.00    90.00 CAS  Hz
    4     0.00     0.00 CAS  Ex
    5    90.00     0.00 CAS  Ey

The handling of rotations in general is not intended to be supported in aurora,
rather this will be done with the TF object in MTpy.
One can also use the angle argument in Ben Murphy's impedance and apparent_resistivity
functions, but these are slated for deprecation.
As a temporary workaround an arguments angle1, angle2 have been added to compare_two_z_files
In the following, we set angle2=13.0 to undo the difference between aurora and SPUD.

"""

import os
import pathlib
from collections import UserDict
from aurora.config.config_creator import ConfigCreator
from aurora.general_helper_functions import get_test_path
from aurora.pipelines.process_mth5 import process_mth5
from aurora.transfer_function.plot.comparison_plots import compare_two_z_files

from mtpy.processing import RunSummary, KernelDataset

from loguru import logger

TEST_PATH = get_test_path()
CAS04_PATH = TEST_PATH.joinpath("cas04")
CONFIG_PATH = CAS04_PATH.joinpath("config")
CONFIG_PATH.mkdir(exist_ok=True)
DATA_PATH = CAS04_PATH.joinpath("data")
H5_PATH = DATA_PATH.joinpath("8P_CAS04_CAV07_NVR11_REV06_v1.h5")
DEFAULT_EMTF_FILE = "emtf_results/CAS04-CAS04bcd_REV06-CAS04bcd_NVR08.zmm"
AURORA_RESULTS_PATH = CAS04_PATH.joinpath("aurora_results")
EMTF_RESULTS_PATH = CAS04_PATH.joinpath("emtf_results")
DEFAULT_EMTF_FILE = EMTF_RESULTS_PATH.joinpath(
    "CAS04-CAS04bcd_REV06-CAS04bcd_NVR08.zmm"
)
AURORA_RESULTS_PATH.mkdir(exist_ok=True)


class StationRuns(UserDict):
    """
    Class that can be populated by a run summary.  Just a wya to track groups of runs
    to process.
    """

    @property
    def label(self):
        station_run_strings = []
        for station_id, run_list in self.items():
            runs_str = "_".join(run_list)
            station_run_string = "__".join([station_id, runs_str])
            station_run_strings.append(station_run_string)
        out_str = "-".join(station_run_strings)
        return out_str

    def restrict_to_stations(self, station_ids):
        """
        Return an instance of self
        Parameters
        ----------
        station_ids: list

        Returns
        -------
        new: list
        """
        new = self.copy()
        [new.pop(k) for k in list(new.keys()) if k not in station_ids]
        return new

    @property
    def z_file_base(self):
        ext = "zss"
        if len(self.keys()) > 1:
            ext = "zrr"
        z_file_base = f"{self.label}.{ext}"
        return z_file_base

    def z_file_name(self, target_dir):
        if isinstance(target_dir, pathlib.Path):
            out_file = target_dir.joinpath(self.z_file_base)
        elif isinstance(target_dir, str):
            out_file = os.path.join(target_dir, self.z_file_base)
        return out_file


def process_station_runs(
    local_station_id, remote_station_id="", station_runs={}
):
    """

    Parameters
    ----------
    local_station_id: str
        The label of the station to process
    remote_station_id: str or None
    station_runs: StationRuns
        Dictionary keyed by station_id, values are run labels to process

    Returns
    -------
    tf_result: TransferFunctionCollection or mt_metadata.transfer_fucntions.TF
    """

    # identify the h5 files that you will use
    relevant_h5_list = [
        H5_PATH,
    ]

    # get a merged run summary from all h5_list
    run_summary = RunSummary()
    run_summary.from_mth5s(relevant_h5_list)

    # Pass the run_summary to a Dataset class
    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(
        run_summary, local_station_id, remote_station_id
    )

    # reduce station_runs_dict to only relevant stations

    relevant_stations = [
        local_station_id,
    ]
    if remote_station_id:
        relevant_stations.append(remote_station_id)
    if station_runs:
        tmp_station_runs = station_runs.restrict_to_stations(relevant_stations)
        kernel_dataset.select_station_runs(tmp_station_runs, "keep")

    logger.info(kernel_dataset.df)

    cc = ConfigCreator()
    pc = cc.create_from_kernel_dataset(kernel_dataset)
    z_file_name = tmp_station_runs.z_file_name(AURORA_RESULTS_PATH)
    tf_result = process_mth5(
        pc,
        kernel_dataset,
        show_plot=False,
        z_file_path=z_file_name,
    )
    xml_file_base = f"{tmp_station_runs.label}.xml"
    xml_file_name = AURORA_RESULTS_PATH.joinpath(xml_file_base)
    tf_result.write(fn=xml_file_name, file_type="emtfxml")
    return tf_result


def compare_results(
    z_file_name, label="aurora", emtf_file=DEFAULT_EMTF_FILE, out_file="aab.png"
):
    """

    Parameters
    ----------
    z_file_name: str or pathlib.Path
        Where is the tf info stored in a z-file
    label: str
        a tag that is put on the plot
    emtf_file: str or pathlib.Path
        Z-file from emtf to compare
    out_file: str or pathlib.Path
        png where plot is saved

    """
    compare_two_z_files(
        emtf_file,
        z_file_name,
        angle2=13.0,
        label1="emtf",
        label2=label,
        scale_factor1=1,
        out_file=out_file,
        markersize=3,
        # rho_ylims=[1e-20, 1e-6],
        # rho_ylims=[1e-8, 1e6],
        xlims=[1, 5000],
    )


def process_all_runs_individually(station_id="CAS04", reprocess=True):
    """

    Parameters
    ----------
    station_id: str
        The station to process (CAS04)

    """
    all_runs = ["a", "b", "c", "d"]
    for run_id in all_runs:
        station_runs = StationRuns()
        station_runs[station_id] = [
            run_id,
        ]
        if reprocess:
            process_station_runs(station_id, station_runs=station_runs)
        z_file_name = station_runs.z_file_name(AURORA_RESULTS_PATH)
        png_name = str(z_file_name).replace(".zss", ".png")
        compare_results(
            station_runs.z_file_name(AURORA_RESULTS_PATH), out_file=png_name
        )


def process_run_list(station_id, run_list, reprocess=True):
    """

    Parameters
    ----------
    station_id: str
        Name of the station to process (CAS04)
    run_list:
        list of runs to process drawn from [a,b,c,d]

    """
    station_runs = StationRuns()
    station_runs[station_id] = run_list
    if reprocess:
        process_station_runs(station_id, station_runs=station_runs)
    compare_results(station_runs.z_file_name(AURORA_RESULTS_PATH))


def process_with_remote(
    h5_paths, local, remote=None, band_setup_file="band_setup_emtf_nims.txt"
):
    """
    How this works:
    1. Make Run Summary
    2. Select station to process and remote
    3. Make a KernelDataset
    4. Slice KernelDataset to Simultaneos data
    5. (Optionally) Drop runs that are shorter than 15000s
    6. Make a config
    7. Process the data

    Parameters
    ----------
    local: str
    remote: str

    Returns
    -------

    """
    run_summary = RunSummary()
    run_summary.from_mth5s(h5_paths)
    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, local, remote)
    if remote:
        kernel_dataset.restrict_run_intervals_to_simultaneous()
    kernel_dataset.drop_runs_shorter_than(15000)

    # Add a method to ensure all samplintg rates are the same

    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(
        kernel_dataset, emtf_band_file=band_setup_file
    )
    for decimation in config.decimations:
        decimation.window.type = "hamming"
    show_plot = False
    if remote:
        z_file_base = f"{local}_RR{remote}.zrr"
    else:
        z_file_base = f"{local}.zss"
    z_file_path = AURORA_RESULTS_PATH.joinpath(z_file_base)
    tf_cls = process_mth5(
        config,
        kernel_dataset,
        units="MT",
        show_plot=show_plot,
        z_file_path=z_file_path,
    )
    logger.info(f"{tf_cls}")
    return


def compare_aurora_vs_emtf(local_station_id, remote_station_id, coh=False):
    """

    Parameters
    ----------
    local_station_id: str
        The label of the local station (its CAS04 in this case)
    remote_station_id: str
        The label of the remote station, one of CAV07, NVR11, REV06
    coh: bool
        Was cohernece sorting applied?

    Returns
    -------

    """
    if remote_station_id is None:
        emtf_file_base = "CAS04bcd_REV06.zrr"
        logger.warning(
            "Warning: No Single station EMTF results were provided for CAS04 by USGS"
        )
        logger.warning(f"Using {emtf_file_base}")
    else:
        emtf_file_base = f"{local_station_id}bcd_{remote_station_id}.zrr"
    emtf_file = EMTF_RESULTS_PATH.joinpath(emtf_file_base)
    aurora_label = f"{local_station_id} RR vs {remote_station_id}"
    if remote_station_id is None:
        z_file_base = f"{local_station_id}.zss"
    else:
        if coh:
            z_file_base = f"{local_station_id}_RR{remote_station_id}_coh.zrr"
        else:
            z_file_base = f"{local_station_id}_RR{remote_station_id}.zrr"
    aurora_z_file = AURORA_RESULTS_PATH.joinpath(z_file_base)
    out_png = str(aurora_z_file)
    out_png = out_png[:-3] + "png"

    compare_results(
        aurora_z_file, label=aurora_label, emtf_file=emtf_file, out_file=out_png
    )
    return


def old_main():
    h5_paths = [
        H5_PATH,
    ]
    # process_all_runs_individually()  # reprocess=False)
    # process_run_list("CAS04", ["b", "c", "d"])  # , reprocess=False)
    # process_with_remote(h5_paths, "CAS04", "CAV07")
    # process_with_remote(h5_paths, "CAS04", "NVR11", band_setup_file=BANDS_DEFAULT_FILE)
    process_with_remote(h5_paths, "CAS04", "REV06")

    # for RR in ["CAV07", "NVR11", "REV06"]:
    for RR in [
        "REV06",
    ]:
        compare_aurora_vs_emtf("CAS04", RR, coh=False)
        # compare_aurora_vs_emtf("CAS04", RR, coh=True)


def main():
    old_main()


#    process_all_runs_individually(reprocess=False)

# h5_paths = [H5_PATH,]
# RR = None# "REV06"
# process_with_remote(h5_paths, "CAS04", RR)
# compare_aurora_vs_emtf("CAS04", RR, coh=False)

if __name__ == "__main__":
    main()
