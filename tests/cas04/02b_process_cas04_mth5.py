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

number of channels   5   number of frequencies   25
 orientations and tilts of each channel
    1    13.20     0.00 CAS  Hx
    2   103.20     0.00 CAS  Hy
    3     0.00    90.00 CAS  Hz
    4    13.20     0.00 CAS  Ex
    5   103.20     0.00 CAS  Ey

orientations and tilts of each channel
    1   -13.20     0.00 CAS  Hx
    2    86.70     0.00 CAS  Hy
    3     0.00    90.00 CAS  Hz
    4   -13.20     0.00 CAS  Ex
    5    86.70     0.00 CAS  Ey

"""
# import matplotlib.pyplot as plt
from collections import UserDict
from aurora.config import BANDS_DEFAULT_FILE
from aurora.config.config_creator import ConfigCreator
from aurora.general_helper_functions import TEST_PATH
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.transfer_function.plot.comparison_plots import compare_two_z_files
from aurora.transfer_function.kernel_dataset import KernelDataset
from mth5.utils.helpers import initialize_mth5


CAS04_PATH = TEST_PATH.joinpath("cas04")
CONFIG_PATH = CAS04_PATH.joinpath("config")
CONFIG_PATH.mkdir(exist_ok=True)
DATA_PATH = CAS04_PATH.joinpath("data")
H5_PATH = DATA_PATH.joinpath("8P_CAS04_CAV07_NVR11_REV06.h5")


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
        station_ids

        Returns
        -------

        """
        new = self.copy()
        [new.pop(k) for k in list(new.keys()) if k not in station_ids]
        return new

    @property
    def z_file_name(self):
        ext = "ss"
        if len(self.keys()) > 1:
            ext = "zrr"
        z_file_name = f"{self.label}.{ext}"
        return z_file_name


def process_station_runs(
    local_station_id, remote_station_id=None, station_runs={}, return_collection=False
):

    # identify the h5 files that you will use
    relevant_h5_list = [
        H5_PATH,
    ]

    # get a merged run summary from all h5_list
    run_summary = RunSummary()
    run_summary.from_mth5s(relevant_h5_list)

    # Pass the run_summary to a Dataset class
    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, local_station_id, remote_station_id)

    # reduce station_runs_dict to only relevant stations

    relevant_stations = [
        local_station_id,
    ]
    if remote_station_id:
        relevant_stations.append(remote_station_id)
    tmp_station_runs = station_runs.restrict_to_stations(relevant_stations)
    kernel_dataset.select_station_runs(tmp_station_runs, "keep")
    print(kernel_dataset.df)

    cc = ConfigCreator()
    cc = ConfigCreator(config_path=CONFIG_PATH)
    pc = cc.create_run_processing_object(
        emtf_band_file=BANDS_DEFAULT_FILE, sample_rate=1.0
    )
    pc.stations.from_dataset_dataframe(kernel_dataset.df)
    pc.validate()
    z_file_name = tmp_station_runs.z_file_name
    tf_result = process_mth5(
        pc,
        kernel_dataset,
        show_plot=False,
        z_file_path=z_file_name,
        return_collection=return_collection,
    )
    if not return_collection:
        xml_file_name = f"{tmp_station_runs.label}.xml"
        tf_result.write_tf_file(fn=xml_file_name, file_type="emtfxml")
    return tf_result


def compare_results(z_file_name, label="aurora"):
    emtf_file = "emtf_results/CAS04-CAS04bcd_REV06-CAS04bcd_NVR08.zmm"
    compare_two_z_files(
        emtf_file,
        z_file_name,
        label1="emtf",
        label2=label,
        scale_factor1=1,
        out_file="aab.png",
        markersize=3,
        # rho_ylims=[1e-20, 1e-6],
        # rho_ylims=[1e-8, 1e6],
        xlims=[1, 5000],
    )


def process_all_runs_individually(station_id="CAS04"):
    all_runs = ["a", "b", "c", "d"]
    for run_id in all_runs:
        station_runs = StationRuns()
        station_runs[station_id] = [
            run_id,
        ]
        process_station_runs(station_id, station_runs=station_runs)
        compare_results(station_runs.z_file_name)


def get_channel_summary(h5_path):
    h5_path = DATA_PATH.joinpath("8P_CAS04_CAV07_NVR11_REV06.h5")
    mth5_obj = initialize_mth5(
        h5_path=h5_path,
    )
    mth5_obj.channel_summary.summarize()
    channel_summary_df = mth5_obj.channel_summary.to_dataframe()
    mth5_obj.close_mth5()
    print(channel_summary_df)
    return channel_summary_df


def get_run_summary(h5_path):
    """
    Use this method to take a look at what runs are available for processing

    Parameters
    ----------
    h5_path: str or pathlib.Path
        Target mth5 file

    Returns
    -------
    run_summary: aurora.pipelines.run_summary.RunSummary
        object that has the run summary
    """
    run_summary = RunSummary()
    run_summary.from_mth5s(
        [
            h5_path,
        ]
    )
    # print(run_summary.df)
    return run_summary


def process_with_remote(local, remote):
    h5_path = DATA_PATH.joinpath("8P_CAS04_CAV07_NVR11_REV06.h5")
    # channel_summary = get_channel_summary(h5_path)
    run_summary = get_run_summary(h5_path)
    kernel_dataset = KernelDataset()
    #    kernel_dataset.from_run_summary(run_summary, "CAS04")
    kernel_dataset.from_run_summary(run_summary, local, remote)
    kernel_dataset.restrict_run_intervals_to_simultaneous()
    kernel_dataset.drop_runs_shorter_than(15000)

    # Add a method to ensure all samplintg rates are the same
    sr = kernel_dataset.df.sample_rate.unique()

    cc = ConfigCreator()  # config_path=CONFIG_PATH)
    config = cc.create_run_processing_object(
        emtf_band_file=BANDS_DEFAULT_FILE, sample_rate=sr[0]
    )
    config.stations.from_dataset_dataframe(kernel_dataset.df)
    show_plot = False
    z_file_path = f"{local}_RR{remote}.zrr"
    tf_cls = process_mth5(
        config,
        kernel_dataset,
        units="MT",
        show_plot=show_plot,
        z_file_path=z_file_path,
        return_collection=False,
    )
    print(f"{tf_cls}")
    return


def main():
    process_all_runs_individually()
    import pdb

    pdb.set_trace()
    srl = StationRuns()
    srl["CAS04"] = [
        "b",
        "c",
    ]
    srl["NVR11"] = [
        "a",
        "c",
        "d",
    ]
    print(srl.label)
    srl2 = srl.restrict_to_stations(
        [
            "CAS04",
        ]
    )
    print(srl2.label)
    process_station_runs("CAS04", srl)

    # process_with_remote("CAS04", "CAV07")
    # TODO:
    #  1. Make Run Summary
    #  2. Drop runs that are shorter than X (1h?)
    #  3. Select CAS04 as station to process
    #  4. Give a list of reference stations
    #  5. Selected reference station,
    #  6. Generate a tfk_dataset obj that is "sliced" to the station-pair
    #  7. Define Reference station
    #  8 Create processing config
    # 9. Process see what we get ...
    # process_all_runs_individually()

    run_list = [
        "b",
        "c",
        "d",
    ]
    # process_runlist(run_list)
    srl = StationRuns()
    srl["CAS04"] = ["b", "c", "d"]
    aurora_label = f"{'_'.join(run_list)}-SS"
    compare_results(srl, label=aurora_label)
    aurora_label = "RR vs CAV07"
    compare_results("CAS04_RRCAV07.zrr", label=aurora_label)
    aurora_label = "RR vs CAV07 coh"
    compare_results("CAS04_RRCAV07_coh.zrr", label=aurora_label)
    print("OK")


if __name__ == "__main__":
    main()
