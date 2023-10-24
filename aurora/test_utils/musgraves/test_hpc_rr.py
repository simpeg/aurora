import argparse
import pathlib
import pandas as pd

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.musgraves.helpers import get_results_dir
from aurora.test_utils.musgraves.rr_mappings import station_combinations
from aurora.transfer_function.kernel_dataset import KernelDataset

USE_PANDARALEL = True
# RESULTS_PATH = get_results_dir()
RESULTS_PATH = pathlib.Path("/scratch/tq84/kk9397/musgraves/aurora_results/level_1/remote_reference")
L1_PATH = pathlib.Path("/g/data/my80/AuScope_MT_collection/AuScope_AusLAMP/Musgraves_APY/WA/level_1/Concatenated_Resampled_Rotated_Time_Series_MTH5")

def make_processing_df():
    from aurora.test_utils.musgraves.rr_mappings import station_combinations
    n_combos = len(station_combinations)
    df_dict = {"station_id":n_combos*[""], "remote_id":n_combos*[""], }
    for i,combo in enumerate(station_combinations):
        df_dict["station_id"][i] = combo[0]
        df_dict["remote_id"][i] = combo[1]
    df = pd.DataFrame(data=df_dict)
    return df


def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


def enrich_row_with_processing(row):
    station_path = L1_PATH.joinpath(f"{row.station_id}.h5")
    remote_path = L1_PATH.joinpath(f"{row.remote_id}.h5")
    mth5_files = [station_path, remote_path]
    #print("path", row.path, "type", type(row.path))
    #my_h5 = pathlib.Path(row.path)
    #print(f"{my_h5.exists()} my_h5.exists()")
    print(f"{station_path.exists()} station_path.exists()")
    print(f"{remote_path.exists()} remote_path.exists()")
    xml_file_path = RESULTS_PATH.joinpath(f"{row.station_id}_RR{row.remote_id}.xml")
    print(xml_file_path)
    # if xml_file_path.exists():
    #     if not self.force_reprocess:
    #         print("WARNING: Skipping processing as xml results alread exist")
    #         print("set force_reprocess True to avoid this")
    #         return row
    try:
        mth5_run_summary = RunSummary()
        mth5_run_summary.from_mth5s(mth5_files)
        run_summary = mth5_run_summary.clone()
        #run_summary.check_runs_are_valid(drop=True)
        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, row.station_id, row.remote_id)
        print("WORKAROUND")
        kernel_dataset.df["survey"] = "AusLAMP_Musgraves"
        kernel_dataset.df["run_id"] = "001"
        print("/WORKAROUND")
        
        
        print(kernel_dataset.df)
        
        # kernel_dataset.drop_runs_shorter_than(5000)
        # if len(kernel_dataset.df) == 0:
        #     print("No RR Coverage, casting to single station processing")
        #     kernel_dataset.from_run_summary(run_summary, row.station_id)

        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(kernel_dataset,)
#                                               emtf_band_file=BANDS_TEST_FAST_FILE)
        config.channel_nomenclature.keyword = "musgraves"
        config.set_default_input_output_channels()
        show_plot = False
              
        z_file = str(xml_file_path).replace("xml", "zrr")
        tf_cls = process_mth5(config,
                              kernel_dataset,
                              units="MT",
                              show_plot=show_plot,
                              z_file_path=z_file,
                              )
        tf_cls.write(fn=xml_file_path, file_type="emtfxml")
        # consider add xml_file_path.timestamp to columns??
    except Exception as e:
        row.exception = e.__class__.__name__
        row.error_message = e.args[0]
    return row



def process_lots_of_mth5s(df, use_pandarallel=False):
    """

    Parameters
    ----------
    df : pd.DataFrame
        This is a list of the files

    Returns
    -------
    df:  pd.DataFrame
        SAme as input but with new columns

    """
    df["exception"] = ""
    df["error"] = ""
    if use_pandarallel:
        from pandarallel import pandarallel
        pandarallel.initialize(verbose=3)
        enriched_df = df.parallel_apply(enrich_row_with_processing, axis=1)
    else:
        print("NO PANDARALEL")
        #return None
        enriched_df = df.apply(enrich_row_with_processing, axis=1)

    return enriched_df


def parse_args():
    """Argparse tutorial: https://docs.python.org/3/howto/argparse.html"""
    parser = argparse.ArgumentParser(description="Wide Scale Musgraves Test")
    parser.add_argument("--startrow", help="First row to process (zero-indexed)", type=int, default=0)
    parser.add_argument("--endrow", help="Last row to process (zero-indexed)", type=none_or_int, default=None,
                        nargs='?', )
    parser.add_argument("--use_pandarallel", help="Will use default pandarallel if True", type=bool,
                        default=False)
    args, unknown = parser.parse_known_args()


    print(f"startrow = {args.startrow}")
    print(f"endrow = {args.endrow}")
    print(f"use_pandarallel = {args.use_pandarallel}")
    return args

def main():
    processing_df = make_processing_df()
#    processing_df = pd.read_csv("/scratch/tq84/kk9397/musgraves/l1_processing_list_with_workarounds.csv")
    print(processing_df)

    #processing_df = processing_df[0:2]
    args = parse_args()
    enriched_df = process_lots_of_mth5s(processing_df, use_pandarallel=args.use_pandarallel)
    print(enriched_df)


if __name__ == "__main__":
    main()



