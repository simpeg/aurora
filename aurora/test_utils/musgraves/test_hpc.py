import pandas as pd

from aurora.config.config_creator import ConfigCreator
from aurora.config import BANDS_TEST_FAST_FILE
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.musgraves.helpers import get_musgraves_availability_df
from aurora.test_utils.musgraves.helpers import get_results_dir
from aurora.transfer_function.kernel_dataset import KernelDataset

USE_PANDARALEL = True
RESULTS_PATH = get_results_dir()

processing_df = pd.read_csv("/scratch/tq84/kk9397/musgraves/1_processing_list_with_workarounds.csv")
processing_df

# availability_df = get_musgraves_availability_df()
# processing_df = availability_df

processing_df = processing_df[0:2]



def enrich_row_with_processing(row):
    mth5_files = [row.path,]
    xml_file_path = RESULTS_PATH.joinpath(f"{row.station_id}.xml")
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
        kernel_dataset.from_run_summary(run_summary, row.station_id, None)
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
              
        z_file = str(xml_file_path).replace("xml", "zss")
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



def process_lots_of_mth5s(df):
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
    if USE_PANDARALEL:
        from pandarallel import pandarallel
        pandarallel.initialize(verbose=3)
        enriched_df = df.parallel_apply(enrich_row_with_processing, axis=1)
    else:
        enriched_df = df.apply(enrich_row_with_processing, axis=1)

    return enriched_df


enriched_df = process_lots_of_mth5s(processing_df)
print(enriched_df)

#enriched_df.error




