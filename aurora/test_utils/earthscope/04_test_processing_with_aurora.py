"""
Daskification will involve handling of BlockingIOError
"""



import pandas as pd
import time

from aurora.test_utils.earthscope.helpers import DATA_PATH
from aurora.test_utils.earthscope.helpers import AURORA_TF_PATH
from aurora.test_utils.earthscope.helpers import AURORA_Z_PATH
from aurora.test_utils.earthscope.helpers import load_most_recent_summary
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import restrict_to_mda

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.pipelines.run_summary import RunSummary
from aurora.test_utils.earthscope.widescale import WidesScaleTest
from aurora.transfer_function.kernel_dataset import KernelDataset

from mth5.clients import FDSN, MakeMTH5
from mth5.helpers import close_open_files
from mth5.mth5 import MTH5

STAGE_ID = 4


class TestAuroraProcessing(WidesScaleTest):

    def __init__(self, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.use_skeleton = kwargs.get("use_skeleton", False)
        self.skeleton_file = f"skeleton_{str(STAGE_ID).zfill(2)}.csv"
        self.xml_source = kwargs.get("xml_source", "data")
        self.force_reprocess = kwargs.get("force_reprocess", False)


    def read_skeleton_with_dtype(self):
        """ Where should this go?  In general, I would like to read all schema with dtype"""
        df = pd.read_csv(self.skeleton_file, dtype=self.df_schema_dtypes)
        print("OK?")

    def prepare_jobs_dataframe(self):
        if self.use_skeleton:
            try:
                df = pd.read_csv(self.skeleton_file, dtype=self.df_schema_dtypes)
                return df
            except FileNotFoundError:
                print(f"File {self.skeleton_file} not found - will build it")



        spud_schema = get_summary_table_schema(1)
        dtypes = {x.name:x.dtype for x in spud_schema}
        spud_df = load_most_recent_summary(1, dtypes=dtypes)
        spud_df = restrict_to_mda(spud_df, RR="Robust Remote Reference")

        # try:
        #     processing_df = pd.read_csv(processing_summary_csv)
        # except FileNotFoundError:
        processing_df = pd.DataFrame(columns=self.df_column_names) # Doesn't support multidtype on init (appraetnly sas the docs)

        for i_row, row in spud_df.iterrows():
            print(row)  # station_id = row.station_id; network_id = row.network_id

            # Confirm XML is readable
            if row[f"{self.xml_source}_error"] is True:
                print(f"Skipping {row} for now, tf not reading in")
                continue

            # confirm we have h5 data
            # this could be maybe better checked by using table from 03, since h5 may exist but have no data
            data_file_base = f"{row.network_id}_{row.station_id}.h5"
            data_file = DATA_PATH.joinpath(data_file_base)
            if not data_file.exists():
                print(f"skipping proc {data_file} DNE")
                continue

            # check for a remote reference
            # probably don't need mth5_files variable in this preparation of dataframe
            remotes = row.data_remotes.split(",") # row[f"{xml_source}_remotes"].split(",")
            has_a_remote = False
            while remotes:
                if has_a_remote is False:
                    candidate_remote = remotes.pop(0)
                    remote_file_base = f"{row.network_id}_{candidate_remote}.h5"
                    rr_file = DATA_PATH.joinpath(remote_file_base)
                    if rr_file.exists():
                        remote_id = candidate_remote
                        mth5_files = [data_file, rr_file]
                        has_a_remote = True
                        remotes = []
                    else:
                        rr_file = None
                        remote_id = None
                        mth5_files = [data_file, ]

            xml_file_base = f"{row.network_id}_{row.station_id}_RR{remote_id}.xml"


            new_row = {
                "data_id": row.data_id,
                "network_id": row.network_id,
                "station_id": row.station_id,
                "aurora_xml_filebase": xml_file_base,
                "exception": "",
                "error_message": "",
                "data_xml_filebase": row.data_xml_filebase}
            if remote_id:
                new_row["remote_id"] = remote_id

            processing_df = processing_df.append(new_row, ignore_index=True)


        print(f"len(processing_df) {len(processing_df)}")
        # If uniquification is needed in future, uncomment the two lines below
        # subset = ['network_id', 'station_id', 'remote_id']
        # processing_df = processing_df.drop_duplicates(subset=subset, keep='first')
        # print(f"length after uniquification {len(ucpf)}")

        # Cast Dtypes:
        # for col in schema:
        #     if col.dtype == "string":
        #         processing_df[col.name] = processing_df[col.name].astype(str)

        processing_df.to_csv(self.skeleton_file, index=False)
        # reread and handle nan
        df = pd.read_csv(self.skeleton_file, dtype=self.df_schema_dtypes)
        return df



    def enrich_row(self, row):
        mth5_file_base = f"{row.network_id}_{row.station_id}.h5"
        data_file = DATA_PATH.joinpath(mth5_file_base)
        if pd.isna(row.remote_id):
            rr_file = None
            remote_id = None
        else:
            remote_file_base = f"{row.network_id}_{row.remote_id}.h5"
            rr_file = DATA_PATH.joinpath(remote_file_base)
            remote_id = row.remote_id
        mth5_files = [data_file, rr_file]
        mth5_files = [x for x in mth5_files if x is not None]

        xml_file_path = AURORA_TF_PATH.joinpath(row.aurora_xml_filebase)
        if xml_file_path.exists():
            if not self.force_reprocess:
                print("WARNING: Skipping processing as xml results alread exist")
                print("set force_reprocess True to avoid this")
                return row


        try:
            mth5_run_summary = RunSummary()
            mth5_run_summary.from_mth5s(mth5_files)
            run_summary = mth5_run_summary.clone()
            run_summary.check_runs_are_valid(drop=True)
            # run_summary.mini_summary

            kernel_dataset = KernelDataset()
            kernel_dataset.from_run_summary(run_summary, row.station_id, remote_id)
            #kernel_dataset.drop_runs_shorter_than(5000)
            if len(kernel_dataset.df) == 0:
                print("No RR Coverage, casting to single station processing")
                kernel_dataset.from_run_summary(run_summary, row.station_id)

            cc = ConfigCreator()
            config = cc.create_from_kernel_dataset(kernel_dataset)
            show_plot = False
            z_file = AURORA_Z_PATH.joinpath(row.aurora_xml_filebase.replace("xml", "zrr"))
            tf_cls = process_mth5(config,
                                  kernel_dataset,
                                  units="MT",
                                  show_plot=show_plot,
                                  z_file_path=z_file,
                                  )
            xml_file_path = AURORA_TF_PATH.joinpath(row.aurora_xml_filebase)
            tf_cls.write(fn=xml_file_path, file_type="emtfxml")
            # consider add xml_file_path.timestamp to columns??
        except Exception as e:
            row.exception = e.__class__.__name__
            row.error_message = e.args[0]
        print("check if the mth5 objects get closed when we leave this function ... ")
        # close_open_files() wont work under dask, because we can't just blanket close all h5 files,
        # since other, parallel processes will be interacting with these files.
        return row




def compare_graphics():
    from aurora.transfer_function.plot.comparison_plots import compare_two_z_files
    compare_two_z_files(
        "/home/kkappler/Downloads/ORF08bc_G8.zrr",
        "Z/EM_ORF08_RRORG08.zrr",
        angle2=0.0,
        label1="emtf",
        label2="aurora",
        scale_factor1=1,
        out_file="",
        markersize=3,
        rho_ylims=[1e0, 1e3],
        # rho_ylims=[1e-8, 1e6],
        xlims=[1, 5000],
    )


def main():
    t0 = time.time()
    tester = TestAuroraProcessing(stage_id=STAGE_ID,
                                  save_csv=True,
                                  use_skeleton=True,
                                  force_reprocess=True)
    #qq = tester.read_skeleton_with_dtype()
    # tester.startrow = 1306
    # tester.startrow = 1
    # tester.endrow = 1307
    #jdf = tester.prepare_jobs_dataframe()
    tester.run_test()

    total_time_elapsed = time.time() - t0
    print(f"Total scraping & review time {total_time_elapsed:.2f}s using {tester.n_partitions} partitions")

if __name__ == "__main__":
    main()
