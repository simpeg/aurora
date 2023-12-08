"""
20230619
First attempt:
For every mda-containing data-xml in SPUD, we have tried to process and aurora TF.

We will output a summary of the difference between the two TF values

TFs are associated with input-output relationship, inout channels are normally
'hx' 'hy'
output are normally
'ex' 'ey' 'hz'

Thus there are normally 6 coefficients in the TF


And a summary table will of differences and ratios will be computed
"""


import numpy as np
import pandas as pd
import pathlib
import time

from matplotlib import pyplot as plt
from pathlib import Path

from aurora.test_utils.earthscope.helpers import AURORA_TF_PATH
from aurora.test_utils.earthscope.helpers import load_xml_tf
from aurora.test_utils.earthscope.helpers import load_most_recent_summary
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import restrict_to_mda
from aurora.test_utils.earthscope.helpers import SPUD_XML_PATHS
from aurora.test_utils.earthscope.widescale import WidesScaleTest


STAGE_ID = 5


def in_out_str(measurand, ch_in, ch_out):
    col_name = f"{measurand}_{ch_in}_{ch_out}"
    return col_name


def define_dataframe_schema():
    """
    builds the csv defining column names, dtypes, and default values, and saves in standards/

    In this specific case, we start with the schema from the previous stage (0) and add columns

    Flow:
        - read previous CSV
        - augment with new columns
        - save with a new name
    """
    # read previous CSV
    from aurora.test_utils.earthscope.standards import SCHEMA_CSVS

    schema_csv = SCHEMA_CSVS[4]
    df = pd.read_csv(schema_csv)

    # augment with new columns
    inputs = ["hx", "hy"]
    outputs = ["ex", "ey", "hz"]
    for ch_in in inputs:
        for ch_out in outputs:
            for comparison in [
                "delta",
                "ratio",
            ]:
                name = in_out_str(comparison, ch_in, ch_out)
                dtype = "float"
                default = 0
                df.loc[len(df)] = [name, dtype, default]

    # save with a new name
    print("OK")
    new_schema_csv = schema_csv.__str__().replace("04", "05")
    df.to_csv(new_schema_csv, index=False)


class TestCompareTFs(WidesScaleTest):
    def __init__(self, **kwargs):
        """ """
        super().__init__(**kwargs)
        # self.augment_with_existing = kwargs.get("augment_with_existing", True)
        self.skeleton_file = f"skeleton_{str(STAGE_ID).zfill(2)}.csv"

    def prepare_jobs_dataframe(self):

        stage_04_schema = get_summary_table_schema(4)
        dtypes = {x.name: x.dtype for x in stage_04_schema}
        stage_04_df = load_most_recent_summary(4, dtypes=dtypes)
        if "data_xml_path" in stage_04_df.columns:
            stage_04_df.drop(
                [
                    "data_xml_path",
                ],
                axis=1,
                inplace=True,
            )
        tf_report_df = pd.DataFrame(columns=self.df_column_names)

        for i_row, row in stage_04_df.iterrows():
            # print(row)  # station_id = row.station_id; network_id = row.network_id

            # check for exception
            if pd.isna(row.exception):
                pass
            else:
                print(f"Skipping {row} for now, Exceptiion detected ")
                continue

            # Otherwise, populate a row of jobs df
            for col in tf_report_df.columns:
                if col not in row.keys():
                    row[col] = 0.0
            tf_report_df = tf_report_df.append(row, ignore_index=True)
        # Finally cast dtypes on rows

        # save skeleton
        tf_report_df.to_csv(self.skeleton_file, index=False)
        # reread and handle nan
        df = pd.read_csv(self.skeleton_file, dtype=self.df_schema_dtypes)
        return df

    def enrich_row(self, row):
        spud_tf = load_xml_tf(SPUD_XML_PATHS["data"].joinpath(row.data_xml_filebase))
        aurora_tf = load_xml_tf(AURORA_TF_PATH.joinpath(row.aurora_xml_filebase))

        # Find Overlap of Periods where both TFs are defined
        print("TODO: Add some accounting here for how much is dropped from each")
        # Selecting glb and lub
        shortest_period = max(
            aurora_tf.transfer_function.period.data.min(),
            spud_tf.transfer_function.period.data.min(),
        )
        longest_period = min(
            aurora_tf.transfer_function.period.data.max(),
            spud_tf.transfer_function.period.data.max(),
        )
        print(f"shortest_period {shortest_period}")
        print(f"longest_period {longest_period}")
        cond1 = spud_tf.transfer_function.period >= shortest_period
        cond2 = spud_tf.transfer_function.period <= longest_period
        reduced_spud_tf = spud_tf.transfer_function.where(cond1 & cond2, drop=True)
        cond1 = aurora_tf.transfer_function.period >= shortest_period
        cond2 = aurora_tf.transfer_function.period <= longest_period
        reduced_aurora_tf = aurora_tf.transfer_function.where(cond1 & cond2, drop=True)

        # TODO:
        print(
            "should probably assert that the same input and output dimensions exist here"
        )
        try:
            assert (
                spud_tf.transfer_function.input == aurora_tf.transfer_function.input
            ).all()
            assert (
                spud_tf.transfer_function.output == aurora_tf.transfer_function.output
            ).all()
        except AssertionError:
            print(spud_tf.transfer_function.input)
            print(aurora_tf.transfer_function.input)
            print(spud_tf.transfer_function.output)
            print(aurora_tf.transfer_function.output)
            print("???")
        print("LOOKS OK")
        # Now we have arrays that share a range (mostly)
        # try interpolating, in this case we interp aurora onto spud freqs
        interped_aurora = reduced_aurora_tf.interp(period=reduced_spud_tf.period)

        for input in spud_tf.transfer_function.input.data:
            for output in spud_tf.transfer_function.output.data:
                # reduce to 1D at a time
                # input = "hx"
                # output = "ex"
                aurora_1d = interped_aurora.sel(input=input, output=output)
                spud_1d = reduced_spud_tf.sel(input=input, output=output)
                delta_along_dim = spud_1d.data - aurora_1d.data
                delta_along_dim = delta_along_dim[np.isnan(delta_along_dim) is False]
                delta_along_dim = np.abs(delta_along_dim)

                # THIS IS THE ANSWER
                delta = np.linalg.norm(delta_along_dim)
                col_name = in_out_str("delta", input, output)
                row[col_name] = delta

                ratio = spud_1d.data / aurora_1d.data
                ratio = ratio[np.isnan(ratio) is False]
                ratio = np.abs(ratio)
                # ratio = np.linalg.norm(ratio)/np.sqrt(len(ratio))
                ratio = np.median(ratio)
                col_name = in_out_str("ratio", input, output)
                row[col_name] = ratio
        return row


def review_results(df):
    print("get some stats")
    delta_cols = [x for x in df.columns if "delta" in x]
    ratio_cols = [x for x in df.columns if "ratio" in x]
    for delta_col in delta_cols:
        fig, ax = plt.subplots()
        ax.hist(np.log10(df[delta_col]), 100)
        ax.set_title(
            f"Average difference between archived SPUD and Aurora-computed TF \n {delta_col}"
        )
        ax.set_xlabel("log_{10}(diff)")
        ax.set_ylabel(f"num_occurences / {len(df)}")
        plt.savefig(f"{delta_col}.png")
    for ratio_col in ratio_cols:
        fig, ax = plt.subplots()
        ax.hist(np.log10(df[ratio_col]), 100)
        ax.set_title(
            f"Average ratio of SPUD and Aurora TFs: \n {ratio_col.split('_')[1:]}"
        )
        ax.set_xlabel("log_{10}(ratio)")
        ax.set_ylabel(f"# occurrences / ({len(df)} total)")
        ax.set_xlim(-2, 2)
        plt.savefig(f"{ratio_col}.png")
        # plt.show()
    print("DONE")


def main():
    df = pd.read_csv(
        "/home/kkappler/.cache/earthscope/summary_tables/05_tf_comparison_review.csv"
    )
    review_results(df)

    # define_dataframe_schema()
    t0 = time.time()
    # tester = TestCompareTFs(stage_id=STAGE_ID,
    #                           save_csv=True,
    #                         use_skeleton=True)

    # tester.startrow = 1679
    # tester.endrow = 1680
    # df = tester.run_test()
    # review_results(df)
    total_time_elapsed = time.time() - t0
    print(f"Total time elapsed {total_time_elapsed}")


if __name__ == "__main__":
    main()
