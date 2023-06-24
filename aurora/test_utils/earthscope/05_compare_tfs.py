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

from aurora.test_utils.earthscope.helpers import load_xml_tf
from aurora.test_utils.earthscope.helpers import load_most_recent_summary
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import restrict_to_mda


spud_df = load_most_recent_summary(1)
spud_df = restrict_to_mda(spud_df, RR="Robust Remote Reference")

STAGE_ID = 5


processing_summary_df = load_most_recent_summary(4)
processing_summary_df.exception[processing_summary_df.exception.isna()] = "" #dtype handling :/
tf_summary_csv = get_summary_table_filename(STAGE_ID)
print(tf_summary_csv)

def in_out_str(measurand, ch_in, ch_out):
    col_name = f"{measurand}_{ch_in}_{ch_out}"
    return col_name

def make_tf_report_schema():
    tf_report_schema = ["data_id", "station_id", "network_id", "remote_id",
                        "aurora_xml_path", "exception", "error_message", "data_xml_path", ]
    inputs = ['hx', 'hy']
    outputs = ['ex', 'ey', 'hz']
    for ch_in in inputs:
        for ch_out in outputs:
            col_name = in_out_str("delta", ch_in, ch_out)
            col_name = in_out_str("ratio", ch_in, ch_out)
            tf_report_schema.append(col_name)
    return  tf_report_schema

TF_REPORT_SCHEMA = make_tf_report_schema()

def initialize_tf_df():
    tf_report_schema = make_tf_report_schema()
    df = pd.DataFrame(columns=tf_report_schema)
    return df

def make_tf_row_dict():
    n_cols = len(TF_REPORT_SCHEMA)
    return dict(zip(TF_REPORT_SCHEMA, n_cols*["",]))



def batch_compare(xml_source="data_xml_path"):
    """


    """
    tf_df = initialize_tf_df()

    for i_row, row in processing_summary_df.iterrows():

        print(i_row)
        print(row)
        if bool(row.exception):
            print(f"SKIPPING EXCEPTION in processing {row.exception}")
            continue

        new_row = make_tf_row_dict()
        new_row["data_id"] = row["data_id"]
        new_row["network_id"] = row["network_id"]
        new_row["station_id"] = row["station_id"]
        new_row["remote_id"] = row["remote_id"]
        new_row["aurora_xml_path"] = row["filename"]

        spud_tf = load_xml_tf(row.data_xml_path)
        aurora_tf = load_xml_tf(row.filename)


        # Find Overlap of Periods where both TFs are defined
        print("TODO: Add some accounting here for how much is dropped from each")
        # Selecting glb and lub
        #lowest_freq = max(aurora_tf.frequency.min(), spud_tf.frequency.min())
        #highest_freq = min(aurora_tf.frequency.max(), spud_tf.frequency.max())
        shortest_period = max(aurora_tf.transfer_function.period.data.min(),
                              spud_tf.transfer_function.period.data.min())
        longest_period = min(aurora_tf.transfer_function.period.data.max(),
                              spud_tf.transfer_function.period.data.max())
        print(f"shortest_period {shortest_period}")
        print(f"longest_period {longest_period}")
        cond1 = spud_tf.transfer_function.period >= shortest_period
        cond2 = spud_tf.transfer_function.period <= longest_period
        reduced_spud_tf = spud_tf.transfer_function.where(cond1 & cond2, drop=True)
        cond1 = aurora_tf.transfer_function.period >= shortest_period
        cond2 = aurora_tf.transfer_function.period <= longest_period
        reduced_aurora_tf = aurora_tf.transfer_function.where(cond1 & cond2, drop=True)

        # TODO:
        print("should probably assert that the same input and output dimensions exist here")
        try:
            assert (spud_tf.transfer_function.input == aurora_tf.transfer_function.input).all()
            assert (spud_tf.transfer_function.output == aurora_tf.transfer_function.output).all()
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
                #input = "hx"
                #output = "ex"
                aurora_1d = interped_aurora.sel(input=input, output=output)
                spud_1d = reduced_spud_tf.sel(input=input, output=output)
                delta_along_dim = spud_1d.data - aurora_1d.data
                delta_along_dim = delta_along_dim[np.isnan(delta_along_dim)==False]
                delta_along_dim = np.abs(delta_along_dim)

                # THIS IS THE ANSWER
                delta = np.linalg.norm(delta_along_dim)
                col_name = in_out_str("delta", input, output)
                new_row[col_name] = delta

                ratio = spud_1d.data / aurora_1d.data
                ratio = ratio[np.isnan(ratio) == False]
                ratio = np.abs(ratio)
                ratio = np.linalg.norm(ratio)
                col_name = in_out_str("ratio", input, output)
                new_row[col_name] = ratio
        tf_df = tf_df.append(new_row, ignore_index=True)


        # except Exception as e:
        #     new_row["exception"] =  e.__class__.__name__
        #     new_row["exception"] =   e.args[0],

        if np.mod(i_row,10)==0:
           tf_df.to_csv(tf_summary_csv, index=False)


def main():
    batch_compare()
    df = pd.read_csv(tf_summary_csv)
    print("get some stats")
    delta_cols = [x for x in df.columns if "delta" in x]
    ratio_cols = [x for x in df.columns if "ratio" in x]
    for delta_col in delta_cols:
        fig,ax = plt.subplots()
        ax.hist(np.log10(df[delta_col]), 100)
        ax.set_title(f"Average difference between archvied SPUD and Aurora-computed TF \n {delta_col}")
        ax.set_xlabel("log_{10}(diff)")
        ax.set_ylabel(f"num_occurences / {len(df)}")
        plt.savefig(f"{delta_col}.png")
    for ratio_col in ratio_cols:
        fig, ax = plt.subplots()
        ax.hist(np.log10(df[ratio_col]), 100)
        ax.set_title(f"Average ratio between archvied SPUD and Aurora-computed TF \n {ratio_col}")
        ax.set_xlabel("log_{10}(diff)")
        ax.set_ylabel(f"num_occurences / {len(df)}")
        plt.savefig(f"{ratio_col}.png")
        #plt.show()
    print("DONE")

if __name__ == "__main__":
    main()