"""
This script iterates over all of the scraped XML from SPUD and registers information about success or failure of
ingest into a mt_metadata TF object.

There are two possible places to access an xml in each row, called emtf_xml_path and data_xml_path.


Dask Notes:
- 0 partitions 720s
- 1 partitions 682s
- 2 partitions 723s
- 4 partitions 882s
- 12 partitions 866s
- 32 partitions 857s

Not much difference.

Link where I originally got into dask apply with partitions: ____find this again__

But I am starting to suspect multiprocessing is the right solution..
https://stackoverflow.com/questions/67457956/how-to-parallelize-the-row-wise-pandas-dataframes-apply-method
but read this first:
https://examples.dask.org/applications/embarrassingly-parallel.html
"""



import datetime
import pandas as pd
import pathlib
import time

from aurora.test_utils.earthscope.helpers import SPUD_XML_PATHS
from aurora.test_utils.earthscope.helpers import load_xml_tf
from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import get_summary_table_schema_v2
from aurora.test_utils.earthscope.helpers import load_most_recent_summary


STAGE_ID = 1


# Config Params
XML_SOURCES = ["emtf", "data"]
N_PARTITIONS = 1

def define_dataframe_schema():
    """
    builds the csv defining column names, dtypes, and default values, and saves in standards/

    In this specific case, we start with the schema from the previous stage (0) and add columns

    Flow:
        - read previous CSV
        - augment with new columns
        - save with a new name
    """
    #read previous CSV
    from aurora.test_utils.earthscope.standards import SCHEMA_CSVS
    schema_csv = SCHEMA_CSVS[0]
    df = pd.read_csv(schema_csv)

    # augment with new columns
    for xml_source in XML_SOURCES:
        name = f"{xml_source}_error"
        dtype = "bool"
        default = 0
        df.loc[len(df)] = [name, dtype, default]

        name = f"{xml_source}_exception"
        dtype = "string"
        default = ""
        df.loc[len(df)] = [name, dtype, default]

        name = f"{xml_source}_error_message"
        dtype = "string"
        default = ""
        df.loc[len(df)] = [name, dtype, default]

        name = f"{xml_source}_processing_type" # was _remote_ref_type
        dtype = "string"
        default = ""
        df.loc[len(df)] = [name, dtype, default]
        print("DONT FORGET TO ADD HANDLING FOR THE CHANGE IN COLUMN NAME")
        print("LEGACY: _remote_ref_type, MODERN _processing_type")
        name = f"{xml_source}_remotes"
        dtype = "string"
        default = ""
        df.loc[len(df)] = [name, dtype, default]

    # save with a new name
    new_schema_csv = schema_csv.__str__().replace("00", "01")
    df.to_csv(new_schema_csv)


def prepare_dataframe():
    """
    Define the data structure that is output from this stage of processing
    It is basically the df from the prvious stage (0) with some new rows added.
    """
    spud_xml_csv = get_summary_table_filename(0)
    df = pd.read_csv(spud_xml_csv)

    schema = get_summary_table_schema_v2(1)
    for col in schema:
        default = col.default
        if col.dtype == "int64":
            default = int(default)
        if col.dtype == "bool":
            default = bool(int(default))
        if col.name not in df.columns:
            df[col.name] = default
            if col.dtype == "string":
                df[col.name] = ""

    return df

def enrich_row(row):
    """
    This will eventually get used by dask, but as a step we need to make this a method
    that works with df.apply()
    Returns:

    """
    for xml_source in XML_SOURCES:
        xml_path = SPUD_XML_PATHS[xml_source].joinpath(row[f"{xml_source}_xml_filebase"])
        try:
            tf = load_xml_tf(xml_path)
            remotes = tf.station_metadata.transfer_function.remote_references
            rr_type = tf.station_metadata.transfer_function.processing_type
            row[f"{xml_source}_processing_type"] = rr_type
            row[f"{xml_source}_remotes"] = ",".join(remotes)

        except Exception as e:
            row[f"{xml_source}_error"] = True
            row[f"{xml_source}_exception"] = e.__class__.__name__
            row[f"{xml_source}_error_message"] = e.args[0]
    return row


def batch_process(row_start=0, row_end=None):
    t0 = time.time()
    df = prepare_dataframe()
    if row_end is None:
        row_end = len(df)
    df = df[row_start:row_end]
    if not N_PARTITIONS:
        enriched_df = df.apply(enrich_row, axis=1)
    else:
        import dask.dataframe as dd
        ddf = dd.from_pandas(df, npartitions=N_PARTITIONS)
        n_rows = len(df)
        print(f"nrows ---> {n_rows}")
        # df_schema = get_summary_table_schema(STAGE_ID)
        # enriched_df = ddf.apply(enrich_row, axis=1, meta=df_schema).compute()
        schema = get_summary_table_schema_v2(STAGE_ID)
        meta = {x.name: x.dtype for x in schema}
        enriched_df = ddf.apply(enrich_row, axis=1, meta=meta).compute()

    results_csv = get_summary_table_filename(STAGE_ID)
    enriched_df.to_csv(results_csv, index=False)
    print(f"Took {time.time()-t0}s to review spud tfs, running with {N_PARTITIONS} partitions")
    return enriched_df


def summarize_errors():
    xml_sources = ["data", "emtf"]
    df = load_most_recent_summary(1)
    for xml_source in xml_sources:
        print(f"{xml_source} error \n {df[f'{xml_source}_error'].value_counts()}\n\n")

    print("OK")

def main():
    define_dataframe_schema()
    # normal
    # results_df = batch_process(row_end=1)
    results_df = batch_process()

    # run only data
    #results_df = review_spud_tfs(xml_sources = ["data_xml_path", ])

    summarize_errors()


    # # DEBUGGING
    df = load_most_recent_summary(1)
    # n_xml = len(df)
    # is_not_mda = df.data_xml_path.str.contains("__")
    # n_non_mda = is_not_mda.sum()
    # n_mda = len(df) - n_non_mda
    # print(f"There are {n_mda} / {n_xml} files with mda string ")
    # print(f"There are {n_non_mda} / {n_xml} files without mda string ")
    # non_mda_df = df[is_not_mda]
    print("summarize")

    

if __name__ == "__main__":
    main()
