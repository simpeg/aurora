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
from aurora.test_utils.earthscope.helpers import load_most_recent_summary
from aurora.test_utils.earthscope.standards import SCHEMA_CSVS
from aurora.test_utils.earthscope.widescale_test import WidesScaleTest



STAGE_ID = 1


# Config Params
DEFAULT_XML_SOURCES = ["emtf", "data"]

def define_dataframe_schema(xml_sources=DEFAULT_XML_SOURCES):
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
    for xml_source in xml_sources:
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
        name = f"{xml_source}_remotes"
        dtype = "string"
        default = ""
        df.loc[len(df)] = [name, dtype, default]

    # save with a new name
    new_schema_csv = schema_csv.__str__().replace("00", "01")
    df.to_csv(new_schema_csv)

class TestLoadSPUDTFs(WidesScaleTest):

    def __init__(self, **kwargs):
        """

        """
        #super(WidesScaleTest, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.xml_sources = kwargs.get("xml_sources", DEFAULT_XML_SOURCES)


    def prepare_jobs_dataframe(self):
        """
        Define the data structure that is output from this stage of processing
        It is basically the df from the prvious stage (0) with some new rows added.
        """
        spud_xml_csv = get_summary_table_filename(0)
        df = pd.read_csv(spud_xml_csv)

        schema = self.get_dataframe_schema()
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

    def enrich_row(self, row):
        """
        This will eventually get used by dask, but as a step we need to make this a method
        that works with df.apply()
        Returns:

        """
        for xml_source in self.xml_sources:
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




def summarize_errors(xml_sources=DEFAULT_XML_SOURCES):
    df = load_most_recent_summary(STAGE_ID)
    for xml_source in xml_sources:
        print(f"{xml_source} error \n {df[f'{xml_source}_error'].value_counts()}\n\n")

    n_xml = len(df)
    is_not_mda = df.data_xml_filebase.str.contains("__")
    n_non_mda = is_not_mda.sum()
    n_mda = len(df) - n_non_mda
    print(f"There are {n_mda} / {n_xml} files with mda string ")
    print(f"There are {n_non_mda} / {n_xml} files without mda string ")
    # non_mda_df = df[is_not_mda]
    return


def main():
    schema_csv = SCHEMA_CSVS[STAGE_ID]
    if not schema_csv.exists():
        define_dataframe_schema()

    # normal
    tester = TestLoadSPUDTFs(stage_id=STAGE_ID)
    # tester.endrow = 5
    tester.run_test()

    summarize_errors()
    # run only data
    # tester = TestLoadSPUDTFs(stage_id=STAGE_ID, xml_sources=["data",])
    # tester.run_test()
    # summarize_errors()

    return
    

if __name__ == "__main__":
    main()
