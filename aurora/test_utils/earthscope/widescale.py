"""
Notes on parallelization:

In the approach in this module, a dataframe is used to keep track of the tasks to do.  Each row corresponds to a
processing task.  The specific processing task is defined in the extension of the abstract base class WideScaleTest
defined here.
The row of the df is modified during the processing.  It turns out that this is a constraint
(I didn't know that this would be restrictive).  It seems that modifying the df in-place makes the function not callable
 via standard multiprocessing (or the pathos fork).  The three ways this can execute are:
  df.apply()            # pandas (slow)
  ddf.apply()           # dask data frame (slow and I don't understand why)
  df.parallel_apply()   # pandarallel (fast)

It is possible that I am doing something wrong however when I tried multiprocessing.

In any case, the recommended usage is pandarallel for now.


Link where I originally got into dask apply with partitions: ____find this again__

But I am starting to suspect multiprocessing is the right solution..
https://stackoverflow.com/questions/67457956/how-to-parallelize-the-row-wise-pandas-dataframes-apply-method
but read this first:
https://examples.dask.org/applications/embarrassingly-parallel.html

2023-10-07
https://stackoverflow.com/questions/26784164/pandas-multiprocessing-apply suggests using pandarallel
This seems to give decent performance improvement (https://github.com/nalepae/pandarallel)

Also, we are cautioned away from .apply() method in both pandas and dask here:
https://stackoverflow.com/questions/31361721/python-dask-dataframe-support-for-trivially-parallelizable-row-apply
"""
import argparse
import copy
#import multiprocessing as mp
from pandarallel import pandarallel
import time

from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema


def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

DEFAULT_N_PARTITIONS = 1
class WidesScaleTest(object):

    def __init__(self, **kwargs):
        self.parse_args()
        self.stage_id = kwargs.get("stage_id", None)
        self.stage_name = kwargs.get("stage_name", None)
        self.jobs_df = None
        self.save_csv = kwargs.get("save_csv", True)
        self._df_schema = None
        # self.add_timestamp_to_csv_filename = kwargs.get("add_timestamp_to_csv_filename", False)


    def prepare_jobs_dataframe(self):
        """ Makes the dataframe that will be populated/iterated over """
        print("ERROR: prepare_jobs_dataframe is not defined for Abstract Base Class")
        raise NotImplementedError

    def enrich_row(self, row):
        """
        Operation perfomed by df.apply()
        Will eventually get used by dask
        """
        print("ERROR: Enrich Row is not defined for Abstract Base Class")
        raise NotImplementedError

    @property
    def df_schema(self):
        if self._df_schema is None:
            self._df_schema = get_summary_table_schema(self.stage_id)
        return self._df_schema

    @property
    def df_schema_dtypes(self):
        return {x.name:x.dtype for x in self.df_schema}

    @property
    def df_column_names(self):
        return [x.name for x in self.df_schema]

    @property
    def summary_table_filename(self):
        out_csv = get_summary_table_filename(self.stage_id)
        return out_csv

    def run_test(self):#, row_start=0, row_end=None, **kwargs):
        """ iterates over dataframe, enriching rows"""
        t0 = time.time()
        self.jobs_df = self.prepare_jobs_dataframe()
        df = copy.deepcopy(self.jobs_df)
        info_str = f"There are {len(df)} rows in jobs dataframe"
        if self.endrow is None:
            self.endrow = len(df)
        if (self.startrow != 0) or (self.endrow != len(df)):
            info_str += f"\n restricting df to df[{self.startrow}:{self.endrow}] for testing"
        df = df[self.startrow:self.endrow]
        info_str += f"\n Processing {len(df)} rows"
        print(info_str)

        if not self.n_partitions:
            enriched_df = df.apply(self.enrich_row, axis=1)
        elif self._use_dask:
            import dask.dataframe as dd
            ddf = dd.from_pandas(df, npartitions=self.n_partitions)
            schema = get_summary_table_schema(self.stage_id)
            meta = {x.name: x.dtype for x in schema}
            enriched_df = ddf.apply(self.enrich_row, axis=1, meta=meta).compute()
        elif self._use_pandarallel:
            from pandarallel import pandarallel
            pandarallel.initialize(verbose=3)
            enriched_df = df.parallel_apply(self.enrich_row, axis=1)
        else:
            msg = "WARNING No dask, or pandrarallel specified --- using normal df.apply()"
            print(msg)
            enriched_df = df.apply(self.enrich_row, axis=1)


        if self.save_csv:
            results_csv = self.summary_table_filename
            print(f"Saving results to {results_csv}")
            enriched_df.to_csv(results_csv, index=False)
        print(f"Took {time.time() - t0}s to run STAGE {self.stage_id} with {self.n_partitions} partitions")
        return enriched_df

    def parse_args(self):
        """Argparse tutorial: https://docs.python.org/3/howto/argparse.html"""
        parser = argparse.ArgumentParser(description="Wide Scale Earthscpe Test")
        parser.add_argument("--npart", help="how many partitions to use (triggers dask dataframe if > 0", type=int,
                            default=DEFAULT_N_PARTITIONS)
        parser.add_argument("--startrow", help="First row to process (zero-indexed)", type=int, default=0)
        # parser.add_argument('category', type=none_or_str, nargs='?', default=None,
        # 					help='the category of the stuff')
        parser.add_argument("--endrow", help="Last row to process (zero-indexed)", type=none_or_int, default=None,
                            nargs='?', )
        parser.add_argument("--use_pandarallel", help="Will use default pandarallel if True", type=bool,
                            default=False)
        parser.add_argument("--use_dask", help="Will use dask if True, but you will need to define --npart", type=bool,
                            default=False)
        args, unknown = parser.parse_known_args()


        print(f"npartitions = {args.npart}")
        self.n_partitions = args.npart
        print(f"startrow = {args.startrow} {type(args.startrow)}")
        self.startrow = args.startrow
        print(f"endrow = {args.endrow} {type(args.endrow)}")
        self.endrow = args.endrow
        print(f"use_dask = {args.use_dask} {type(args.use_dask)}")
        self._use_dask = args.use_dask
        print(f"use_pandarallel = {args.use_pandarallel} {type(args.use_pandarallel)}")
        self._use_pandarallel = args.use_pandarallel

        if isinstance(args.endrow, str):
            args.endrow = int(args.endrow)

        return args

    def report(self):
        pass


#
# def main(self):
#     parser = argparse.ArgumentParser(description='Do some cool stuff.')
#
#     parser.add_argument('category', type=none_or_str, nargs='?', default=None,
#                         help='the category of the stuff')
#
#     args = parser.parse_args()
#
#     print(args.category)
#     print(type(args.category))
#
#
# if __name__ == "__main__":
#     main()

