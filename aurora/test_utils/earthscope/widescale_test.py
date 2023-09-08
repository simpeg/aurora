import argparse
import copy
import time

from aurora.test_utils.earthscope.helpers import get_summary_table_filename
from aurora.test_utils.earthscope.helpers import get_summary_table_schema
from aurora.test_utils.earthscope.helpers import get_summary_table_schema_v2


def none_or_str(value):
    if value == 'None':
        return None
    return value


DEFAULT_N_PARTITIONS = 1
class WidesScaleTest(object):

    def __init__(self, **kwargs):
        self.stage_id = kwargs.get("stage_id", None)
        self.jobs_df = None
        self.n_partitions = kwargs.get("n_partitions", DEFAULT_N_PARTITIONS)

    def prepare_jobs_dataframe(self):
        """ Makes the dataframe that will be populated/iterated over """
        pass

    def enrich_row(self, row):
        print("Enrich Row is not defined for Abstract Base Class")
        raise NotImplementedError

    def get_dataframe_schema(self):
        #df_schema = get_summary_table_schema(self.stage_id)
        df_schema = get_summary_table_schema_v2(self.stage_id)
        return df_schema

    @property
    def summary_table_filename(self):
        out_csv = get_summary_table_filename(self.stage_id)
        return out_csv

    def run_test(self, row_start=0, row_end=None):


        results_csv = self.summary_table_filename
        enriched_df.to_csv(results_csv, index=False)
        print(f"Took {time.time() - t0}s to review spud tfs, running with {N_PARTITIONS} partitions")
        return enriched_df

    def run_test(self, row_start=0, row_end=None, **kwargs):
        """ iterates over dataframe, enriching rows"""
        t0 = time.time()
        self.jobs_df = self.prepare_jobs_dataframe()
        df = copy.deepcopy(self.jobs_df)

        if row_end is None:
            row_end = len(df)
        df = df[row_start:row_end]
        n_rows = len(df)
        print(f"nrows ---> {n_rows}")

        if not self.n_partitions:
            enriched_df = df.apply(self.enrich_row, axis=1)
        else:
            import dask.dataframe as dd
            ddf = dd.from_pandas(df, npartitions=self.n_partitions)
            schema = get_summary_table_schema_v2(self.stage_id)
            meta = {x.name: x.dtype for x in schema}
            enriched_df = ddf.apply(self.enrich_row, axis=1, meta=meta).compute()

        results_csv = self.summary_table_filename
        enriched_df.to_csv(results_csv, index=False)
        print(f"Took {time.time() - t0}s to run STAGE {self.stage_id} with {self.n_partitions} partitions")
        return enriched_df

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

