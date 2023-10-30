# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26  2023

:copyright: 
    Karl Kappler (karl.kappler@berkeley.edu)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd

from mt_metadata.base.helpers import write_lines
from mt_metadata.base import get_schema, Base
#from .standards import SCHEMA_FN_PATHS
from aurora.test_utils.earthscope.standards import SCHEMA_FN_PATHS

# =============================================================================
attr_dict = get_schema("dataframe_column", SCHEMA_FN_PATHS)
# =============================================================================
class DataFrameColumn(Base):
    __doc__ = write_lines(attr_dict)

    def __init__(self, **kwargs):

        super().__init__(attr_dict=attr_dict, **kwargs)

def make_schema_list(stage_id):
    """

    # fix nans in blanks
    # fix_nans_in_columns = ["default", ]
    # for col in fix_nans_in_columns:
    #     df[col].replace(np.nan, "")
    #     df[col] = df[col].astype(str).replace("nan","")

        # row_json = row.to_json()
        # tmp.from_json(row_json)

    Parameters
    ----------
    stage_id

    Returns
    -------

    """
    schema_list = []
    from aurora.test_utils.earthscope.standards import SCHEMA_CSVS
    schema_csv = SCHEMA_CSVS[stage_id]
    df = pd.read_csv(schema_csv)
    for i, row in df.iterrows():
        tmp = DataFrameColumn()
        row_dict = row.to_dict()
        if row_dict["dtype"] == "int64":
            row_dict["default"] = int(row_dict["default"])
        if row_dict["dtype"] == "bool":
            row_dict["default"] = bool(int(row_dict["default"]))
        if row_dict["dtype"] == "float":
            row_dict["default"] = float(row_dict["default"])
        tmp.from_dict(row_dict)
        schema_list.append(tmp)
    return schema_list

def main():
    schema = make_schema_list(0)
    #
    print("OK")


if __name__ == "__main__":
    main()