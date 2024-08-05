"""
    This module contains functions to used by test that involve Fourier Coefficients in MTH5.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
import pathlib
import xarray as xr


def read_fc_csv(
    csv_name: Union[pathlib.Path, str], as_xarray: Optional[bool] = True
) -> Union[xr.Dataset, pd.DataFrame]:
    """

    Load Fourier coefficients from a csv file and return as xarray or dataframe

    Usage:
    xrds_obj = read_fc_csv(csv_name)
    df = read_fc_csv(csv_name, as_xarry=False)

    Parameters
    ----------
    csv_name: Union[pathlib.Path, str]
        Path to csv file to read
    as_xarray: Optional[bool]
        If true return xr.Dataset

    Returns
    -------
    output: xr.Dataset or pd.DataFrame
    """
    df = pd.read_csv(
        csv_name,
        index_col=[0, 1],
        parse_dates=[
            "time",
        ],
        skipinitialspace=True,
    )
    for col in df.columns:
        df[col] = np.complex128(df[col])
    if as_xarray:
        output = df.to_xarray()
    else:
        output = df
    return output
