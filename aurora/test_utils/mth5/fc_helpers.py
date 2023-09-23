import numpy as np
import pandas as pd


def read_fc_csv(csv_name, as_xarray=True):
    """
    Usage:
    xrds_obj = read_fc_csv(csv_name)
    df = read_fc_csv(csv_name, as_xarry=False)
    xrds =
    Returns a data
    Parameters
    ----------
    csv_name: str or pathlib.Path
    as_xarray: bool'

    Returns
    -------

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
        xrds_out = df.to_xarray()
        return xrds_out
    else:
        return df
