import pandas as pd

from aurora.tf_kernel.dataset import DatasetDefinition

from mth5.utils.helpers import initialize_mth5

def extract_run_summaries_from_mth5s(mth5_paths, type="run", deduplicate=True):
    """
    ToDo: Move this method into mth5? or mth5_helpers?
    ToDo: Make this a class so that the __repr__ is a nice visual representation of the
    df, like what channel summary does in mth5

    Given a list of mth5s, this returns a dataframe of all available runs

    In order to drop duplicates I used the solution here:
    https://stackoverflow.com/questions/43855462/pandas-drop-duplicates-method-not-working-on-dataframe-containing-lists

    Parameters
    ----------
    mth5_paths: list
        paths or strings that point to mth5s
    type: string
        one of ["channel", "run"]
        "channel" returns concatenated channel summary, 
        "channel" returns concatenated run summary, 

    Returns
    -------
    super_summary: pd.DataFrame

    """
    dfs = len(mth5_paths) * [None]

    for i, mth5_path in enumerate(mth5_paths):
        mth5_obj = initialize_mth5(mth5_path, mode="a")
        ch_summary = mth5_obj.channel_summary
        if type == "run":
            dataset_definition = DatasetDefinition()
            dataset_definition.from_mth5_channel_summary(ch_summary)
            df = dataset_definition.df
            df["mth5_path"] = str(mth5_path)
        elif type == "channel":
            df = ch_summary.to_dataframe()
        mth5_obj.close_mth5()
        dfs[i] = df

    super_summary = pd.concat(dfs)
    super_summary.reset_index(drop=True, inplace=True)
    if deduplicate:
        keep_indices = super_summary.astype(str).drop_duplicates().index
        super_summary = super_summary.loc[keep_indices]
        super_summary.reset_index(drop=True, inplace=True)
    return super_summary