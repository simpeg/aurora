import pandas as pd

from aurora.test_utils.earthscope.helpers import PUBLIC_DATA_AVAILABILITY_PATH


def load_data_availability_dfs():
    output = {}
    globby = PUBLIC_DATA_AVAILABILITY_PATH.glob("*txt")
    for txt_file in globby:
        print(txt_file)
        network_id = txt_file.name.split("_")[-1].split(".txt")[0]
        df = pd.read_csv(txt_file, parse_dates=['Earliest', 'Latest', ])
        output[network_id] = df
        print(f"loaded {network_id}")
    return output

class DataAvailabilityException(Exception):
    pass

class DataAvailability(object):
    def __init__(self):
        self.df_dict = load_data_availability_dfs()

    def get_available_channels(self, network_id, station_id):
        availability_df = self.df_dict[network_id]
        sub_availability_df = availability_df[availability_df["Station"] == station_id]
        availabile_channels = sub_availability_df['Channel'].unique()
        return availabile_channels

    def get_available_time_period(self, network_id, station_id, channel_id):
        """Note this can only work with an explicit channel_id, wildcards not supported"""
        availability_df = self.df_dict[network_id]
        cond1 = availability_df["Station"] == station_id
        cond2 = availability_df["Channel"] == channel_id
        sub_availability_df = availability_df[cond1 & cond2]
        earliest = sub_availability_df["Earliest"].min()
        latest = sub_availability_df["Latest"].max()
        interval = pd.Interval(earliest, latest)
        return interval

    def raise_exception(self, msg=""):
        raise DataAvailabilityException(msg)

def url_maker(net, sta, level="response"):
    """
    URL = "https://service.iris.edu/fdsnws/station/1/query?net=8P&sta=REU09&level=response&format=xml&includecomments=true&includeavailability=true&nodata=404"
    Parameters
    ----------
    net
    sta

    Returns
    -------

    """
    fdsn_URL = "http://service.iris.edu/fdsnws"
    url = f"{fdsn_URL}/station/1/query?net={net}&sta={sta}&level={level}&format=xml&includecomments=true&includeavailability=true&nodata=404"
#    url = f"https://service.iris.edu/fdsnws/station/1/query?net={net}&sta={sta}&level=response&format=xml&includecomments=true&includeavailability=true&nodata=404"
    return url
