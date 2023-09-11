import pandas as pd

from aurora.sandbox.mth5_helpers import build_request_df
from aurora.test_utils.earthscope.helpers import DATA_AVAILABILITY_PATHS


def make_data_availability_txts():
    """Exceutes the main code from get_MT_numbers.ipynb"""
    import time
    include_restricted = False
    if include_restricted:
        public_or_restricted = "restricted"
    else:
        public_or_restricted = "public"
    out_folder = DATA_AVAILABILITY_PATHS[public_or_restricted]
    print(out_folder.absolute())
    filebase = "mt_availability"
    outfile = out_folder.joinpath(f"{filebase}.txt")
    fdsn_URL = "http://service.iris.edu/fdsnws"
    channels = ['?FE', '?FN', '?FZ', '?F1', '?F2', '?QE', '?QN', '?QZ', '?Q1', '?Q2']
    channels = ','.join(channels)

    with open(outfile, 'w') as f:
        f.write("net.sta,chan,hours")

    sta_URL = f"{fdsn_URL}/station/1/query?cha={channels}&level=channel&format=text&includecomments=true&nodata=204"
    print(sta_URL)

    try:
        network_df = pd.read_csv(sta_URL, sep='|')
    except Exception as e:
        print(f"ERROR with station service {sta_URL}")
        print(f"ERROR: {e}")
        quit()

    network_df
    networks = network_df['#Network '].unique()
    print(f"Identified {len(networks)} unique networks: \n {networks}")
    # grouped = stations.groupby(by=['#Network ', ' Station '])
    # grouped = network_df.groupby(by='#Network ')

    for network in networks:
        print(network)
        netfile = f'{filebase}_{network}.txt'
        netfile = out_folder.joinpath(netfile)

        av_URL = f"{fdsn_URL}/availability/1/query?format=text&net={network}&cha={channels}&orderby=nslc_time_quality_samplerate&includerestricted={include_restricted}&nodata=204"

        print(av_URL)
        try:
            avail = pd.read_csv(av_URL, sep=" ")
        except Exception as e:
            print(f"ERROR with availability service {av_URL} ")
            print(f"ERROR: {e}")
            with open(outfile, 'a') as f:
                f.write(f"\n#ERROR with {network}")
            time.sleep(2)
            continue

        avail.columns = avail.columns.str.strip()
        avail['Latest'] = pd.to_datetime(avail['Latest'], format="%Y-%m-%dT%H:%M:%S.%f")
        avail['Earliest'] = pd.to_datetime(avail['Earliest'], format="%Y-%m-%dT%H:%M:%S.%f")
        avail['Span'] = avail.Latest - avail.Earliest

        avail.to_csv(netfile, index=False)

        grouped_chan = avail.groupby(by=['Station', 'Channel'])
        for name, group in grouped_chan:
            station = name[0]
            channel = name[1]
            total_time = group['Span'].sum()
            with open(outfile, 'a') as f:
                f.write(f"\n{network}.{station},{channel},{'%.2f' % (total_time / pd.Timedelta(hours=1))}")
        #                 f.write(f"\n{network}.{station},{channel},{total_time}")

        time.sleep(5)
        print("DONE")

def load_data_availability_dfs(public_or_restricted="public"):
    data_availability_path = DATA_AVAILABILITY_PATHS[public_or_restricted]
    output = {}
    globby = data_availability_path.glob("mt_availability_*txt")
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


def row_to_request_df(row, data_availability_obj, verbosity=1, use_channel_wildcards=False,
                      raise_exception_if_data_availability_empty=True):
    """

    Parameters
    ----------
    row: pandas.core.series.Series
        Row of a custom dataframe used in widescale earthscope tests.
        The only information we currently take from this row is the network_id and station_id
    data_availability: This is an instance of DataAvailability object.
        the data_availability object is a global varaible in 02 and 03, and so I anticipate there
        could be issues running this in parallel ...
        This could be handled in future with webcalls
        Also, if I passed instead row.network_id, row.station_id, I could just pass
        the
    verbosity: int
        Print request df to screen, to be deprecated
    use_channel_wildcards: bool
        If True look for ["*Q*", "*F*", ]

    Returns
    -------

    """
    time_period_dict = {}
    network_id = row.network_id
    station_id = row.station_id
    if use_channel_wildcards:
        availabile_channels = ["*Q*", "*F*", ]
    else:
        availabile_channels = data_availability_obj.get_available_channels(network_id, station_id)
        for ch in availabile_channels:
            tp = data_availability_obj.get_available_time_period(network_id, station_id, ch)
            time_period_dict[ch] = tp

    if len(availabile_channels) == 0:
        if raise_exception_if_data_availability_empty:
            msg = f"No data from {network_id}_{station_id}"
            raise DataAvailabilityException(msg)
        else:
            print("Setting channels to wildcards because local data_availabilty query returned empty list")
            availabile_channels = ["*Q*", "*F*", ]

    request_df = build_request_df(network_id, station_id,
                                  channels=availabile_channels,
                                  start=None, end=None,
                                  time_period_dict=time_period_dict)
    if verbosity > 1:
        print(f"request_df: \n {request_df}")
    return request_df


def main():
    make_data_availability_txts()

if __name__ == "__main__":
    main()