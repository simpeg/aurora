"""
follows Gary's TFHeader.m
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes
"""


class TransferFunctionHeader(object):
    """
    class for storing metadata for a TF estimate

    This class should inherit the metadata from the remote and reference
    stations. As of 2021-07-20 the class functions with only the station_id
    and channel_id values.

    <See Issue #41>
    </See Issue #41>

    """

    def __init__(self, **kwargs):
        """
        Parameters
        local_station : mt_metadata.transfer_functions.tf.station.Station()
            Station metadata object for the station to be estimated (
            location, channel_azimuths, etc.)
        remote_site: same object type as local site
            if no remote reference then this can be None
        output_channels: list
            Probably a list of channel keys -- usually ["ex","ey","hz"]
        input_channels : list
            Probably a list of channel keys -- usually ["hx","hy"]
            These are the channels being provided as input to the regression.
        reference_channels : list
            These are the channels being used from the RR station. This is a
            channel list -- usually [?, ?]
        processing_scheme: str
            Denotes the regression engine used to estimate the transfer
            function.  One of "OLS" or "RME", "RME_RR.  Future
            versions could include , "multivariate array", "multiple remote",
            etc.

        """
        self.processing_scheme = kwargs.get("processing_scheme", None)
        self._local_station = kwargs.get("local_station", None)
        self._reference_station = kwargs.get("reference_station", None)
        self.input_channels = kwargs.get("input_channels", ["hx", "hy"])
        self.output_channels = kwargs.get("output_channels", ["ex", "ey"])
        self.reference_channels = kwargs.get("reference_channels", [])
        self.user_meta_data = None  # placeholder for anything

        # <ByPass mt_metadata classes>
        self._local_station_id = kwargs.get("local_station_id", None)
        self._remote_station_id = kwargs.get("remote_station_id", None)
        # </ByPass mt_metadata classes>

    @property
    def local_station_id(self):
        try:
            station_id = self.local_station.id
        except AttributeError:
            station_id = self._local_station_id
        return station_id

    @property
    def remote_station_id(self):
        try:
            station_id = self.remote_station.id
        except AttributeError:
            station_id = self._remote_station_id
        return station_id

    @property
    def local_station(self):
        return self._local_station

    @property
    def num_input_channels(self):
        return len(self.input_channels)

    @property
    def num_output_channels(self):
        return len(self.output_channels)

    @property
    def local_channels(self):
        return self.input_channels + self.output_channels
