"""
follows Gary's TTF.m in iris_mt_scratch
egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes
"""

import numpy as np
import xarray as xr

from mt_metadata.base import Base


class TransferFunction(Base):
    """Class to contain transfer function array.
    

    Parameters
    ----------
    TF : numpy array
        array of transfer functions: TF(Nout, Nin, Nperiods)
    T : numpy array
        list of periods
    cov_ss_inv : numpy array
        inverse signal power matrix.  aka Cov_SS in EMTF matlab codes
    cov_nn : numpy array
        noise covariance matrix: aka Cov_NN in EMTF matlab codes
    num_segments : integer array?
        Number of samples used to estimate TF for each band, and for each \
        output channel (might be different for different channels)
    R2 : xarray.DataArray
        multiple coherence for each output channel / band
    FullCov : boolean
        true if full covariance is provided

    Returns
    -------
    properties (Dependent)
    StdErr % standard errors of TF components, same size and order as TF
    NBands
    freqs % inverse of period
    Nout
    Nin
    """

    def __init__(self, decimation_level_id, frequency_bands, **kwargs):
        """
        change 2021-07-23 to require a frequency_bands object.  We may want
        to just pass the band_edges.  I'm not a fan of forcing dependency of
        TF on the FrequencyBands class, but it will simplify writting the z_file
        interfaces.  The old TTF.py class shows an example that just accepted
        num_bands as an initialation variable.

        Parameters
        ----------
        _emtf_header : legacy header information used by Egbert's matlab class.  Header contains
        local site header, remote site header if appropriate, and information about estimation approach
        decimation_level_id: int
            Identifies the relevant decimation level.  Used for accessing the
            appropriate info in self.processing config.
        frequency_bands: aurora.time_series.frequency_band.FrequencyBands
            frequency bands object
        """
        self._emtf_tf_header = None
        self.decimation_level_id = decimation_level_id
        self.frequency_bands = frequency_bands
        self.num_segments = None
        self.cov_ss_inv = None
        self.cov_nn = None
        self.R2 = None
        self.initialized = False
        self.processing_config = kwargs.get("processing_config", None)

        if self.emtf_tf_header is not None:
            if self.num_bands is not None:
                self._initialize_arrays()

    @property
    def emtf_tf_header(self):
        if self.processing_config is None:
            print("No header is available without a processing config")
            self._emtf_tf_header = None
        else:
            if self._emtf_tf_header is None:
                tfh = self.processing_config.emtf_tf_header(self.decimation_level_id)
                self._emtf_tf_header = tfh
        return self._emtf_tf_header

    @property
    def tf_header(self):
        return self.emtf_tf_header

    @property
    def tf(self):
        return self.transfer_function

    @property
    def num_bands(self):
        """_summary_

        Returns
        -------
        num_bands: int
            a count of the frequency bands associated with the TF
        """
        return self.frequency_bands.number_of_bands

    @property
    def periods(self):
        periods = self.frequency_bands.band_centers(frequency_or_period="period")
        periods = np.flipud(periods)
        return periods

    def _initialize_arrays(self):
        """
        There are four separate data strucutres, indexed by period here:

        TF (num_channels_out, num_channels_in)
        cov_ss_inv (num_channels_in, num_channels_in, num_bands),
        Cov_NN (num_channels_out, num_channels_out),
        R2 num_channels_out)

        We use frequency in Fourier domain and period in TF domain.  Might
        be better to be consistent.  It would be inexpensive to duplicate
        the axis and simply provide both.

        Each of these is indexed by period (or frequency)
        TODO: These may be better cast as xarrays.  Review after curves up
        and running

        TODO: also, I prefer np.full(dim_tuple, np.nan) for init
        Returns
        -------

        """
        if self.tf_header is None:
            print("header needed to allocate transfer function arrays")
            raise Exception

        # <transfer function xarray>
        tf_array_dims = (self.num_channels_out, self.num_channels_in, self.num_bands)
        tf_array = np.zeros(tf_array_dims, dtype=np.complex128)
        self.transfer_function = xr.DataArray(
            tf_array,
            dims=["output_channel", "input_channel", "period"],  # frequency"],
            coords={
                "period": self.periods,
                "output_channel": self.tf_header.output_channels,
                "input_channel": self.tf_header.input_channels,
            },
        )

        # num_segments xarray
        num_segments = np.zeros((self.num_channels_out, self.num_bands), dtype=np.int32)
        num_segments_xr = xr.DataArray(
            num_segments,
            dims=["channel", "period"],
            coords={
                "period": self.periods,
                "channel": self.tf_header.output_channels,
            },
        )
        self.num_segments = num_segments_xr

        # Inverse signal covariance
        cov_ss_dims = (self.num_channels_in, self.num_channels_in, self.num_bands)
        cov_ss_inv = np.zeros(cov_ss_dims, dtype=np.complex128)
        self.cov_ss_inv = xr.DataArray(
            cov_ss_inv,
            dims=["input_channel_1", "input_channel_2", "period"],
            coords={
                "input_channel_1": self.tf_header.input_channels,
                "input_channel_2": self.tf_header.input_channels,
                "period": self.periods,
            },
        )

        # Noise covariance
        cov_nn_dims = (self.num_channels_out, self.num_channels_out, self.num_bands)
        cov_nn = np.zeros(cov_nn_dims, dtype=np.complex128)
        self.cov_nn = xr.DataArray(
            cov_nn,
            dims=["output_channel_1", "output_channel_2", "period"],
            coords={
                "output_channel_1": self.tf_header.output_channels,
                "output_channel_2": self.tf_header.output_channels,
                "period": self.periods,
            },
        )

        # Coefficient of determination
        self.R2 = xr.DataArray(
            np.zeros((self.num_channels_out, self.num_bands)),
            dims=["output_channel", "period"],
            coords={
                "output_channel": self.tf_header.output_channels,
                "period": self.periods,
            },
        )

        self.initialized = True

    @property
    def minimum_period(self):
        return np.min(self.periods)

    @property
    def maximum_period(self):
        return np.max(self.periods)

    @property
    def num_channels_in(self):
        return len(self.tf_header.input_channels)

    @property
    def num_channels_out(self):
        return len(self.tf_header.output_channels)

    def frequency_index(self, frequency):
        return self.period_index(1.0 / frequency)
        # frequency_index = np.isclose(self.num_segments.frequency, frequency)
        # frequency_index = np.where(frequency_index)[0][0]
        # return frequency_index

    def period_index(self, period):
        period_index = np.isclose(self.num_segments.period, period)
        period_index = np.where(period_index)[0][0]
        return period_index
        # return self.frequency_index(1.0 / period)

    def tf_to_df(self):
        pass
        # import pandas as pd
        # columns = ["input_channel", "output_channel", "frequency", ]
        #            #"decimation_level"]
        # df = pd.DataFrame(columns=columns)

    def set_tf(self, regression_estimator, period):
        """
        This sets TF elements for one band, using contents of regression_estimator
        object.  This version assumes there are estimates for Nout output channels
        """
        index = self.period_index(period)

        tf = regression_estimator.b_to_xarray()
        output_channels = list(regression_estimator._Y.data_vars)
        input_channels = list(regression_estimator._X.data_vars)

        for out_ch in output_channels:
            for inp_ch in input_channels:
                self.tf[:, :, index].loc[out_ch, inp_ch] = tf.loc[out_ch, inp_ch]

        if regression_estimator.noise_covariance is not None:
            for out_ch_1 in output_channels:
                for out_ch_2 in output_channels:
                    tmp = regression_estimator.cov_nn.loc[out_ch_1, out_ch_2]
                    self.cov_nn[:, :, index].loc[out_ch_1, out_ch_2] = tmp

        if regression_estimator.inverse_signal_covariance is not None:
            for inp_ch_1 in input_channels:
                for inp_ch_2 in input_channels:
                    tmp = regression_estimator.cov_ss_inv.loc[inp_ch_1, inp_ch_2]
                    self.cov_ss_inv[:, :, index].loc[inp_ch_1, inp_ch_2] = tmp

        if regression_estimator.R2 is not None:
            for out_ch in output_channels:
                tmp = regression_estimator.R2.loc[out_ch]
                self.R2[:, index].loc[out_ch] = tmp

        self.num_segments.data[:, index] = regression_estimator.n_data

        return
