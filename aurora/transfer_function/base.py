"""
follows Gary's TTF.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes

2021-07-20: Addressing Issue #12.  If we are going to use xarray it is
tempting to use the frequency band centers as the axis for the arrays
here, rather than simple integer indexing.  This has the advantage of
making the data structures more explicit and self describing.  We can also
continue to use integer indices to assign and access tf values if needed.
However, one concern I have is that if we use floating point numbers for the
frequencies (or periods) we run the risk of machine roundoff error giving
problems down stream.  One way around this is to add a .band_centers()
method to FrequencyBands() which will provide is a list of band centers and
then rather than access by the band center, we can use an access method that
gets us the frequencies between the band_egdes, which will be a unique frequency
... however, if we use overlapping bands this will in general get complicated.
However, for an MT TF representation, we do not in general use overlapping
bands.  A reasonably general, and simple solution is to make FrequencyBands
support an iterable of bands, accessable by integer position, and by center
frequency.
"""

import numpy as np
import xarray as xr


class TransferFunction(object):
    """
    Class to contain transfer function array.
    2021-07-21: adding processing_config so that it lives with this object.
    This will facilitate the writing of z-files if we use a band setup file
    that is not of EMTF style


    Parameters:
    TF : numpy array
        array of transfer functions: TF(Nout, Nin, Nperiods)
    T : numpy array
        list of periods
    Header : transfer_function_header.TransferFunctionHeader() object.
        TF header contains local site header, remote site header if
        appropriate, and information about estimation approach???
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

    properties (Dependent)
    StdErr % standard errors of TF components, same size and order as TF
    NBands
    freqs % inverse of period
    Nout
    Nin
    """

    def __init__(self, tf_header, frequency_bands, **kwargs):
        """
        change 2021-07-23 to require a frequency_bands object.  We may want
        to just pass the band_edges.  I'm not a fan of forcing dependency of
        TF on the FrequencyBands class, but it will simplify writting the z_file
        interfaces.  The old TTF.py class shows an example that just accepted
        num_bands as an initialation variable.

        Parameters
        ----------
        tf_header
        frequency_bands
        """
        self.tf_header = tf_header
        self.frequency_bands = frequency_bands
        self.num_segments = None
        self.cov_ss_inv = None
        self.cov_nn = None
        self.R2 = None
        self.initialized = False
        self.processing_config = kwargs.get("processing_config", None)
        self.num_segments
        if self.tf_header is not None:
            if self.num_bands is not None:
                self._initialize_arrays()

    @property
    def tf(self):
        return self.transfer_function

    @property
    def num_bands(self):
        """
        temporary function to allow access to old property num_bands used in
        the matlab codes for initialization
        Returns num_bands : int
            a count of the frequency bands associated with the TF
        -------

        """
        return self.frequency_bands.number_of_bands

    @property
    def periods(self):
        periods = self.frequency_bands.band_centers(frequency_or_period="period")
        periods = np.flipud(periods)
        return periods
        # return self.frequency_bands.band_centers(frequency_or_period="period")

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
                # "frequency": self.frequency_bands.band_centers(),
                "period": self.periods,
                "output_channel": self.tf_header.output_channels,
                "input_channel": self.tf_header.input_channels,
            },
        )
        # </transfer function xarray>

        # <num_segments xarray>
        num_segments = np.zeros((self.num_channels_out, self.num_bands), dtype=np.int32)
        num_segments_xr = xr.DataArray(
            num_segments,
            dims=["channel", "period"],  # "frequency"],
            coords={
                # "frequency": self.frequency_bands.band_centers(),
                "period": self.periods,
                "channel": self.tf_header.output_channels,
            },
        )
        self.num_segments = num_segments_xr
        # <num_segments xarray>

        # <Inverse signal covariance>
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
        # </Inverse signal covariance>

        # <Noise covariance>
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
        # </Noise covariance>

        # <Coefficient of determination>
        self.R2 = xr.DataArray(
            np.zeros((self.num_channels_out, self.num_bands)),
            dims=["output_channel", "period"],
            coords={
                "output_channel": self.tf_header.output_channels,
                "period": self.periods,
            },
        )
        # </Coefficient of determination>
        self.initialized = True

    @property
    def minimum_period(self):
        return np.min(self.periods)

    @property
    def maximum_period(self):
        return np.max(self.periods)

    @property
    def num_channels_in(self):
        return self.tf_header.num_input_channels

    @property
    def num_channels_out(self):
        return self.tf_header.num_output_channels

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
        This sets TF elements for one band, using contents of TRegression
        object.  This version assumes there are estimates for Nout output
        channels

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

    def standard_error(self):
        """
        TODO: make this a property that returns self._standard_error so it doesn't
        compute every time you call it.
        Returns
        -------

        """
        stderr = np.zeros(self.tf.data.shape)
        standard_error = xr.DataArray(
            stderr,
            dims=["output_channel", "input_channel", "period"],
            coords={
                "output_channel": self.tf_header.output_channels,
                "input_channel": self.tf_header.input_channels,
                "period": self.periods,
            },
        )
        for out_ch in self.tf_header.output_channels:
            for inp_ch in self.tf_header.input_channels:
                for T in self.periods:
                    cov_ss = self.cov_ss_inv.loc[inp_ch, inp_ch, T]
                    cov_nn = self.cov_nn.loc[out_ch, out_ch, T]
                    std_err = np.sqrt(np.abs(cov_ss * cov_nn))
                    standard_error.loc[out_ch, inp_ch, T] = std_err

        return standard_error

    def from_emtf_zfile(self):
        pass

    def to_emtf_zfile(self):
        pass
