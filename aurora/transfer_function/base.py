"""
follows Gary's TTF.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes

2021-07-02 Removed some prototype methods intended to edit specific station data
when many stations are being processed.  MMT methods to be addressed later.

2021-07-20: Addressing Issue #12.  If we are going to use xarray it is
tempting to use the frequency band centers as the axis for the arrays
here, rather than simple integer indexing.  This has the advantage of
making the data structures more explicit and self describing.  We can also
continue to use integer indices to assign and access tf values if needed.
However, one concern I have is that if we use floating point numbers for the
frequencies (or periods) we run the risk of machine roundoff error giving
problems down stream.  One way around this is to add a .band_centers()
methdd to FrequencyBands() which will provide is a list of band centers and
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
    Cov_SS : numpy array
        inverse signal power matrix.  How do we track the channels
        relationship? maybe xarray here as well
    Cov_NN : numpy array
        noise covariance matrix: see comment at Cov_SS above
    num_segments : integer array?
        Number of samples used to estimate TF for each band, and for each \
        output channel (might be different for different channels)
    R2 : numpy array
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
        print("TODO: change self.T to self.period")
        self.tf_header = tf_header
        self.frequency_bands = frequency_bands
        self.num_segments = None
        self.Cov_SS = None
        self.Cov_NN = None
        self.R2 = None
        self.initialized = False
        self.processing_config = kwargs.get("processing_config", None)
        self.num_segments
        if self.tf_header is not None:
            if self.num_bands is not None:
                self._initialize_arrays()

    @property
    def num_bands(self):
        #temporary function to allow access to old property num_bands used in
        # the matlab codes for initialization
        return self.frequency_bands.number_of_bands

    def _initialize_arrays(self):
        """
        There are four separate data strucutres, indexed by period here:

        TF (num_channels_out, num_channels_in)
        Cov_SS (num_channels_in, num_channels_in),
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
        if self.tf_header is not None:
            self.TF = np.zeros((self.num_channels_out, self.num_channels_in,
                               self.num_bands), dtype=np.complex128)
            # <make num_segments an xarray>

            self.num_segments = np.zeros((self.num_channels_out,self.num_bands))
            num_segments = np.zeros((self.num_channels_out, self.num_bands))
            num_segments_xr = xr.DataArray(num_segments,
                                           dims=["channel", "frequency"],
                               coords={"frequency":
                                           self.frequency_bands.band_centers(),
                                       "channel":
                                           self.tf_header.output_channels})
            self.num_segments = num_segments_xr
            # </make num_segments an xarray>

            self.Cov_SS = np.zeros((self.num_channels_in,
                                   self.num_channels_in, self.num_bands))
            self.Cov_NN = np.zeros((self.num_channels_out,
                                   self.num_channels_out, self.num_bands))
            self.R2 = np.zeros((self.num_channels_out, self.num_bands))
            self.initialized = True

    @property
    def T(self):
        return self.frequency_bands.band_centers(
            frequency_or_period="period")

    @property
    def periods(self):
        return self.T

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
        frequency_index = np.isclose(self.num_segments.frequency, frequency)
        frequency_index = np.where(frequency_index)[0][0]
        return frequency_index

    def period_index(self, period):
        return self.frequency_index(1./period)

    def tf_to_df(self):
        pass
        # import pandas as pd
        # columns = ["input_channel", "output_channel", "frequency", ]
        #            #"decimation_level"]
        # df = pd.DataFrame(columns=columns)

    def to_xarray(self, array="TF"):
        xra = xr.DataArray(self.TF,
                           dims=["output_channel", "input_channel", "period"],
                                       coords={"output_channel":
                                                   self.tf_header.output_channels,
                                               "input_channel":
                                                   self.tf_header.input_channels,
                                               "period": self.T})
        return xra

    def cov_ss_to_xarray(self):
        xra = xr.DataArray(self.Cov_SS,
                           dims=["input_channel_1", "input_channel_2",
                                 "period"],
                           coords={"input_channel_1":
                                       self.tf_header.input_channels,
                                   "input_channel_2":
                                       self.tf_header.input_channels,
                                   "period": self.T})
        return xra

    def cov_nn_to_xarray(self):
        xra = xr.DataArray(self.Cov_NN,
                           dims=["output_channel_1", "output_channel_2",
                                 "period"],
                           coords={"output_channel_1":
                                       self.tf_header.output_channels,
                                   "output_channel_2":
                                       self.tf_header.output_channels,
                                   "period": self.T})
        return xra

    def set_tf(self, regression_estimator, period):
        """
        This sets TF elements for one band, using contents of TRegression
        object.  This version assumes there are estimates for Nout output
        channels

        """
        index = self.period_index(period)
        if self.TF is None:
            print('Initialize TransferFunction obejct before calling setTF')
            raise Exception

        # use TregObj to fill in full impedance, error bars for a
        print("TODO: Convert the following commented check into python")
        print("although an exception will br raised anyhow actually")
        # if any(size(TRegObj.b)~=[obj.Nin obj.Nout])
        #     error('Regression object not consistent with declared dimensions of TF')
        #     raise Exception
        self.TF[:,:, index] = regression_estimator.b.T #check dims are consitent
        if regression_estimator.noise_covariance is not None:
            self.Cov_NN[:,:, index] = regression_estimator.noise_covariance
        if regression_estimator.inverse_signal_covariance is not None:
            self.Cov_SS[:,:, index] = regression_estimator.inverse_signal_covariance
        if regression_estimator.R2 is not None:
            self.R2[:, index] = regression_estimator.R2;
        # <assign with xarray>
        self.num_segments.data[:,index] = regression_estimator.n_data
        # </assign with xarray>
        #self.num_segments[:self.num_channels_out, index] =
        # regression_estimator.n_data
        return


    def standard_error(self):
        stderr = np.zeros(self.TF.shape)
        for j in range(self.num_channels_out):
            for k in range(self.num_channels_in):
                stderr[j, k,:] = np.sqrt(self.Cov_NN[j, j,:] * self.Cov_SS[k, k,:]);
        return stderr


    def from_emtf_zfile(self):
        pass

    def to_emtf_zfile(self):
        pass

def test_ttf():
    from aurora.transfer_function.transfer_function_header \
        import TransferFunctionHeader
    tfh = TransferFunctionHeader()
    ttf = TransferFunction(tfh, 32)


def main():
    test_ttf()

if __name__ == '__main__':
    main()
