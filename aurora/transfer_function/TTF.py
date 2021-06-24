"""
follows Gary's TTF.m in
iris_mt_scratch/egbert_codes-20210121T193218Z-001/egbert_codes/matlabPrototype_10-13-20/TF/classes

"""

import numpy as np
import xarray as xr


class TTF(object):
    """
    Class to contain transfer function array.

    Supports full covariance, arbitrary number of input / output
    channels (but for MT  # input channels = Nin is always 2!)

    Instantiated with no arguments or 2. When there are 2, these must be:
    TTF(NBands,Header).

    Example:
    Zxx = Z(1, 1, Period)
    Zxy = Z(1, 2, Period)
    Zyx = Z(2, 1, Period)
    Zyy = Z(2, 2, Period)

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
    def __init__(self, tf_header, num_bands):
        print("TODO: change self.T to self.period")
        self.tf_header = tf_header
        self.num_bands = num_bands #it would be nice if this was a property
        # that depended on some other attr of the processing scheme... surely
        # this number is already known-- it comes from the data being processed
        self.T = None # replace with periods
        self.num_segments = None
        #self.periods = None
        self.Cov_SS = None
        self.Cov_NN = None
        self.R2 = None
        self.initialized = False
        if self.tf_header is not None:
            if self.num_bands is not None:
                self._initialize_arrays()



    def _initialize_arrays(self):
        """
        TODO: These may be better cast as xarrays.  Review after curves up
        and running

        TODO: also, I prefer np.full(dim_tuple, np.nan) for init
        Returns
        -------

        """
        if self.tf_header is not None:
            self.T =  np.zeros(self.num_bands)
            self.TF = np.zeros((self.num_channels_out, self.num_channels_in,
                               self.num_bands), dtype=np.complex128)
            self.num_segments = np.zeros((self.num_channels_out,
                                          self.num_bands))
            self.Cov_SS = np.zeros((self.num_channels_in,
                                   self.num_channels_in, self.num_bands))
            self.Cov_NN = np.zeros((self.num_channels_out,
                                   self.num_channels_out, self.num_bands))
            self.R2 = np.zeros((self.num_channels_out, self.num_bands))
            self.initialized = True

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


    def set_tf(self, i_band, regression_estimator, T):
        """
        This sets TF elements for one band, using contents of TRegression
        object.  This version assumes there are estimates for Nout output
        channels
        TODO: can we get away from the integer i_band index and obtain the
        integer from a band_object? Please
        TODO: i_band and T are not independent and not both needed here. A
        BandAveragingScheme() class will
        """
        if self.TF is None:
            print('Initialize TTF obect before calling setTF')
            raise Exception

        n_data = regression_estimator.n_data #use the class
        # use TregObj to fill in full impedance, error bars for a
        print("TODO: Convert the following commented check into python")
        print("although an exception will br raised anyhow actually")
        # if any(size(TRegObj.b)~=[obj.Nin obj.Nout])
        #     error('Regression object not consistent with declared dimensions of TF')
        #     raise Exception
        self.T[i_band] = T;
        self.TF[:,:, i_band] = regression_estimator.b.T #check dims are consitent
        if regression_estimator.noise_covariance is not None:
            self.Cov_NN[:,:, i_band] = regression_estimator.noise_covariance
        if regression_estimator.inverse_signal_covariance is not None:
            self.Cov_SS[:,:, i_band] = regression_estimator.inverse_signal_covariance
        if regression_estimator.R2 is not None:
            self.R2[:, i_band] = regression_estimator.R2;
        self.num_segments[:self.num_channels_out, i_band] = regression_estimator.n_data
        return

    # def set_tf_row(self,i_band, i_row, regression_estimator, T):
    #     """
    #     @Gary this TF object appears to be 4-dimensional, not 3D ...
    #     This was going back and forth with Maxim and Gary and this stuff was
    #     being used in the context of the Multiple Station program.
    #
    #     Perhaps consider using this later.
    #     This was notionally about fixing individual rows of the TF
    #     like say you have a good channel and a bad channel at a station,
    #     you can esimate at least part of the TF
    #
    #     Parameters
    #     ----------
    #     i_band
    #     i_row
    #     regression_estimator
    #     T
    #
    #     Returns
    #     -------
    #
    #     """
    #     if not self.initialized:
    #         print('Initialize TTrFunGeneral obect before calling setTF')
    #         raise Exception
    #     print("@Gary: What is up with this block here?")
    #     #if nargin < 6:
    #     #    iSite = 1;
    #
    #     n_data = regression_estimator.n_data  # use the class luke
    #     #[nData, ~] = size(TRegObj.Y);
    #     n, m = regression_estimator.b.shape
    #
    #     self.FullCov[ib, iSite] = 0;
    #     if (n == self.num_channels_in) & (m == 1):
    #         self.TF[i_row,:, i_band, iSite] = TRegObj.b
    #         self.R2[ir, ib, iSite] = regression_estimator.R2
    #         self.periods[ib, iSite] = T
    #         self.num_segments[ir, ib, iSite] = n_data
    #     else:
    #         print('regression_estimator not proper size for operation in '
    #               'setTFRow')
    #         raise Exception
    #     return

    def standard_error(self):
        stderr = np.zeros(self.TF.shape)
        for j in range(self.num_channels_out):
            for k in range(self.num_channels_in):
                stderr[j, k,:] = np.sqrt(self.Cov_NN[j, j,:] * self.Cov_SS[k, k,:]);
        return stderr

    #<TO BE DEPRECATED/MERGE WITH BandAveragingScheme>
    def get_num_bands(self):
        return len(self.T)

    def get_frequencies(self):
        return 1. / self.T
    #</TO BE DEPRECATED/MERGE WITH BandAveragingScheme>

def test_ttf():
    from iris_mt_scratch.sandbox.transfer_function.transfer_function_header \
        import TransferFunctionHeader
    tfh = TransferFunctionHeader()
    ttf = TTF(tfh, 32)
    ttf.set_tf(1,2,3)

def main():
    test_ttf()

if __name__ == '__main__':
    main()
