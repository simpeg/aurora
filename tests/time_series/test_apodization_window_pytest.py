"""
Tests for ApodizationWindow class.

Tests window generation, properties, and various taper families.
"""

import numpy as np
import pytest
from loguru import logger

from aurora.time_series.apodization_window import ApodizationWindow


# Fixtures for commonly used window configurations
@pytest.fixture
def boxcar_window():
    """Default boxcar window."""
    return ApodizationWindow(num_samples_window=4)


@pytest.fixture
def hamming_window():
    """Standard Hamming window."""
    return ApodizationWindow(taper_family="hamming", num_samples_window=128)


@pytest.fixture
def blackmanharris_window():
    """Blackman-Harris window."""
    return ApodizationWindow(taper_family="blackmanharris", num_samples_window=256)


class TestApodizationWindowBasic:
    """Test basic ApodizationWindow functionality."""

    def test_default_boxcar(self, boxcar_window):
        """Test default boxcar window properties."""
        assert boxcar_window.nenbw == 1.0
        assert boxcar_window.coherent_gain == 1.0
        assert boxcar_window.apodization_factor == 1.0
        logger.info(boxcar_window.summary)

    def test_hamming(self, hamming_window):
        """Test Hamming window properties."""
        assert np.isclose(hamming_window.nenbw, 1.362825788751716)
        assert np.isclose(hamming_window.coherent_gain, 0.54)
        assert np.isclose(hamming_window.apodization_factor, 0.6303967004989797)
        logger.info(hamming_window.summary)

    def test_blackmanharris(self, blackmanharris_window):
        """Test Blackman-Harris window properties."""
        assert np.isclose(blackmanharris_window.nenbw, 2.0043529382170493)
        assert np.isclose(blackmanharris_window.coherent_gain, 0.35874999999999996)
        assert np.isclose(blackmanharris_window.apodization_factor, 0.5079009302511663)
        logger.info(blackmanharris_window.summary)

    def test_kaiser(self):
        """Test Kaiser window with beta parameter."""
        window = ApodizationWindow(
            taper_family="kaiser",
            num_samples_window=128,
            taper_additional_args={"beta": 8},
        )
        logger.info(window.summary)

        # Verify window properties are calculated
        assert window.nenbw > 0
        assert window.coherent_gain > 0
        assert window.apodization_factor > 0
        assert len(window.taper) == 128

    def test_tukey(self):
        """Test Tukey window with alpha parameter."""
        window = ApodizationWindow(
            taper_family="tukey",
            num_samples_window=30000,
            taper_additional_args={"alpha": 0.25},
        )
        logger.info(window.summary)

        # Verify window is created correctly
        assert len(window.taper) == 30000
        assert window.nenbw > 0

    def test_dpss(self):
        """Test DPSS (Slepian) window."""
        window = ApodizationWindow(
            taper_family="dpss",
            num_samples_window=64,
            taper_additional_args={"NW": 3.0},
        )
        logger.info(window.summary)

        assert len(window.taper) == 64
        assert window.nenbw > 0

    def test_custom(self):
        """Test custom window from user-provided array."""
        custom_taper = np.abs(np.random.randn(64))
        window = ApodizationWindow(
            taper_family="custom",
            num_samples_window=64,
            taper=custom_taper,
        )
        logger.info(window.summary)

        # Verify custom taper is used
        assert np.allclose(window.taper, custom_taper)
        assert len(window.taper) == 64


class TestApodizationWindowProperties:
    """Test window properties and attributes."""

    def test_window_length(self, subtests):
        """Test that window length matches requested samples."""
        window_lengths = [16, 32, 64, 128, 256, 512]

        for length in window_lengths:
            with subtests.test(length=length):
                window = ApodizationWindow(num_samples_window=length)
                assert len(window.taper) == length

    def test_coherent_gain_range(self, subtests):
        """Test that coherent gain is in valid range for standard windows."""
        taper_families = ["boxcar", "hamming", "hann", "blackman", "blackmanharris"]

        for family in taper_families:
            with subtests.test(taper_family=family):
                window = ApodizationWindow(taper_family=family, num_samples_window=128)
                # Coherent gain should be between 0 and 1
                assert 0 < window.coherent_gain <= 1.0

    def test_nenbw_positive(self, subtests):
        """Test that NENBW is positive for all window types."""
        taper_families = ["boxcar", "hamming", "hann", "blackman", "blackmanharris"]

        for family in taper_families:
            with subtests.test(taper_family=family):
                window = ApodizationWindow(taper_family=family, num_samples_window=128)
                assert window.nenbw > 0

    def test_window_normalization(self, subtests):
        """Test that windows are properly normalized."""
        taper_families = ["boxcar", "hamming", "hann", "blackman"]

        for family in taper_families:
            with subtests.test(taper_family=family):
                window = ApodizationWindow(taper_family=family, num_samples_window=128)
                # Maximum value should be close to 1 (normalized)
                assert np.max(window.taper) <= 1.0
                assert np.max(window.taper) >= 0.9  # Allow some tolerance


class TestApodizationWindowEdgeCases:
    """Test edge cases and error handling."""

    def test_small_window(self):
        """Test with very small window size."""
        window = ApodizationWindow(num_samples_window=2)
        assert len(window.taper) == 2
        assert window.nenbw > 0

    def test_large_window(self):
        """Test with large window size."""
        window = ApodizationWindow(num_samples_window=10000)
        assert len(window.taper) == 10000
        assert window.nenbw > 0

    def test_power_of_two_windows(self, subtests):
        """Test common power-of-two window sizes used in FFT."""
        powers = [4, 5, 6, 7, 8, 9, 10]  # 16, 32, 64, 128, 256, 512, 1024

        for power in powers:
            with subtests.test(power=power):
                length = 2**power
                window = ApodizationWindow(num_samples_window=length)
                assert len(window.taper) == length
                assert window.nenbw > 0


class TestApodizationWindowCalculations:
    """Test window calculations and derived properties."""

    def test_apodization_factor_range(self, subtests):
        """Test that apodization factor is in valid range."""
        taper_families = ["boxcar", "hamming", "hann", "blackman"]

        for family in taper_families:
            with subtests.test(taper_family=family):
                window = ApodizationWindow(taper_family=family, num_samples_window=256)
                # Apodization factor should be between 0 and 1
                assert 0 < window.apodization_factor <= 1.0

    def test_boxcar_unity_properties(self):
        """Test that boxcar window has unity properties."""
        window = ApodizationWindow(num_samples_window=100)

        # Boxcar should have all properties equal to 1
        assert window.nenbw == 1.0
        assert window.coherent_gain == 1.0
        assert window.apodization_factor == 1.0
        # All samples should be 1
        assert np.allclose(window.taper, 1.0)

    def test_window_energy_conservation(self, subtests):
        """Test that window energy is properly calculated."""
        taper_families = ["boxcar", "hamming", "hann", "blackman"]

        for family in taper_families:
            with subtests.test(taper_family=family):
                window = ApodizationWindow(taper_family=family, num_samples_window=128)
                # Energy should be positive and finite
                energy = np.sum(window.taper**2)
                assert energy > 0
                assert np.isfinite(energy)

    def test_linear_spectral_density_factor(self, subtests) -> None:
        r"""
        This is just a test to verify some algebra

        Claim:
        The lsd_calibration factors
        A      (1./coherent_gain)*np.sqrt((2*dt)/(nenbw*N))
        and
        B      np.sqrt(2/(sample_rate*self.S2))

        Note sqrt(2*dt)==sqrt(2*sample_rate) so we can cancel these terms and A=B IFF

        (1./coherent_gain) * np.sqrt(1/(nenbw*N)) == 1/np.sqrt(S2)
        which is shown in github aurora issue #3 via (CG**2) * NENBW *N = S2

        """
        taper_families = ["boxcar", "hamming", "hann", "blackman"]

        for family in taper_families:
            with subtests.test(taper_family=family):
                window = ApodizationWindow(taper_family=family, num_samples_window=256)
                lsd_factor1 = (1.0 / window.coherent_gain) * np.sqrt(
                    1.0 / (window.nenbw * window.num_samples_window)
                )
                lsd_factor2 = 1.0 / np.sqrt(window.S2)
                if not np.isclose(lsd_factor1, lsd_factor2):
                    msg = f"Linear spectral density factors do not match for {family} window: \n"
                    msg += f"lsd_factor1 {lsd_factor1} vs lsd_factor2 {lsd_factor2}"
                    logger.error(msg)
                    raise Exception(msg)


class TestApodizationWindowParameterVariations:
    """Test windows with various parameter combinations."""

    def test_kaiser_beta_variations(self, subtests):
        """Test Kaiser window with different beta values."""
        beta_values = [0, 2, 5, 8, 14]

        for beta in beta_values:
            with subtests.test(beta=beta):
                window = ApodizationWindow(
                    taper_family="kaiser",
                    num_samples_window=128,
                    taper_additional_args={"beta": beta},
                )
                assert len(window.taper) == 128
                assert window.nenbw > 0
                # Higher beta should give wider main lobe (higher NENBW)
                logger.info(f"Kaiser beta={beta}: NENBW={window.nenbw}")

    def test_tukey_alpha_variations(self, subtests):
        """Test Tukey window with different alpha values."""
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for alpha in alpha_values:
            with subtests.test(alpha=alpha):
                window = ApodizationWindow(
                    taper_family="tukey",
                    num_samples_window=256,
                    taper_additional_args={"alpha": alpha},
                )
                assert len(window.taper) == 256
                assert window.nenbw > 0
                logger.info(f"Tukey alpha={alpha}: NENBW={window.nenbw}")

    def test_dpss_nw_variations(self, subtests):
        """Test DPSS window with different NW values."""
        nw_values = [2.0, 2.5, 3.0, 3.5, 4.0]

        for nw in nw_values:
            with subtests.test(NW=nw):
                window = ApodizationWindow(
                    taper_family="dpss",
                    num_samples_window=128,
                    taper_additional_args={"NW": nw},
                )
                assert len(window.taper) == 128
                assert window.nenbw > 0
                logger.info(f"DPSS NW={nw}: NENBW={window.nenbw}")


class TestApodizationWindowComparison:
    """Test comparisons between different window types."""

    def test_window_selectivity_ordering(self):
        """Test that windows follow expected selectivity ordering."""
        # Create windows with same size
        size = 256
        boxcar = ApodizationWindow(taper_family="boxcar", num_samples_window=size)
        hann = ApodizationWindow(taper_family="hann", num_samples_window=size)
        hamming = ApodizationWindow(taper_family="hamming", num_samples_window=size)
        blackman = ApodizationWindow(taper_family="blackman", num_samples_window=size)

        # Boxcar should have lowest NENBW (narrowest main lobe)
        assert boxcar.nenbw < hamming.nenbw
        assert hamming.nenbw < hann.nenbw
        # Blackman has wider main lobe than Hamming
        assert hamming.nenbw < blackman.nenbw

    def test_different_sizes_same_family(self, subtests):
        """Test that window properties scale appropriately with size."""
        sizes = [64, 128, 256, 512]

        for size in sizes:
            with subtests.test(size=size):
                window = ApodizationWindow(
                    taper_family="hamming", num_samples_window=size
                )
                # Coherent gain should be constant for same family
                assert np.isclose(window.coherent_gain, 0.54, atol=0.01)


class TestApodizationWindowSummary:
    """Test summary and string representations."""

    def test_summary_not_empty(self, subtests):
        """Test that summary is generated for all window types."""
        taper_families = ["boxcar", "hamming", "hann", "blackman", "blackmanharris"]

        for family in taper_families:
            with subtests.test(taper_family=family):
                window = ApodizationWindow(taper_family=family, num_samples_window=128)
                summary = window.summary
                assert isinstance(summary, str)
                assert len(summary) > 0
                assert family in summary.lower() or "boxcar" in summary.lower()
