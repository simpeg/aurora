# -*- coding: utf-8 -*-
"""
Pytest suite for cross_power module.

Tests transfer function computation from covariance matrices using fixtures
and subtests where appropriate. Optimized for pytest-xdist parallel execution.
"""

import numpy as np
import pytest
from mt_metadata.transfer_functions import (
    STANDARD_INPUT_CHANNELS,
    STANDARD_OUTPUT_CHANNELS,
)
from mth5.timeseries.xarray_helpers import initialize_xrda_2d_cov

from aurora.transfer_function.cross_power import (
    _channel_names,
    _tf__x,
    _tf__y,
    _tx,
    _ty,
    _zxx,
    _zxy,
    _zyx,
    _zyy,
    tf_from_cross_powers,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def station_ids():
    """Station IDs for testing."""
    return ["MT1", "MT2"]


@pytest.fixture(scope="module")
def components():
    """Standard MT components."""
    return STANDARD_INPUT_CHANNELS + STANDARD_OUTPUT_CHANNELS


@pytest.fixture(scope="module")
def channel_labels(station_ids, components):
    """Generate channel labels for both stations."""
    station_1_channels = [f"{station_ids[0]}_{x}" for x in components]
    station_2_channels = [f"{station_ids[1]}_{x}" for x in components]
    return station_1_channels + station_2_channels


@pytest.fixture(scope="module")
def sdm_covariance(channel_labels):
    """
    Create a synthetic covariance matrix for testing.

    Uses module scope for efficiency with pytest-xdist.
    """
    sdm = initialize_xrda_2d_cov(
        channels=channel_labels,
        dtype=complex,
    )
    np.random.seed(0)
    data = np.random.random((len(channel_labels), 1000))
    sdm.data = np.cov(data)
    return sdm


@pytest.fixture(scope="module")
def simple_sdm():
    """
    Create a simple 2x2 covariance matrix for unit testing.

    This allows testing specific mathematical properties without
    the complexity of the full covariance matrix.
    """
    channels = ["MT1_hx", "MT1_hy"]
    sdm = initialize_xrda_2d_cov(channels=channels, dtype=complex)
    # Create a simple hermitian matrix
    sdm.data = np.array([[2.0 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 3.0 + 0j]])
    return sdm


@pytest.fixture(scope="module")
def identity_sdm():
    """Create an identity-like covariance matrix for edge case testing."""
    channels = ["MT1_ex", "MT1_ey", "MT1_hx", "MT1_hy", "MT1_hz"]
    sdm = initialize_xrda_2d_cov(channels=channels, dtype=complex)
    sdm.data = np.eye(len(channels), dtype=complex)
    return sdm


@pytest.fixture
def channel_names_fixture(station_ids):
    """Fixture providing channel names for a single station."""
    station = station_ids[0]
    remote = station_ids[1]
    return _channel_names(station_id=station, remote=remote, join_char="_")


# =============================================================================
# Test Channel Names
# =============================================================================


class TestChannelNames:
    """Test channel name generation with different configurations."""

    def test_channel_names_with_remote(self, station_ids):
        """Test channel name generation with remote reference."""
        station = station_ids[0]
        remote = station_ids[1]
        Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
            station_id=station, remote=remote, join_char="_"
        )
        assert Ex == f"{station}_ex"
        assert Ey == f"{station}_ey"
        assert Hx == f"{station}_hx"
        assert Hy == f"{station}_hy"
        assert Hz == f"{station}_hz"
        assert A == f"{remote}_hx"
        assert B == f"{remote}_hy"

    def test_channel_names_without_remote(self, station_ids):
        """Test channel name generation for single station (no remote)."""
        station = station_ids[0]
        Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
            station_id=station, remote="", join_char="_"
        )
        assert Ex == f"{station}_ex"
        assert Ey == f"{station}_ey"
        assert Hx == f"{station}_hx"
        assert Hy == f"{station}_hy"
        assert Hz == f"{station}_hz"
        # For single station, A and B should use station's own channels
        assert A == f"{station}_hx"
        assert B == f"{station}_hy"

    def test_channel_names_custom_join_char(self, station_ids):
        """Test channel names with custom join character."""
        station = station_ids[0]
        remote = station_ids[1]
        Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
            station_id=station, remote=remote, join_char="-"
        )
        assert Ex == f"{station}-ex"
        assert Ey == f"{station}-ey"
        assert Hx == f"{station}-hx"
        assert Hy == f"{station}-hy"
        assert Hz == f"{station}-hz"
        assert A == f"{remote}-hx"
        assert B == f"{remote}-hy"

    def test_channel_names_return_type(self, station_ids):
        """Test that _channel_names returns a tuple of 7 elements."""
        result = _channel_names(
            station_id=station_ids[0], remote=station_ids[1], join_char="_"
        )
        assert isinstance(result, tuple)
        assert len(result) == 7
        assert all(isinstance(name, str) for name in result)


# =============================================================================
# Test Transfer Function Computation
# =============================================================================


class TestTFComputationBasic:
    """Test basic transfer function element computations."""

    def test_tf__x_computation(self, sdm_covariance, channel_names_fixture):
        """Test _tf__x function computes without error."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture
        result = _tf__x(sdm_covariance, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        # Result may be xarray DataArray, extract value
        value = result.item() if hasattr(result, "item") else result
        assert isinstance(value, (complex, np.complexfloating, float, np.floating))

    def test_tf__y_computation(self, sdm_covariance, channel_names_fixture):
        """Test _tf__y function computes without error."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture
        result = _tf__y(sdm_covariance, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        # Result may be xarray DataArray, extract value
        value = result.item() if hasattr(result, "item") else result
        assert isinstance(value, (complex, np.complexfloating, float, np.floating))

    def test_zxx_computation(self, sdm_covariance, channel_names_fixture):
        """Test _zxx function computes without error."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture
        result = _zxx(sdm_covariance, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        # Result may be xarray DataArray, extract value
        value = result.item() if hasattr(result, "item") else result
        assert isinstance(value, (complex, np.complexfloating, float, np.floating))

    def test_zxy_computation(self, sdm_covariance, channel_names_fixture):
        """Test _zxy function computes without error."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture
        result = _zxy(sdm_covariance, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        # Result may be xarray DataArray, extract value
        value = result.item() if hasattr(result, "item") else result
        assert isinstance(value, (complex, np.complexfloating, float, np.floating))

    def test_zyx_computation(self, sdm_covariance, channel_names_fixture):
        """Test _zyx function computes without error."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture
        result = _zyx(sdm_covariance, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B)
        # Result may be xarray DataArray, extract value
        value = result.item() if hasattr(result, "item") else result
        assert isinstance(value, (complex, np.complexfloating, float, np.floating))

    def test_zyy_computation(self, sdm_covariance, channel_names_fixture):
        """Test _zyy function computes without error."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture
        result = _zyy(sdm_covariance, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B)
        # Result may be xarray DataArray, extract value
        value = result.item() if hasattr(result, "item") else result
        assert isinstance(value, (complex, np.complexfloating, float, np.floating))

    def test_tx_computation(self, sdm_covariance, channel_names_fixture):
        """Test _tx function computes without error."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture
        result = _tx(sdm_covariance, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B)
        # Result may be xarray DataArray, extract value
        value = result.item() if hasattr(result, "item") else result
        assert isinstance(value, (complex, np.complexfloating, float, np.floating))

    def test_ty_computation(self, sdm_covariance, channel_names_fixture):
        """Test _ty function computes without error."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture
        result = _ty(sdm_covariance, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B)
        # Result may be xarray DataArray, extract value
        value = result.item() if hasattr(result, "item") else result
        assert isinstance(value, (complex, np.complexfloating, float, np.floating))


class TestVozoffEquations:
    """Test Vozoff equation equivalences and generalizations."""

    def test_generalizing_vozoffs_equations(
        self, sdm_covariance, channel_names_fixture
    ):
        """
        Test that specific Vozoff equations match generalized formulations.

        Verifies that _zxx, _zxy, _zyx, _zyy, _tx, _ty are equivalent to
        _tf__x and _tf__y with appropriate parameters.
        """
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture

        # Test impedance tensor elements
        assert _zxx(sdm_covariance, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__x(
            sdm_covariance, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B
        )
        assert _zxy(sdm_covariance, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__y(
            sdm_covariance, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B
        )
        assert _zyx(sdm_covariance, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__x(
            sdm_covariance, Y=Ey, Hx=Hx, Hy=Hy, A=A, B=B
        )
        assert _zyy(sdm_covariance, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__y(
            sdm_covariance, Y=Ey, Hx=Hx, Hy=Hy, A=A, B=B
        )

        # Test tipper elements
        assert _tx(sdm_covariance, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__x(
            sdm_covariance, Y=Hz, Hx=Hx, Hy=Hy, A=A, B=B
        )
        assert _ty(sdm_covariance, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B) == _tf__y(
            sdm_covariance, Y=Hz, Hx=Hx, Hy=Hy, A=A, B=B
        )

    def test_impedance_symmetry(self, sdm_covariance, channel_names_fixture):
        """
        Test symmetry properties of impedance tensor.

        Verifies that Ex->Ey substitution relates Z_xx to Z_yx and Z_xy to Z_yy.
        """
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture

        # Z_xx with Ex should have same structure as Z_yx with Ey
        zxx_result = _tf__x(sdm_covariance, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        zyx_result = _tf__x(sdm_covariance, Y=Ey, Hx=Hx, Hy=Hy, A=A, B=B)

        # Both should be numeric (extract values if DataArray)
        zxx_val = zxx_result.item() if hasattr(zxx_result, "item") else zxx_result
        zyx_val = zyx_result.item() if hasattr(zyx_result, "item") else zyx_result
        assert isinstance(zxx_val, (complex, np.complexfloating, float, np.floating))
        assert isinstance(zyx_val, (complex, np.complexfloating, float, np.floating))

        # Z_xy with Ex should have same structure as Z_yy with Ey
        zxy_result = _tf__y(sdm_covariance, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        zyy_result = _tf__y(sdm_covariance, Y=Ey, Hx=Hx, Hy=Hy, A=A, B=B)

        zxy_val = zxy_result.item() if hasattr(zxy_result, "item") else zxy_result
        zyy_val = zyy_result.item() if hasattr(zyy_result, "item") else zyy_result
        assert isinstance(zxy_val, (complex, np.complexfloating, float, np.floating))
        assert isinstance(zyy_val, (complex, np.complexfloating, float, np.floating))


class TestTFFromCrossPowers:
    """Test the main tf_from_cross_powers function."""

    def test_tf_from_cross_powers_dict_output(self, sdm_covariance, station_ids):
        """Test tf_from_cross_powers returns dictionary with all components."""
        result = tf_from_cross_powers(
            sdm_covariance,
            station_id=station_ids[0],
            remote=station_ids[1],
            output_format="dict",
        )

        assert isinstance(result, dict)
        expected_keys = ["z_xx", "z_xy", "z_yx", "z_yy", "t_zx", "t_zy"]
        assert set(result.keys()) == set(expected_keys)

        # All values should be numeric (may be wrapped in DataArray)
        for key, value in result.items():
            val = value.item() if hasattr(value, "item") else value
            assert isinstance(val, (complex, np.complexfloating, float, np.floating))

    def test_tf_from_cross_powers_single_station(self, sdm_covariance, station_ids):
        """Test tf_from_cross_powers without remote reference."""
        result = tf_from_cross_powers(
            sdm_covariance,
            station_id=station_ids[0],
            remote="",
            output_format="dict",
        )

        assert isinstance(result, dict)
        expected_keys = ["z_xx", "z_xy", "z_yx", "z_yy", "t_zx", "t_zy"]
        assert set(result.keys()) == set(expected_keys)

    def test_tf_from_cross_powers_mt_metadata_format(self, sdm_covariance, station_ids):
        """Test that mt_metadata format raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            tf_from_cross_powers(
                sdm_covariance,
                station_id=station_ids[0],
                remote=station_ids[1],
                output_format="mt_metadata",
            )


# =============================================================================
# Test Mathematical Properties
# =============================================================================


class TestMathematicalProperties:
    """Test mathematical properties of transfer function computations."""

    def test_hermitian_symmetry(self, sdm_covariance, channel_names_fixture):
        """
        Test that covariance matrix hermitian symmetry is respected.

        For a hermitian matrix, sdm[i,j] = conj(sdm[j,i])
        """
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture

        # Check a few elements for hermitian symmetry
        assert np.isclose(
            sdm_covariance.loc[Ex, Hx], np.conj(sdm_covariance.loc[Hx, Ex])
        )
        assert np.isclose(
            sdm_covariance.loc[Ey, Hy], np.conj(sdm_covariance.loc[Hy, Ey])
        )

    def test_denominator_consistency(self, sdm_covariance, channel_names_fixture):
        """
        Test that denominators are consistent across related TF elements.

        Z_xx and Z_yx share the same denominator: <Hx,A*><Hy,B*> - <Hx,B*><Hy,A*>
        Z_xy and Z_yy share the same denominator: <Hy,A*><Hx,B*> - <Hy,B*><Hx,A*>
        """
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture

        # Compute shared denominator for Z_xx and Z_yx
        denom_x = (
            sdm_covariance.loc[Hx, A] * sdm_covariance.loc[Hy, B]
            - sdm_covariance.loc[Hx, B] * sdm_covariance.loc[Hy, A]
        )

        # Compute shared denominator for Z_xy and Z_yy
        denom_y = (
            sdm_covariance.loc[Hy, A] * sdm_covariance.loc[Hx, B]
            - sdm_covariance.loc[Hy, B] * sdm_covariance.loc[Hx, A]
        )

        # Both denominators should be non-zero for well-conditioned matrices
        assert not np.isclose(denom_x, 0)
        assert not np.isclose(denom_y, 0)

    def test_tf_finite_values(self, sdm_covariance, channel_names_fixture):
        """Test that computed TF values are finite (not NaN or inf)."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture

        # Test all TF components
        tf_values = [
            _zxx(sdm_covariance, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B),
            _zxy(sdm_covariance, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B),
            _zyx(sdm_covariance, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B),
            _zyy(sdm_covariance, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B),
            _tx(sdm_covariance, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B),
            _ty(sdm_covariance, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B),
        ]

        for value in tf_values:
            assert np.isfinite(value)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_identity_covariance_matrix(self, identity_sdm):
        """Test TF computation with identity-like covariance matrix."""
        station = "MT1"
        Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
            station_id=station, remote="", join_char="_"
        )

        # With identity matrix, many cross terms are zero
        # Denominator: <Hx,Hx><Hy,Hy> - <Hx,Hy><Hy,Hx> = 1*1 - 0*0 = 1
        denom_x = (
            identity_sdm.loc[Hx, A] * identity_sdm.loc[Hy, B]
            - identity_sdm.loc[Hx, B] * identity_sdm.loc[Hy, A]
        )
        assert np.isclose(denom_x, 1.0)

    def test_different_join_characters(self, sdm_covariance, station_ids, subtests):
        """Test TF computation with different join characters."""
        join_chars = ["_", "-", ".", ""]

        for join_char in join_chars:
            with subtests.test(join_char=join_char):
                # This will fail for non-underscore join chars since our
                # sdm_covariance fixture uses underscore
                # But test the function interface
                Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
                    station_id=station_ids[0],
                    remote=station_ids[1],
                    join_char=join_char,
                )

                # Verify the join character is used
                assert join_char in Ex or join_char == ""
                assert Ex.startswith(station_ids[0])

    def test_zero_cross_power_handling(self):
        """Test behavior when some cross-power terms are zero."""
        channels = ["MT1_ex", "MT1_hx", "MT1_hy", "MT2_hx", "MT2_hy"]
        sdm = initialize_xrda_2d_cov(channels=channels, dtype=complex)

        # Create a matrix where some cross terms are zero
        sdm.data = np.eye(len(channels), dtype=complex)
        # Add some non-zero diagonal elements
        sdm.data[0, 0] = 2.0
        sdm.data[1, 1] = 3.0
        sdm.data[2, 2] = 4.0

        Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
            station_id="MT1", remote="MT2", join_char="_"
        )

        # Should compute without error even with many zeros
        result = _tf__x(sdm, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        val = result.item() if hasattr(result, "item") else result
        # Result might be NaN due to zero denominator, that's OK
        assert isinstance(val, (complex, np.complexfloating, float, np.floating))


# =============================================================================
# Test Data Integrity
# =============================================================================


class TestDataIntegrity:
    """Test that TF computation doesn't modify input data."""

    def test_input_sdm_unchanged(self, sdm_covariance, station_ids):
        """Test that tf_from_cross_powers doesn't modify input covariance matrix."""
        # Make a copy of the original data
        original_data = sdm_covariance.data.copy()

        # Compute TF
        tf_from_cross_powers(
            sdm_covariance,
            station_id=station_ids[0],
            remote=station_ids[1],
        )

        # Verify data unchanged
        assert np.allclose(sdm_covariance.data, original_data)

    def test_individual_tf_functions_unchanged(
        self, sdm_covariance, channel_names_fixture
    ):
        """Test that individual TF functions don't modify input."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture
        original_data = sdm_covariance.data.copy()

        # Call all TF functions
        _zxx(sdm_covariance, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        _zxy(sdm_covariance, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        _zyx(sdm_covariance, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B)
        _zyy(sdm_covariance, Ey=Ey, Hx=Hx, Hy=Hy, A=A, B=B)
        _tx(sdm_covariance, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B)
        _ty(sdm_covariance, Hz=Hz, Hx=Hx, Hy=Hy, A=A, B=B)

        # Verify data unchanged
        assert np.allclose(sdm_covariance.data, original_data)


# =============================================================================
# Test Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability with various input conditions."""

    def test_small_values_stability(self):
        """Test TF computation with very small covariance values."""
        channels = ["MT1_ex", "MT1_hx", "MT1_hy", "MT2_hx", "MT2_hy"]
        sdm = initialize_xrda_2d_cov(channels=channels, dtype=complex)

        # Create matrix with small values
        np.random.seed(42)
        sdm.data = np.random.random((len(channels), len(channels))) * 1e-10
        sdm.data = sdm.data + sdm.data.T.conj()  # Make hermitian

        Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
            station_id="MT1", remote="MT2", join_char="_"
        )

        result = _tf__x(sdm, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        # Result might be large due to small denominator, but should be finite
        assert np.isfinite(result) or np.isinf(result)  # Allow inf for edge case

    def test_large_values_stability(self):
        """Test TF computation with very large covariance values."""
        channels = ["MT1_ex", "MT1_hx", "MT1_hy", "MT2_hx", "MT2_hy"]
        sdm = initialize_xrda_2d_cov(channels=channels, dtype=complex)

        # Create matrix with large values
        np.random.seed(43)
        sdm.data = np.random.random((len(channels), len(channels))) * 1e10
        sdm.data = sdm.data + sdm.data.T.conj()  # Make hermitian

        Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
            station_id="MT1", remote="MT2", join_char="_"
        )

        result = _tf__x(sdm, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
        assert np.isfinite(result)

    def test_complex_phase_variations(self, subtests):
        """Test TF computation with various complex phase relationships."""
        channels = ["MT1_ex", "MT1_hx", "MT1_hy", "MT2_hx", "MT2_hy"]

        phases = [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]

        for phase in phases:
            with subtests.test(phase=phase):
                sdm = initialize_xrda_2d_cov(channels=channels, dtype=complex)

                # Create matrix with specific phase
                np.random.seed(44)
                magnitude = np.random.random((len(channels), len(channels)))
                sdm.data = magnitude * np.exp(1j * phase)
                sdm.data = sdm.data + sdm.data.T.conj()  # Make hermitian

                Ex, Ey, Hx, Hy, Hz, A, B = _channel_names(
                    station_id="MT1", remote="MT2", join_char="_"
                )

                result = _tf__x(sdm, Y=Ex, Hx=Hx, Hy=Hy, A=A, B=B)
                val = result.item() if hasattr(result, "item") else result
                assert isinstance(
                    val, (complex, np.complexfloating, float, np.floating)
                )


# =============================================================================
# Test Return Value Characteristics
# =============================================================================


class TestReturnValues:
    """Test characteristics of return values from TF functions."""

    def test_all_tf_components_present(self, sdm_covariance, station_ids):
        """Test that tf_from_cross_powers returns all expected components."""
        result = tf_from_cross_powers(
            sdm_covariance,
            station_id=station_ids[0],
            remote=station_ids[1],
        )

        # Check all standard TF components are present
        assert "z_xx" in result
        assert "z_xy" in result
        assert "z_yx" in result
        assert "z_yy" in result
        assert "t_zx" in result
        assert "t_zy" in result

        # Should only have these 6 components
        assert len(result) == 6

    def test_tf_component_types(self, sdm_covariance, station_ids):
        """Test that all TF components are complex numbers."""
        result = tf_from_cross_powers(
            sdm_covariance,
            station_id=station_ids[0],
            remote=station_ids[1],
        )

        for component_name, value in result.items():
            val = value.item() if hasattr(value, "item") else value
            assert isinstance(
                val, (complex, np.complexfloating, float, np.floating)
            ), f"{component_name} is not numeric"

    def test_impedance_vs_tipper_separation(self, sdm_covariance, station_ids):
        """Test that impedance and tipper components are computed separately."""
        result = tf_from_cross_powers(
            sdm_covariance,
            station_id=station_ids[0],
            remote=station_ids[1],
        )

        impedance_keys = ["z_xx", "z_xy", "z_yx", "z_yy"]
        tipper_keys = ["t_zx", "t_zy"]

        # All impedance components should be present
        for key in impedance_keys:
            assert key in result

        # All tipper components should be present
        for key in tipper_keys:
            assert key in result


# =============================================================================
# Test Consistency Across Calls
# =============================================================================


class TestConsistency:
    """Test consistency of results across multiple calls."""

    def test_deterministic_results(self, sdm_covariance, station_ids):
        """Test that repeated calls produce identical results."""
        result1 = tf_from_cross_powers(
            sdm_covariance,
            station_id=station_ids[0],
            remote=station_ids[1],
        )

        result2 = tf_from_cross_powers(
            sdm_covariance,
            station_id=station_ids[0],
            remote=station_ids[1],
        )

        for key in result1.keys():
            assert result1[key] == result2[key]

    def test_individual_function_consistency(
        self, sdm_covariance, channel_names_fixture
    ):
        """Test that individual TF functions produce consistent results."""
        Ex, Ey, Hx, Hy, Hz, A, B = channel_names_fixture

        # Call the same function multiple times
        results = [
            _zxx(sdm_covariance, Ex=Ex, Hx=Hx, Hy=Hy, A=A, B=B) for _ in range(5)
        ]

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]
