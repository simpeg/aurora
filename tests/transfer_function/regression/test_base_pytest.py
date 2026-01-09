# -*- coding: utf-8 -*-
"""
Pytest suite for RegressionEstimator base class.

Tests transfer function regression using fixtures and subtests.
Optimized for pytest-xdist parallel execution.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aurora.transfer_function.regression.base import RegressionEstimator
from aurora.transfer_function.regression.iter_control import IterControl


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def expected_solution():
    """Expected solution for mini dataset regression."""
    return np.array([-0.04192569 - 0.36502722j, -3.65284496 - 4.05194938j])


@pytest.fixture(scope="module")
def mini_dataset_full():
    """Create full mini dataset with 3 rows."""
    ex_data = np.array(
        [
            4.39080123e-07 - 2.41097397e-06j,
            -2.33418464e-06 + 2.10752581e-06j,
            1.38642624e-06 - 1.87333571e-06j,
        ]
    )
    hx_data = np.array(
        [
            7.00767250e-07 - 9.18819198e-07j,
            -1.06648904e-07 + 8.19420154e-07j,
            -1.02700963e-07 - 3.73904463e-07j,
        ]
    )
    hy_data = np.array(
        [
            1.94321684e-07 + 3.71934877e-07j,
            1.15361101e-08 - 6.32581646e-07j,
            3.86095787e-08 + 4.33155345e-07j,
        ]
    )
    timestamps = pd.date_range(
        start=pd.Timestamp("1977-03-02T06:00:00"), periods=len(ex_data), freq="s"
    )
    frequency = 0.666 * np.ones(len(ex_data))

    df = pd.DataFrame(
        data={
            "time": timestamps,
            "frequency": frequency,
            "ex": ex_data,
            "hx": hx_data,
            "hy": hy_data,
        }
    )
    df = df.set_index(["time", "frequency"])
    return df.to_xarray()


@pytest.fixture(scope="module")
def mini_dataset_single():
    """Create mini dataset with 1 row (underdetermined)."""
    ex_data = np.array([4.39080123e-07 - 2.41097397e-06j])
    hx_data = np.array([7.00767250e-07 - 9.18819198e-07j])
    hy_data = np.array([1.94321684e-07 + 3.71934877e-07j])

    timestamps = pd.date_range(
        start=pd.Timestamp("1977-03-02T06:00:00"), periods=len(ex_data), freq="s"
    )
    frequency = 0.666 * np.ones(len(ex_data))

    df = pd.DataFrame(
        data={
            "time": timestamps,
            "frequency": frequency,
            "ex": ex_data,
            "hx": hx_data,
            "hy": hy_data,
        }
    )
    df = df.set_index(["time", "frequency"])
    return df.to_xarray()


@pytest.fixture
def dataset_xy_full(mini_dataset_full):
    """Prepare X and Y datasets from full mini dataset."""
    X = mini_dataset_full[["hx", "hy"]]
    X = X.stack(observation=("frequency", "time"))
    Y = mini_dataset_full[["ex"]]
    Y = Y.stack(observation=("frequency", "time"))
    return X, Y


@pytest.fixture
def dataset_xy_single(mini_dataset_single):
    """Prepare X and Y datasets from single-row mini dataset."""
    X = mini_dataset_single[["hx", "hy"]]
    X = X.stack(observation=("frequency", "time"))
    Y = mini_dataset_single[["ex"]]
    Y = Y.stack(observation=("frequency", "time"))
    return X, Y


@pytest.fixture
def regression_estimator(dataset_xy_full):
    """Create a basic RegressionEstimator instance."""
    X, Y = dataset_xy_full
    return RegressionEstimator(X=X, Y=Y)


@pytest.fixture
def simple_regression_data():
    """Create simple synthetic regression data."""
    np.random.seed(100)
    n_obs = 20
    X = np.random.randn(2, n_obs) + 1j * np.random.randn(2, n_obs)
    true_b = np.array([[1.5 + 0.5j], [-0.8 + 1.2j]])
    Y = true_b.T @ X
    return X, Y, true_b


# =============================================================================
# Test Initialization
# =============================================================================


class TestRegressionEstimatorInit:
    """Test RegressionEstimator initialization."""

    def test_init_with_xarray_dataset(self, dataset_xy_full):
        """Test initialization with xarray Dataset."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)

        assert re is not None
        assert re.X is not None
        assert re.Y is not None
        assert isinstance(re.X, np.ndarray)
        assert isinstance(re.Y, np.ndarray)

    def test_init_with_xarray_dataarray(self, dataset_xy_full):
        """Test initialization with xarray DataArray."""
        X, Y = dataset_xy_full
        X_da = X.to_array()
        Y_da = Y.to_array()

        re = RegressionEstimator(X=X_da, Y=Y_da)

        assert re is not None
        assert isinstance(re.X, np.ndarray)
        assert isinstance(re.Y, np.ndarray)

    def test_init_with_numpy_array(self, dataset_xy_full):
        """Test initialization with numpy arrays."""
        X, Y = dataset_xy_full
        X_np = X.to_array().data
        Y_np = Y.to_array().data

        re = RegressionEstimator(X=X_np, Y=Y_np)

        assert re is not None
        assert isinstance(re.X, np.ndarray)
        assert isinstance(re.Y, np.ndarray)

    def test_init_sets_attributes(self, dataset_xy_full):
        """Test that initialization sets expected attributes."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)

        assert re.b is None
        assert re.cov_nn is None
        assert re.cov_ss_inv is None
        assert re.squared_coherence is None
        assert hasattr(re, "iter_control")
        assert isinstance(re.iter_control, IterControl)

    def test_init_with_custom_iter_control(self, dataset_xy_full):
        """Test initialization with custom IterControl."""
        X, Y = dataset_xy_full
        custom_iter = IterControl(max_number_of_iterations=50)
        re = RegressionEstimator(X=X, Y=Y, iter_control=custom_iter)

        assert re.iter_control.max_number_of_iterations == 50

    def test_init_with_channel_names(self, simple_regression_data):
        """Test initialization with explicit channel names."""
        X, Y, _ = simple_regression_data
        input_names = ["hx", "hy"]
        output_names = ["ex"]

        re = RegressionEstimator(
            X=X, Y=Y, input_channel_names=input_names, output_channel_names=output_names
        )

        assert re.input_channel_names == input_names
        assert re.output_channel_names == output_names


# =============================================================================
# Test Properties
# =============================================================================


class TestRegressionEstimatorProperties:
    """Test RegressionEstimator properties."""

    def test_n_data_property(self, regression_estimator):
        """Test n_data property returns correct number of observations."""
        assert regression_estimator.n_data == 3

    def test_n_channels_in_property(self, regression_estimator):
        """Test n_channels_in property returns correct number."""
        assert regression_estimator.n_channels_in == 2

    def test_n_channels_out_property(self, regression_estimator):
        """Test n_channels_out property returns correct number."""
        assert regression_estimator.n_channels_out == 1

    def test_degrees_of_freedom_property(self, regression_estimator):
        """Test degrees_of_freedom property calculation."""
        expected_dof = regression_estimator.n_data - regression_estimator.n_channels_in
        assert regression_estimator.degrees_of_freedom == expected_dof
        assert regression_estimator.degrees_of_freedom == 1

    def test_is_underdetermined_false(self, regression_estimator):
        """Test is_underdetermined returns False for well-determined system."""
        assert regression_estimator.is_underdetermined is False

    def test_is_underdetermined_true(self, dataset_xy_single):
        """Test is_underdetermined returns True for underdetermined system."""
        X, Y = dataset_xy_single
        re = RegressionEstimator(X=X, Y=Y)
        assert re.is_underdetermined is True

    def test_input_channel_names_from_dataset(self, dataset_xy_full):
        """Test input_channel_names extracted from xarray Dataset."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        names = re.input_channel_names

        assert isinstance(names, list)
        assert len(names) == 2
        assert "hx" in names
        assert "hy" in names

    def test_output_channel_names_from_dataset(self, dataset_xy_full):
        """Test output_channel_names extracted from xarray Dataset."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        names = re.output_channel_names

        assert isinstance(names, list)
        assert len(names) == 1
        assert "ex" in names


# =============================================================================
# Test OLS Estimation
# =============================================================================


class TestOLSEstimation:
    """Test ordinary least squares estimation methods."""

    def test_estimate_ols_qr_mode(self, dataset_xy_full, expected_solution):
        """Test estimate_ols with QR mode."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols(mode="qr")

        difference = re.b - np.atleast_2d(expected_solution).T
        assert np.allclose(difference, 0)

    def test_estimate_ols_solve_mode(self, dataset_xy_full, expected_solution):
        """Test estimate_ols with solve mode."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols(mode="solve")

        difference = re.b - np.atleast_2d(expected_solution).T
        assert np.allclose(difference, 0)

    def test_estimate_ols_brute_force_mode(self, dataset_xy_full, expected_solution):
        """Test estimate_ols with brute_force mode."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols(mode="brute_force")

        difference = re.b - np.atleast_2d(expected_solution).T
        assert np.allclose(difference, 0)

    def test_estimate_ols_modes_equivalent(self, dataset_xy_full, subtests):
        """Test that different OLS modes produce equivalent results."""
        X, Y = dataset_xy_full
        modes = ["qr", "solve", "brute_force"]
        results = {}

        for mode in modes:
            with subtests.test(mode=mode):
                re = RegressionEstimator(X=X, Y=Y)
                re.estimate_ols(mode=mode)
                results[mode] = re.b.copy()

        # Compare all modes to each other
        for mode1 in modes:
            for mode2 in modes:
                if mode1 != mode2:
                    assert np.allclose(results[mode1], results[mode2])

    def test_estimate_method(self, dataset_xy_full, expected_solution):
        """Test the estimate() convenience method."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate()

        difference = re.b - np.atleast_2d(expected_solution).T
        assert np.allclose(difference, 0)

    def test_estimate_ols_returns_b(self, dataset_xy_full):
        """Test that estimate_ols returns the b matrix."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        result = re.estimate_ols()

        assert result is not None
        assert np.array_equal(result, re.b)


# =============================================================================
# Test QR Decomposition
# =============================================================================


class TestQRDecomposition:
    """Test QR decomposition functionality."""

    def test_qr_decomposition_basic(self, regression_estimator):
        """Test basic QR decomposition."""
        Q, R = regression_estimator.qr_decomposition()

        assert Q is not None
        assert R is not None
        assert isinstance(Q, np.ndarray)
        assert isinstance(R, np.ndarray)

    def test_qr_decomposition_properties(self, regression_estimator):
        """Test QR decomposition mathematical properties."""
        Q, R = regression_estimator.qr_decomposition()

        # Q should be unitary: Q^H @ Q = I
        QHQ = Q.conj().T @ Q
        assert np.allclose(QHQ, np.eye(Q.shape[1]))

        # R should be upper triangular
        assert np.allclose(R, np.triu(R))

    def test_qr_decomposition_reconstruction(self, regression_estimator):
        """Test that Q @ R reconstructs X."""
        Q, R = regression_estimator.qr_decomposition()
        X_reconstructed = Q @ R

        assert np.allclose(X_reconstructed, regression_estimator.X)

    def test_qr_decomposition_sanity_check(self, regression_estimator):
        """Test QR decomposition with sanity check enabled."""
        Q, R = regression_estimator.qr_decomposition(sanity_check=True)

        assert Q is not None
        assert R is not None

    def test_q_property(self, regression_estimator):
        """Test Q property accessor."""
        regression_estimator.qr_decomposition()
        Q = regression_estimator.Q

        assert Q is not None
        assert isinstance(Q, np.ndarray)

    def test_r_property(self, regression_estimator):
        """Test R property accessor."""
        regression_estimator.qr_decomposition()
        R = regression_estimator.R

        assert R is not None
        assert isinstance(R, np.ndarray)

    def test_qh_property(self, regression_estimator):
        """Test QH (conjugate transpose) property."""
        regression_estimator.qr_decomposition()
        QH = regression_estimator.QH
        Q = regression_estimator.Q

        assert np.allclose(QH, Q.conj().T)

    def test_qhy_property(self, regression_estimator):
        """Test QHY property."""
        regression_estimator.qr_decomposition()
        QHY = regression_estimator.QHY

        expected = regression_estimator.QH @ regression_estimator.Y
        assert np.allclose(QHY, expected)


# =============================================================================
# Test Underdetermined Systems
# =============================================================================


class TestUnderdeterminedSystems:
    """Test handling of underdetermined regression problems."""

    def test_solve_underdetermined(self, dataset_xy_single):
        """Test solve_underdetermined method."""
        X, Y = dataset_xy_single
        re = RegressionEstimator(X=X, Y=Y)
        re.solve_underdetermined()

        assert re.b is not None
        assert isinstance(re.b, np.ndarray)

    def test_underdetermined_sets_covariances(self, dataset_xy_single):
        """Test that solve_underdetermined sets covariance matrices."""
        X, Y = dataset_xy_single
        re = RegressionEstimator(X=X, Y=Y)
        # Enable return_covariance in iter_control
        re.iter_control.return_covariance = True
        re.solve_underdetermined()

        assert re.cov_nn is not None
        assert re.cov_ss_inv is not None

    def test_underdetermined_covariance_shapes(self, dataset_xy_single):
        """Test covariance matrix shapes for underdetermined system."""
        X, Y = dataset_xy_single
        re = RegressionEstimator(X=X, Y=Y)
        # Enable return_covariance in iter_control
        re.iter_control.return_covariance = True
        re.solve_underdetermined()

        assert re.cov_nn.shape == (re.n_channels_out, re.n_channels_out)
        assert re.cov_ss_inv.shape == (re.n_channels_in, re.n_channels_in)


# =============================================================================
# Test Different Input Types
# =============================================================================


class TestDifferentInputTypes:
    """Test RegressionEstimator with different input data types."""

    def test_xarray_dataset_input(self, dataset_xy_full, expected_solution):
        """Test regression with xarray Dataset input."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        difference = re.b - np.atleast_2d(expected_solution).T
        assert np.allclose(difference, 0)

    def test_xarray_dataarray_input(self, dataset_xy_full, expected_solution):
        """Test regression with xarray DataArray input."""
        X, Y = dataset_xy_full
        X_da = X.to_array()
        Y_da = Y.to_array()

        re = RegressionEstimator(X=X_da, Y=Y_da)
        re.estimate_ols()

        difference = re.b - np.atleast_2d(expected_solution).T
        assert np.allclose(difference, 0)

    def test_numpy_array_input(self, dataset_xy_full, expected_solution):
        """Test regression with numpy array input."""
        X, Y = dataset_xy_full
        X_np = X.to_array().data
        Y_np = Y.to_array().data

        re = RegressionEstimator(X=X_np, Y=Y_np)
        re.estimate_ols()

        difference = re.b - np.atleast_2d(expected_solution).T
        assert np.allclose(difference, 0)

    def test_all_input_types_equivalent(self, dataset_xy_full):
        """Test that all input types produce equivalent results."""
        X, Y = dataset_xy_full

        # Dataset
        re_ds = RegressionEstimator(X=X, Y=Y)
        re_ds.estimate_ols()

        # DataArray
        re_da = RegressionEstimator(X=X.to_array(), Y=Y.to_array())
        re_da.estimate_ols()

        # Numpy
        re_np = RegressionEstimator(X=X.to_array().data, Y=Y.to_array().data)
        re_np.estimate_ols()

        assert np.allclose(re_ds.b, re_da.b)
        assert np.allclose(re_ds.b, re_np.b)


# =============================================================================
# Test xarray Conversion
# =============================================================================


class TestXarrayConversion:
    """Test conversion of results to xarray format."""

    def test_b_to_xarray(self, dataset_xy_full):
        """Test b_to_xarray method."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        xr_result = re.b_to_xarray()

        assert isinstance(xr_result, xr.DataArray)
        assert xr_result is not None

    def test_b_to_xarray_dimensions(self, dataset_xy_full):
        """Test b_to_xarray has correct dimensions."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        xr_result = re.b_to_xarray()

        assert "output_channel" in xr_result.dims
        assert "input_channel" in xr_result.dims

    def test_b_to_xarray_coordinates(self, dataset_xy_full):
        """Test b_to_xarray has correct coordinates."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        xr_result = re.b_to_xarray()

        assert "output_channel" in xr_result.coords
        assert "input_channel" in xr_result.coords
        assert len(xr_result.coords["input_channel"]) == 2
        assert len(xr_result.coords["output_channel"]) == 1

    def test_b_to_xarray_values(self, dataset_xy_full, expected_solution):
        """Test b_to_xarray contains correct values."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        xr_result = re.b_to_xarray()

        # Compare transposed b to xarray values
        assert np.allclose(xr_result.values, re.b.T)


# =============================================================================
# Test Data Validation
# =============================================================================


class TestDataValidation:
    """Test data validation and error handling."""

    def test_mismatched_observations_raises_error(self, mini_dataset_full):
        """Test that mismatched X and Y observations raises an error."""
        X = mini_dataset_full[["hx", "hy"]]
        X = X.stack(observation=("frequency", "time"))

        # Create Y with different number of observations
        Y_short = mini_dataset_full[["ex"]].isel(time=slice(0, 2))
        Y_short = Y_short.stack(observation=("frequency", "time"))

        with pytest.raises(Exception):
            RegressionEstimator(X=X, Y=Y_short)


# =============================================================================
# Test Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability of regression methods."""

    def test_ols_with_synthetic_data(self, simple_regression_data):
        """Test OLS with synthetic data of known solution."""
        X, Y, true_b = simple_regression_data

        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert np.allclose(re.b, true_b, rtol=1e-10)

    def test_large_magnitude_values(self):
        """Test regression with large magnitude values."""
        scale = 1e10
        np.random.seed(101)
        X = np.random.randn(2, 10) * scale + 1j * np.random.randn(2, 10) * scale
        true_b = np.array([[1.0 + 0.5j], [-0.5 + 1.0j]])
        Y = true_b.T @ X

        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert np.allclose(re.b, true_b, rtol=1e-6)

    def test_small_magnitude_values(self):
        """Test regression with small magnitude values."""
        scale = 1e-10
        np.random.seed(102)
        X = np.random.randn(2, 10) * scale + 1j * np.random.randn(2, 10) * scale
        true_b = np.array([[1.0 + 0.5j], [-0.5 + 1.0j]])
        Y = true_b.T @ X

        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert np.allclose(re.b, true_b, rtol=1e-6)

    def test_consistency_across_random_seeds(self, subtests):
        """Test that results are consistent across different random seeds."""
        seeds = [200, 201, 202, 203, 204]
        true_b = np.array([[1.5 + 0.3j], [-0.7 + 0.9j]])

        for seed in seeds:
            with subtests.test(seed=seed):
                np.random.seed(seed)
                X = np.random.randn(2, 15) + 1j * np.random.randn(2, 15)
                Y = true_b.T @ X

                re = RegressionEstimator(X=X, Y=Y)
                re.estimate_ols()

                assert np.allclose(re.b, true_b, rtol=1e-10)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_observations(self):
        """Test with minimum number of observations (n = n_channels_in)."""
        # X should be (n_channels_in, n_observations) = (2, 2)
        X = np.array([[1.0 + 0j, 3.0 + 0j], [2.0 + 0j, 4.0 + 0j]])
        # Y should be (n_channels_out, n_observations) = (1, 2)
        Y = np.array([[5.0 + 1j, 6.0 + 2j]])

        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert re.b is not None
        assert np.all(np.isfinite(re.b))

    def test_single_output_channel(self, dataset_xy_full):
        """Test with single output channel."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert re.n_channels_out == 1
        assert re.b.shape[0] == re.n_channels_in

    def test_real_valued_data(self):
        """Test with real-valued (not complex) data."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Y = np.array([[7.0, 8.0, 9.0]])

        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert re.b is not None
        assert np.all(np.isfinite(re.b))


# =============================================================================
# Test Data Integrity
# =============================================================================


class TestDataIntegrity:
    """Test that regression doesn't modify input data."""

    def test_estimate_preserves_input_X(self, dataset_xy_full):
        """Test that estimation doesn't modify input X."""
        X, Y = dataset_xy_full
        X_orig = X.copy(deep=True)

        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert X.equals(X_orig)

    def test_estimate_preserves_input_Y(self, dataset_xy_full):
        """Test that estimation doesn't modify input Y."""
        X, Y = dataset_xy_full
        Y_orig = Y.copy(deep=True)

        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert Y.equals(Y_orig)

    def test_qr_decomposition_preserves_X(self, regression_estimator):
        """Test that QR decomposition doesn't modify X."""
        X_orig = regression_estimator.X.copy()

        regression_estimator.qr_decomposition()

        assert np.allclose(regression_estimator.X, X_orig)


# =============================================================================
# Test Deterministic Behavior
# =============================================================================


class TestDeterministicBehavior:
    """Test that methods produce deterministic results."""

    def test_estimate_ols_deterministic(self, dataset_xy_full):
        """Test that estimate_ols produces same result on repeated calls."""
        X, Y = dataset_xy_full

        results = []
        for _ in range(5):
            re = RegressionEstimator(X=X, Y=Y)
            re.estimate_ols()
            results.append(re.b.copy())

        for result in results[1:]:
            assert np.allclose(result, results[0])

    def test_qr_decomposition_deterministic(self, dataset_xy_full):
        """Test that QR decomposition is deterministic."""
        X, Y = dataset_xy_full

        re = RegressionEstimator(X=X, Y=Y)
        Q1, R1 = re.qr_decomposition()
        Q2, R2 = re.qr_decomposition(re.X)

        assert np.allclose(Q1, Q2)
        assert np.allclose(R1, R2)


# =============================================================================
# Test Mathematical Properties
# =============================================================================


class TestMathematicalProperties:
    """Test mathematical properties of regression."""

    def test_residual_minimization(self, simple_regression_data):
        """Test that OLS minimizes the residual."""
        X, Y, _ = simple_regression_data

        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        # Compute residual
        Y_pred = re.b.T @ X
        residual = Y - Y_pred

        # For exact case (no noise), residual should be near zero
        assert np.linalg.norm(residual) < 1e-10

    def test_solution_shape(self, dataset_xy_full):
        """Test that solution has correct shape."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert re.b.shape == (re.n_channels_in, re.n_channels_out)

    def test_qr_orthogonality(self, regression_estimator):
        """Test Q matrix orthogonality from QR decomposition."""
        Q, _ = regression_estimator.qr_decomposition()

        # Q should satisfy Q^H @ Q = I
        QHQ = Q.conj().T @ Q
        identity = np.eye(Q.shape[1])

        assert np.allclose(QHQ, identity, atol=1e-10)


# =============================================================================
# Test Return Values
# =============================================================================


class TestReturnValues:
    """Test characteristics of return values."""

    def test_b_is_finite(self, dataset_xy_full):
        """Test that regression solution b contains finite values."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert np.all(np.isfinite(re.b))

    def test_b_is_complex(self, dataset_xy_full):
        """Test that regression solution b is complex."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert np.iscomplexobj(re.b)

    def test_b_not_all_zero(self, dataset_xy_full):
        """Test that regression solution b is not all zeros."""
        X, Y = dataset_xy_full
        re = RegressionEstimator(X=X, Y=Y)
        re.estimate_ols()

        assert not np.allclose(re.b, 0)
