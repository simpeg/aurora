# -*- coding: utf-8 -*-
"""
Pytest suite for regression helper_functions module.

Tests transfer function regression methods using fixtures and subtests.
Optimized for pytest-xdist parallel execution.
"""

import numpy as np
import pytest

from aurora.transfer_function.regression.helper_functions import (
    direct_solve_tf,
    rme_beta,
    simple_solve_tf,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def sample_electric_data():
    """Sample electric field data for testing."""
    return np.array(
        [
            4.39080123e-07 - 2.41097397e-06j,
            -2.33418464e-06 + 2.10752581e-06j,
            1.38642624e-06 - 1.87333571e-06j,
        ]
    )


@pytest.fixture(scope="module")
def sample_magnetic_data():
    """Sample magnetic field data for testing."""
    return np.array(
        [
            [7.00767250e-07 - 9.18819198e-07j, 1.94321684e-07 + 3.71934877e-07j],
            [-1.06648904e-07 + 8.19420154e-07j, 1.15361101e-08 - 6.32581646e-07j],
            [-1.02700963e-07 - 3.73904463e-07j, 3.86095787e-08 + 4.33155345e-07j],
        ]
    )


@pytest.fixture(scope="module")
def expected_solution():
    """Expected transfer function solution for sample data."""
    return np.array([-0.04192569 - 0.36502722j, -3.65284496 - 4.05194938j])


@pytest.fixture(scope="module")
def simple_2x2_system():
    """Simple 2x2 system for basic testing."""
    X = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
    Y = np.array([2.0 + 1j, 3.0 - 2j])
    expected = Y.copy()
    return X, Y, expected


@pytest.fixture(scope="module")
def overdetermined_system():
    """Overdetermined system (more equations than unknowns)."""
    np.random.seed(42)
    X = np.random.randn(10, 2) + 1j * np.random.randn(10, 2)
    true_tf = np.array([1.5 + 0.5j, -0.8 + 1.2j])
    Y = X @ true_tf
    return X, Y, true_tf


@pytest.fixture(scope="module")
def remote_reference_data():
    """Data with remote reference channels."""
    np.random.seed(43)
    X = np.random.randn(5, 2) + 1j * np.random.randn(5, 2)
    R = np.random.randn(5, 2) + 1j * np.random.randn(5, 2)
    true_tf = np.array([2.0 + 0j, -1.0 + 0.5j])
    Y = X @ true_tf
    return X, Y, R, true_tf


# =============================================================================
# Test RME Beta Function
# =============================================================================


class TestRMEBeta:
    """Test the rme_beta correction factor function."""

    def test_rme_beta_standard_value(self):
        """Test rme_beta with standard r0=1.5."""
        beta = rme_beta(1.5)
        # For r0=1.5, beta should be approximately 0.78
        assert isinstance(beta, (float, np.floating))
        assert 0.75 < beta < 0.80
        # More precise check
        expected = 1.0 - np.exp(-1.5)
        assert np.isclose(beta, expected)

    def test_rme_beta_zero(self):
        """Test rme_beta with r0=0."""
        beta = rme_beta(0.0)
        # For r0=0, beta = 1 - exp(0) = 1 - 1 = 0
        assert np.isclose(beta, 0.0)

    def test_rme_beta_large_value(self):
        """Test rme_beta with large r0."""
        beta = rme_beta(10.0)
        # For large r0, beta approaches 1.0
        assert isinstance(beta, (float, np.floating))
        assert beta > 0.99
        expected = 1.0 - np.exp(-10.0)
        assert np.isclose(beta, expected)

    def test_rme_beta_small_value(self):
        """Test rme_beta with small positive r0."""
        beta = rme_beta(0.1)
        expected = 1.0 - np.exp(-0.1)
        assert np.isclose(beta, expected)
        # Small r0 should give small beta
        assert 0.0 < beta < 0.1

    def test_rme_beta_range_values(self, subtests):
        """Test rme_beta across a range of r0 values."""
        r0_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

        for r0 in r0_values:
            with subtests.test(r0=r0):
                beta = rme_beta(r0)
                expected = 1.0 - np.exp(-r0)
                assert np.isclose(beta, expected)
                # Beta should always be in [0, 1)
                assert 0.0 <= beta < 1.0

    def test_rme_beta_monotonic(self):
        """Test that rme_beta is monotonically increasing."""
        r0_values = np.linspace(0, 5, 20)
        beta_values = [rme_beta(r0) for r0 in r0_values]

        # Check that each value is greater than or equal to previous
        for i in range(1, len(beta_values)):
            assert beta_values[i] >= beta_values[i - 1]

    def test_rme_beta_asymptotic_behavior(self):
        """Test that rme_beta approaches 1.0 asymptotically."""
        large_r0 = 100.0
        beta = rme_beta(large_r0)
        assert np.isclose(beta, 1.0, rtol=1e-10)


# =============================================================================
# Test Simple Solve TF
# =============================================================================


class TestSimpleSolveTF:
    """Test the simple_solve_tf function."""

    def test_simple_solve_tf_sample_data(
        self, sample_electric_data, sample_magnetic_data, expected_solution
    ):
        """Test simple_solve_tf with provided sample data."""
        z = simple_solve_tf(sample_electric_data, sample_magnetic_data)
        assert np.allclose(z, expected_solution, rtol=1e-8)

    def test_simple_solve_tf_identity_system(self, simple_2x2_system):
        """Test simple_solve_tf with identity-like system."""
        X, Y, expected = simple_2x2_system
        z = simple_solve_tf(Y, X)
        assert np.allclose(z, expected, rtol=1e-10)

    def test_simple_solve_tf_overdetermined(self, overdetermined_system):
        """Test simple_solve_tf with overdetermined system."""
        X, Y, true_tf = overdetermined_system
        z = simple_solve_tf(Y, X)
        # Should recover the true TF exactly (no noise added)
        assert np.allclose(z, true_tf, rtol=1e-10)

    def test_simple_solve_tf_with_remote_reference(self, remote_reference_data):
        """Test simple_solve_tf with remote reference."""
        X, Y, R, true_tf = remote_reference_data
        # Using remote reference R instead of X for conjugate transpose
        z = simple_solve_tf(Y, X, R=R)

        # Result depends on R, not necessarily equal to true_tf
        assert z.shape == true_tf.shape
        assert np.all(np.isfinite(z))

    def test_simple_solve_tf_return_type(
        self, sample_electric_data, sample_magnetic_data
    ):
        """Test that simple_solve_tf returns numpy array."""
        z = simple_solve_tf(sample_electric_data, sample_magnetic_data)
        assert isinstance(z, np.ndarray)
        assert z.dtype == np.complex128 or z.dtype == np.complex64

    def test_simple_solve_tf_shape(self, sample_electric_data, sample_magnetic_data):
        """Test that simple_solve_tf returns correct shape."""
        z = simple_solve_tf(sample_electric_data, sample_magnetic_data)
        # Should return 2 elements for 2-column input
        assert z.shape == (2,)

    def test_simple_solve_tf_no_remote_reference(
        self, sample_electric_data, sample_magnetic_data
    ):
        """Test simple_solve_tf explicitly with R=None."""
        z1 = simple_solve_tf(sample_electric_data, sample_magnetic_data)
        z2 = simple_solve_tf(sample_electric_data, sample_magnetic_data, R=None)
        assert np.allclose(z1, z2)


# =============================================================================
# Test Direct Solve TF
# =============================================================================


class TestDirectSolveTF:
    """Test the direct_solve_tf function."""

    def test_direct_solve_tf_sample_data(
        self, sample_electric_data, sample_magnetic_data, expected_solution
    ):
        """Test direct_solve_tf with provided sample data."""
        z = direct_solve_tf(sample_electric_data, sample_magnetic_data)
        assert np.allclose(z, expected_solution, rtol=1e-8)

    def test_direct_solve_tf_identity_system(self, simple_2x2_system):
        """Test direct_solve_tf with identity-like system."""
        X, Y, expected = simple_2x2_system
        z = direct_solve_tf(Y, X)
        assert np.allclose(z, expected, rtol=1e-10)

    def test_direct_solve_tf_overdetermined(self, overdetermined_system):
        """Test direct_solve_tf with overdetermined system."""
        X, Y, true_tf = overdetermined_system
        z = direct_solve_tf(Y, X)
        # Should recover the true TF exactly (no noise added)
        assert np.allclose(z, true_tf, rtol=1e-10)

    def test_direct_solve_tf_with_remote_reference(self, remote_reference_data):
        """Test direct_solve_tf with remote reference."""
        X, Y, R, true_tf = remote_reference_data
        # Using remote reference R instead of X for conjugate transpose
        z = direct_solve_tf(Y, X, R=R)

        # Result depends on R, not necessarily equal to true_tf
        assert z.shape == true_tf.shape
        assert np.all(np.isfinite(z))

    def test_direct_solve_tf_return_type(
        self, sample_electric_data, sample_magnetic_data
    ):
        """Test that direct_solve_tf returns numpy array."""
        z = direct_solve_tf(sample_electric_data, sample_magnetic_data)
        assert isinstance(z, np.ndarray)
        assert z.dtype == np.complex128 or z.dtype == np.complex64

    def test_direct_solve_tf_shape(self, sample_electric_data, sample_magnetic_data):
        """Test that direct_solve_tf returns correct shape."""
        z = direct_solve_tf(sample_electric_data, sample_magnetic_data)
        # Should return 2 elements for 2-column input
        assert z.shape == (2,)

    def test_direct_solve_tf_no_remote_reference(
        self, sample_electric_data, sample_magnetic_data
    ):
        """Test direct_solve_tf explicitly with R=None."""
        z1 = direct_solve_tf(sample_electric_data, sample_magnetic_data)
        z2 = direct_solve_tf(sample_electric_data, sample_magnetic_data, R=None)
        assert np.allclose(z1, z2)


# =============================================================================
# Test Equivalence Between Methods
# =============================================================================


class TestMethodEquivalence:
    """Test that simple_solve_tf and direct_solve_tf produce equivalent results."""

    def test_methods_equivalent_sample_data(
        self, sample_electric_data, sample_magnetic_data
    ):
        """Test that both methods give same result on sample data."""
        z_simple = simple_solve_tf(sample_electric_data, sample_magnetic_data)
        z_direct = direct_solve_tf(sample_electric_data, sample_magnetic_data)
        assert np.allclose(z_simple, z_direct, rtol=1e-10)

    def test_methods_equivalent_identity(self, simple_2x2_system):
        """Test that both methods give same result on identity system."""
        X, Y, _ = simple_2x2_system
        z_simple = simple_solve_tf(Y, X)
        z_direct = direct_solve_tf(Y, X)
        assert np.allclose(z_simple, z_direct, rtol=1e-10)

    def test_methods_equivalent_overdetermined(self, overdetermined_system):
        """Test that both methods give same result on overdetermined system."""
        X, Y, _ = overdetermined_system
        z_simple = simple_solve_tf(Y, X)
        z_direct = direct_solve_tf(Y, X)
        assert np.allclose(z_simple, z_direct, rtol=1e-10)

    def test_methods_equivalent_with_remote(self, remote_reference_data):
        """Test that both methods give same result with remote reference."""
        X, Y, R, _ = remote_reference_data
        z_simple = simple_solve_tf(Y, X, R=R)
        z_direct = direct_solve_tf(Y, X, R=R)
        assert np.allclose(z_simple, z_direct, rtol=1e-10)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_equation_system(self):
        """Test with minimum size system (1 equation, but need at least 2 for 2 unknowns)."""
        # Actually need at least 2 equations for 2 unknowns
        X = np.array([[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]])
        Y = np.array([5.0 + 1j, 6.0 + 2j])

        z_simple = simple_solve_tf(Y, X)
        z_direct = direct_solve_tf(Y, X)

        # Both should produce valid results
        assert np.all(np.isfinite(z_simple))
        assert np.all(np.isfinite(z_direct))
        assert np.allclose(z_simple, z_direct)

    def test_real_valued_inputs(self):
        """Test with real-valued (not complex) inputs."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Y = np.array([7.0, 8.0, 9.0])

        z_simple = simple_solve_tf(Y, X)
        z_direct = direct_solve_tf(Y, X)

        assert np.all(np.isfinite(z_simple))
        assert np.all(np.isfinite(z_direct))
        assert np.allclose(z_simple, z_direct)

    def test_complex_phases(self, subtests):
        """Test with various complex phase relationships."""
        phases = [0, np.pi / 4, np.pi / 2, np.pi]

        for phase in phases:
            with subtests.test(phase=phase):
                X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) * np.exp(1j * phase)
                Y = np.array([1.0, 2.0, 3.0]) * np.exp(1j * (phase + np.pi / 6))

                z_simple = simple_solve_tf(Y, X)
                z_direct = direct_solve_tf(Y, X)

                assert np.all(np.isfinite(z_simple))
                assert np.all(np.isfinite(z_direct))
                assert np.allclose(z_simple, z_direct)

    def test_large_magnitude_values(self):
        """Test with very large magnitude values."""
        scale = 1e10
        X = np.array([[1.0 + 1j, 2.0 - 1j], [3.0 + 0j, 4.0 + 2j]]) * scale
        Y = np.array([5.0 + 1j, 6.0 - 2j]) * scale

        z_simple = simple_solve_tf(Y, X)
        z_direct = direct_solve_tf(Y, X)

        assert np.all(np.isfinite(z_simple))
        assert np.all(np.isfinite(z_direct))
        assert np.allclose(z_simple, z_direct, rtol=1e-6)

    def test_small_magnitude_values(self):
        """Test with very small magnitude values."""
        scale = 1e-10
        X = np.array([[1.0 + 1j, 2.0 - 1j], [3.0 + 0j, 4.0 + 2j]]) * scale
        Y = np.array([5.0 + 1j, 6.0 - 2j]) * scale

        z_simple = simple_solve_tf(Y, X)
        z_direct = direct_solve_tf(Y, X)

        assert np.all(np.isfinite(z_simple))
        assert np.all(np.isfinite(z_direct))
        assert np.allclose(z_simple, z_direct, rtol=1e-6)


# =============================================================================
# Test Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability of the solvers."""

    def test_well_conditioned_system(self):
        """Test with a well-conditioned system."""
        np.random.seed(44)
        # Create well-conditioned matrix
        X = np.random.randn(10, 2) + 1j * np.random.randn(10, 2)
        X[:, 0] = X[:, 0] / np.linalg.norm(X[:, 0])
        X[:, 1] = X[:, 1] / np.linalg.norm(X[:, 1])

        true_tf = np.array([1.0 + 0.5j, -0.5 + 1.0j])
        Y = X @ true_tf

        z_simple = simple_solve_tf(Y, X)
        z_direct = direct_solve_tf(Y, X)

        assert np.allclose(z_simple, true_tf, rtol=1e-10)
        assert np.allclose(z_direct, true_tf, rtol=1e-10)

    def test_orthogonal_columns(self):
        """Test with orthogonal column vectors."""
        # Create orthogonal columns
        X = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=complex)
        Y = np.array([2.0 + 1j, 3.0 - 2j, 0.0])

        z_simple = simple_solve_tf(Y, X)
        z_direct = direct_solve_tf(Y, X)

        # For orthogonal X, solution should be straightforward
        assert np.allclose(z_simple, z_direct)
        assert np.allclose(z_simple[0], 2.0 + 1j)
        assert np.allclose(z_simple[1], 3.0 - 2j)

    def test_consistency_across_seeds(self, subtests):
        """Test that results are consistent across different random seeds."""
        seeds = [10, 20, 30, 40, 50]

        for seed in seeds:
            with subtests.test(seed=seed):
                np.random.seed(seed)
                X = np.random.randn(8, 2) + 1j * np.random.randn(8, 2)
                true_tf = np.array([1.0 + 1.0j, -1.0 + 1.0j])
                Y = X @ true_tf

                z_simple = simple_solve_tf(Y, X)
                z_direct = direct_solve_tf(Y, X)

                assert np.allclose(z_simple, true_tf, rtol=1e-10)
                assert np.allclose(z_direct, true_tf, rtol=1e-10)
                assert np.allclose(z_simple, z_direct)


# =============================================================================
# Test Data Integrity
# =============================================================================


class TestDataIntegrity:
    """Test that functions don't modify input data."""

    def test_simple_solve_tf_preserves_inputs(
        self, sample_electric_data, sample_magnetic_data
    ):
        """Test that simple_solve_tf doesn't modify input arrays."""
        Y_orig = sample_electric_data.copy()
        X_orig = sample_magnetic_data.copy()

        simple_solve_tf(sample_electric_data, sample_magnetic_data)

        assert np.allclose(sample_electric_data, Y_orig)
        assert np.allclose(sample_magnetic_data, X_orig)

    def test_direct_solve_tf_preserves_inputs(
        self, sample_electric_data, sample_magnetic_data
    ):
        """Test that direct_solve_tf doesn't modify input arrays."""
        Y_orig = sample_electric_data.copy()
        X_orig = sample_magnetic_data.copy()

        direct_solve_tf(sample_electric_data, sample_magnetic_data)

        assert np.allclose(sample_electric_data, Y_orig)
        assert np.allclose(sample_magnetic_data, X_orig)

    def test_remote_reference_preserved(self, remote_reference_data):
        """Test that remote reference array is not modified."""
        X, Y, R, _ = remote_reference_data
        R_orig = R.copy()

        simple_solve_tf(Y, X, R=R)
        direct_solve_tf(Y, X, R=R)

        assert np.allclose(R, R_orig)


# =============================================================================
# Test Mathematical Properties
# =============================================================================


class TestMathematicalProperties:
    """Test mathematical properties of the regression."""

    def test_linearity(self):
        """Test that the solution is linear in Y."""
        X = np.array([[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]])
        Y1 = np.array([1.0 + 1j, 2.0 + 2j])
        Y2 = np.array([3.0 - 1j, 4.0 - 2j])

        z1 = simple_solve_tf(Y1, X)
        z2 = simple_solve_tf(Y2, X)
        z_sum = simple_solve_tf(Y1 + Y2, X)

        # Solution should be linear: z(Y1 + Y2) = z(Y1) + z(Y2)
        assert np.allclose(z_sum, z1 + z2, rtol=1e-10)

    def test_scaling_property(self):
        """Test that scaling Y scales the solution proportionally."""
        X = np.array([[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]])
        Y = np.array([1.0 + 1j, 2.0 + 2j])
        scale = 5.0 + 3j

        z1 = simple_solve_tf(Y, X)
        z2 = simple_solve_tf(scale * Y, X)

        # Scaling Y should scale the solution
        assert np.allclose(z2, scale * z1, rtol=1e-10)

    def test_residual_minimization(self):
        """Test that the solution minimizes the residual in least squares sense."""
        np.random.seed(45)
        X = np.random.randn(10, 2) + 1j * np.random.randn(10, 2)
        true_tf = np.array([1.0 + 0.5j, -0.5 + 1.0j])
        Y = X @ true_tf

        z = simple_solve_tf(Y, X)
        residual = Y - X @ z

        # Residual should be very small (near zero for exact case)
        assert np.linalg.norm(residual) < 1e-10

    def test_conjugate_transpose_property(self):
        """Test the conjugate transpose operations in the formulation."""
        X = np.array([[1.0 + 1j, 2.0 - 1j], [3.0 + 0j, 4.0 + 2j]])
        Y = np.array([5.0 + 1j, 6.0 - 2j])

        # Verify that X^H @ X is Hermitian
        xH = X.conjugate().transpose()
        xHx = xH @ X

        assert np.allclose(xHx, xHx.conj().T, rtol=1e-10)


# =============================================================================
# Test Return Value Characteristics
# =============================================================================


class TestReturnValues:
    """Test characteristics of return values."""

    def test_return_value_finite(self, sample_electric_data, sample_magnetic_data):
        """Test that return values are finite."""
        z_simple = simple_solve_tf(sample_electric_data, sample_magnetic_data)
        z_direct = direct_solve_tf(sample_electric_data, sample_magnetic_data)

        assert np.all(np.isfinite(z_simple))
        assert np.all(np.isfinite(z_direct))

    def test_return_value_complex(self, sample_electric_data, sample_magnetic_data):
        """Test that return values are complex."""
        z_simple = simple_solve_tf(sample_electric_data, sample_magnetic_data)
        z_direct = direct_solve_tf(sample_electric_data, sample_magnetic_data)

        assert np.iscomplexobj(z_simple)
        assert np.iscomplexobj(z_direct)

    def test_return_value_not_all_zero(
        self, sample_electric_data, sample_magnetic_data
    ):
        """Test that return values are not all zero."""
        z_simple = simple_solve_tf(sample_electric_data, sample_magnetic_data)
        z_direct = direct_solve_tf(sample_electric_data, sample_magnetic_data)

        assert not np.allclose(z_simple, 0)
        assert not np.allclose(z_direct, 0)


# =============================================================================
# Test Deterministic Behavior
# =============================================================================


class TestDeterministicBehavior:
    """Test that functions produce deterministic results."""

    def test_simple_solve_tf_deterministic(
        self, sample_electric_data, sample_magnetic_data
    ):
        """Test that simple_solve_tf produces same result on repeated calls."""
        results = [
            simple_solve_tf(sample_electric_data, sample_magnetic_data)
            for _ in range(5)
        ]

        for result in results[1:]:
            assert np.allclose(result, results[0])

    def test_direct_solve_tf_deterministic(
        self, sample_electric_data, sample_magnetic_data
    ):
        """Test that direct_solve_tf produces same result on repeated calls."""
        results = [
            direct_solve_tf(sample_electric_data, sample_magnetic_data)
            for _ in range(5)
        ]

        for result in results[1:]:
            assert np.allclose(result, results[0])

    def test_rme_beta_deterministic(self):
        """Test that rme_beta produces same result on repeated calls."""
        r0 = 1.5
        results = [rme_beta(r0) for _ in range(10)]

        for result in results[1:]:
            assert result == results[0]
