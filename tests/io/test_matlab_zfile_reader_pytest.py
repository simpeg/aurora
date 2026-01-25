"""
Pytest suite for MATLAB Z-file reader functionality.

Tests reading and parsing MATLAB Z-files for different case IDs.
"""

import pytest

from aurora.sandbox.io_helpers.garys_matlab_zfiles.matlab_z_file_reader import (
    test_matlab_zfile_reader,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=["IAK34ss", "synthetic"])
def case_id(request):
    """
    Provide case IDs for MATLAB Z-file reader tests.

    Parameters:
    - IAK34ss: Real data case
    - synthetic: Synthetic data case
    """
    return request.param


@pytest.fixture
def iak34ss_case_id():
    """Fixture for IAK34ss case ID (real data)."""
    return "IAK34ss"


@pytest.fixture
def synthetic_case_id():
    """Fixture for synthetic case ID."""
    return "synthetic"


# =============================================================================
# Tests
# =============================================================================


def test_matlab_zfile_reader_iak34ss(iak34ss_case_id):
    """Test MATLAB Z-file reader with IAK34ss real data case."""
    test_matlab_zfile_reader(case_id=iak34ss_case_id)


@pytest.mark.skip(reason="Synthetic case currently disabled in original test")
def test_matlab_zfile_reader_synthetic(synthetic_case_id):
    """Test MATLAB Z-file reader with synthetic data case."""
    test_matlab_zfile_reader(case_id=synthetic_case_id)


@pytest.mark.parametrize("test_case_id", ["IAK34ss"])
def test_matlab_zfile_reader_parametrized(test_case_id):
    """
    Parametrized test for MATLAB Z-file reader.

    This test runs for each case ID in the parametrize decorator.
    To enable synthetic test, add "synthetic" to the parametrize list.
    """
    test_matlab_zfile_reader(case_id=test_case_id)


class TestMatlabZFileReader:
    """Test class for MATLAB Z-file reader functionality."""

    def test_iak34ss_case(self):
        """Test reading IAK34ss MATLAB Z-file."""
        test_matlab_zfile_reader(case_id="IAK34ss")

    @pytest.mark.skip(reason="Synthetic case needs verification")
    def test_synthetic_case(self):
        """Test reading synthetic MATLAB Z-file."""
        test_matlab_zfile_reader(case_id="synthetic")


# =============================================================================
# Integration Tests
# =============================================================================


class TestMatlabZFileReaderIntegration:
    """Integration tests for MATLAB Z-file reader."""

    @pytest.mark.parametrize(
        "case_id,description",
        [
            ("IAK34ss", "Real data from IAK34ss station"),
            # ("synthetic", "Synthetic test data"),  # Uncomment to enable
        ],
        ids=["IAK34ss"],  # Add "synthetic" when uncommenting above
    )
    def test_reader_with_description(self, case_id, description):
        """
        Test MATLAB Z-file reader with case descriptions.

        Parameters
        ----------
        case_id : str
            The case identifier for the MATLAB Z-file
        description : str
            Human-readable description of the test case
        """
        # Log the test case being run
        print(f"\nTesting case: {case_id} - {description}")
        test_matlab_zfile_reader(case_id=case_id)


# =============================================================================
# Backward Compatibility
# =============================================================================


def test():
    """
    Legacy test function for backward compatibility.

    This maintains the original test interface from test_matlab_zfile_reader.py
    """
    test_matlab_zfile_reader(case_id="IAK34ss")


if __name__ == "__main__":
    # Run pytest on this file
    pytest.main([__file__, "-v"])
