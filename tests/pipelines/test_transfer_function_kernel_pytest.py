"""Pytest translation of `test_transfer_function_kernel.py`.

Uses fixtures and subtests. Designed to be xdist-safe by avoiding global
state and using fixtures from `conftest.py` where appropriate.
"""

from __future__ import annotations

import pytest

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.transfer_function_kernel import TransferFunctionKernel
from aurora.test_utils.synthetic.processing_helpers import get_example_kernel_dataset


@pytest.fixture
def kernel_dataset():
    return get_example_kernel_dataset()


@pytest.fixture
def processing_config(kernel_dataset):
    cc = ConfigCreator()
    return cc.create_from_kernel_dataset(kernel_dataset, estimator={"engine": "RME"})


@pytest.fixture
def tfk(kernel_dataset, processing_config):
    return TransferFunctionKernel(dataset=kernel_dataset, config=processing_config)


def test_init(tfk):
    """Constructing a TransferFunctionKernel with a valid config succeeds."""
    assert isinstance(tfk, TransferFunctionKernel)


def test_cannot_init_without_processing_config():
    """Calling constructor with no args raises TypeError (same as original)."""
    with pytest.raises(TypeError):
        TransferFunctionKernel()


def test_tfk_basic_properties(tfk, subtests):
    """A few lightweight sanity checks using subtests for clearer output."""
    with subtests.test("has_dataset"):
        assert getattr(tfk, "dataset", None) is not None

    with subtests.test("has_config"):
        assert getattr(tfk, "config", None) is not None

    with subtests.test("string_repr"):
        # Ensure a simple repr/str path doesn't error; not asserting exact
        # content since it may change between implementations.
        s = str(tfk)
        assert isinstance(s, str)
