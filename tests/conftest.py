"""Minimal conftest for aurora tests that need small mth5 fixtures.

This provides a small, self-contained subset of the mth5 test fixtures
so aurora tests can create and use `test12rr` MTH5 files without depending
on the mth5 repo's conftest discovery.

Fixtures provided:
- `worker_id` : pytest-xdist aware worker id
- `make_worker_safe_path(base, directory)` : make worker-unique filenames
- `fresh_test12rr_mth5` : creates a fresh `test12rr` MTH5 file in `tmp_path`
- `cleanup_test_files` : register files to be removed at session end
"""

import uuid
import warnings
from pathlib import Path
from typing import Dict

import pytest
from mth5.data.make_mth5_from_asc import create_test12rr_h5


# Suppress noisy third-party deprecation and pydantic serializer warnings
# that are not actionable in these tests. These originate from external
# dependencies (jupyter_client, obspy/pkg_resources) and from pydantic when
# receiving plain strings where enums are expected. Filtering here keeps test
# output focused on real failures.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Pydantic serializer warnings:.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Jupyter is migrating its paths to use standard platformdirs",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"pkg_resources",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"np\.bool",
)


# Process-wide cache for heavyweight test artifacts (keyed by worker id)
# stores the created MTH5 file path so multiple tests in the same session
# / worker can reuse the same file rather than recreating it repeatedly.
_MTH5_GLOBAL_CACHE: Dict[str, str] = {}


@pytest.fixture(scope="session")
def worker_id(request):
    """Return pytest-xdist worker id or 'master' when not using xdist."""
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput.get("workerid", "gw0")
    return "master"


def get_worker_safe_filename(base_filename: str, worker: str) -> str:
    p = Path(base_filename)
    return f"{p.stem}_{worker}{p.suffix}"


@pytest.fixture
def make_worker_safe_path(worker_id):
    """Factory to produce worker-safe paths.

    Usage: `p = make_worker_safe_path('name.zrr', tmp_path)`
    """

    def _make(base_filename: str, directory: Path | None = None) -> Path:
        safe_name = get_worker_safe_filename(base_filename, worker_id)
        if directory is None:
            return Path(safe_name)
        return Path(directory) / safe_name

    return _make


@pytest.fixture(scope="session")
def cleanup_test_files(request):
    files = []

    def _register(p: Path):
        if p not in files:
            files.append(p)

    def _cleanup():
        for p in files:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                # best-effort cleanup
                pass

    request.addfinalizer(_cleanup)
    return _register


@pytest.fixture
def fresh_test12rr_mth5(tmp_path: Path, worker_id, cleanup_test_files):
    """Create a fresh `test12rr` MTH5 file in tmp_path and return its Path.

    This is intentionally simple: it calls `create_test12rr_h5` with a
    temporary target folder. The resulting file is registered for cleanup.
    """
    cache_key = f"test12rr_{worker_id}"

    # Return cached file if present and still exists
    cached = _MTH5_GLOBAL_CACHE.get(cache_key)
    if cached:
        p = Path(cached)
        if p.exists():
            return p

    # create a unique folder for this worker/test
    unique_dir = tmp_path / f"mth5_test12rr_{worker_id}_{uuid.uuid4().hex[:8]}"
    unique_dir.mkdir(parents=True, exist_ok=True)

    # create_test12rr_h5 returns the path to the file it created
    file_path = create_test12rr_h5(target_folder=unique_dir)

    # register cleanup and cache
    ppath = Path(file_path)
    cleanup_test_files(ppath)
    _MTH5_GLOBAL_CACHE[cache_key] = str(ppath)

    return ppath
