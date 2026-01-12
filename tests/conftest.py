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

# Set non-interactive matplotlib backend before any other imports
# This prevents tests from blocking on figure windows
import matplotlib


matplotlib.use("Agg")

from pathlib import Path
from typing import Dict

import pytest
from mt_metadata.transfer_functions.core import TF as _MT_TF
from mth5.data.make_mth5_from_asc import (
    create_test1_h5,
    create_test2_h5,
    create_test3_h5,
    create_test12rr_h5,
)
from mth5.helpers import close_open_files

from aurora.test_utils.synthetic.paths import SyntheticTestPaths


# Monkeypatch TF.write to sanitize None provenance/comment fields that cause
# pydantic validation errors when writing certain formats (e.g., emtfxml).
_orig_tf_write = getattr(_MT_TF, "write", None)


def _safe_tf_write(self, *args, **kwargs):
    # Pre-emptively sanitize station provenance comments to avoid pydantic errors
    try:
        sm = getattr(self, "station_metadata", None)
        if sm is not None:
            # Handle dict-based metadata (from pydantic branch)
            if isinstance(sm, dict):
                prov = sm.get("provenance")
                if prov and isinstance(prov, dict):
                    archive = prov.get("archive")
                    if archive and isinstance(archive, dict):
                        comments = archive.get("comments")
                        if comments and isinstance(comments, dict):
                            if comments.get("value") is None:
                                comments["value"] = ""
            else:
                # Handle object-based metadata (traditional approach)
                sm_list = (
                    sm if hasattr(sm, "__iter__") and not isinstance(sm, str) else [sm]
                )
                for s in sm_list:
                    try:
                        prov = getattr(s, "provenance", None)
                        if prov is None:
                            continue
                        archive = getattr(prov, "archive", None)
                        if archive is None:
                            continue
                        comments = getattr(archive, "comments", None)
                        if comments is None:
                            from types import SimpleNamespace

                            archive.comments = SimpleNamespace(value="")
                        elif getattr(comments, "value", None) is None:
                            comments.value = ""
                    except Exception:
                        pass
    except Exception:
        pass
    # Call original write
    return _orig_tf_write(self, *args, **kwargs)


if _orig_tf_write is not None:
    setattr(_MT_TF, "write", _safe_tf_write)


# Suppress noisy third-party deprecation and pydantic serializer warnings
# that are not actionable in these tests. These originate from external
# dependencies (jupyter_client, obspy/pkg_resources) and from pydantic when
# receiving plain strings where enums are expected. Filtering here keeps test
# output focused on real failures.
# warnings.filterwarnings(
#     "ignore",
#     category=UserWarning,
#     message=r"Pydantic serializer warnings:.*",
# )
# warnings.filterwarnings(
#     "ignore",
#     category=DeprecationWarning,
#     message=r"Jupyter is migrating its paths to use standard platformdirs",
# )
# warnings.filterwarnings(
#     "ignore",
#     category=DeprecationWarning,
#     message=r"pkg_resources",
# )
# warnings.filterwarnings(
#     "ignore",
#     category=DeprecationWarning,
#     message=r"np\.bool",
# )


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
def synthetic_test_paths(tmp_path_factory, worker_id):
    """Create a SyntheticTestPaths instance that writes into a worker-unique tmp sandbox.

    This keeps tests isolated across xdist workers and avoids writing into the repo.
    """
    base = tmp_path_factory.mktemp(f"synthetic_{worker_id}")
    stp = SyntheticTestPaths(sandbox_path=base)
    stp.mkdirs()
    return stp


@pytest.fixture(autouse=True)
def ensure_closed_files():
    """Ensure mth5 open files are closed before/after each test to avoid cross-test leaks."""
    # run before test
    close_open_files()
    yield
    # run after test
    close_open_files()


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


@pytest.fixture(scope="session")
def fresh_test12rr_mth5(mth5_target_dir: Path, worker_id, cleanup_test_files):
    """Create a fresh `test12rr` MTH5 file in mth5_target_dir and return its Path.

    This is intentionally simple: it calls `create_test12rr_h5` with a
    temporary target folder. The resulting file is registered for cleanup.
    Session-scoped for efficiency.
    """
    cache_key = f"test12rr_{worker_id}"

    # Return cached file if present and still exists
    cached = _MTH5_GLOBAL_CACHE.get(cache_key)
    if cached:
        p = Path(cached)
        if p.exists():
            return p

    # create_test12rr_h5 returns the path to the file it created
    # Use the session-scoped mth5_target_dir
    file_path = create_test12rr_h5(target_folder=mth5_target_dir)

    # register cleanup and cache
    ppath = Path(file_path)
    cleanup_test_files(ppath)
    _MTH5_GLOBAL_CACHE[cache_key] = str(ppath)

    return ppath


@pytest.fixture(scope="session")
def mth5_target_dir(tmp_path_factory, worker_id):
    """Create a worker-safe directory for MTH5 file creation.

    This directory is shared across all tests in a worker session,
    allowing MTH5 files to be cached and reused within a worker.
    """
    base_dir = tmp_path_factory.mktemp(f"mth5_files_{worker_id}")
    return base_dir


def _create_worker_safe_mth5(
    mth5_name: str,
    create_func,
    target_dir: Path,
    worker_id: str,
    file_version: str = "0.1.0",
    channel_nomenclature: str = "default",
    **kwargs,
) -> Path:
    """Helper to create worker-safe MTH5 files with caching.

    Parameters
    ----------
    mth5_name : str
        Base name for the MTH5 file (e.g., "test1", "test2")
    create_func : callable
        Function to create the MTH5 file (e.g., create_test1_h5)
    target_dir : Path
        Directory where the MTH5 file should be created
    worker_id : str
        Worker ID for pytest-xdist
    file_version : str
        MTH5 file version
    channel_nomenclature : str
        Channel nomenclature to use
    **kwargs
        Additional arguments to pass to create_func

    Returns
    -------
    Path
        Path to the created MTH5 file
    """
    cache_key = f"{mth5_name}_{worker_id}_{file_version}_{channel_nomenclature}"

    # Return cached file if present and still exists
    cached = _MTH5_GLOBAL_CACHE.get(cache_key)
    if cached:
        p = Path(cached)
        if p.exists():
            return p

    # Create the MTH5 file in the worker-safe directory
    file_path = create_func(
        file_version=file_version,
        channel_nomenclature=channel_nomenclature,
        target_folder=target_dir,
        force_make_mth5=True,
        **kwargs,
    )

    # Cache the path
    ppath = Path(file_path)
    _MTH5_GLOBAL_CACHE[cache_key] = str(ppath)

    return ppath


@pytest.fixture(scope="session")
def worker_safe_test1_h5(mth5_target_dir, worker_id):
    """Create test1.h5 in a worker-safe directory."""
    return _create_worker_safe_mth5(
        "test1", create_test1_h5, mth5_target_dir, worker_id
    )


@pytest.fixture(scope="session")
def worker_safe_test2_h5(mth5_target_dir, worker_id):
    """Create test2.h5 in a worker-safe directory."""
    return _create_worker_safe_mth5(
        "test2", create_test2_h5, mth5_target_dir, worker_id
    )


@pytest.fixture(scope="session")
def worker_safe_test3_h5(mth5_target_dir, worker_id):
    """Create test3.h5 in a worker-safe directory."""
    return _create_worker_safe_mth5(
        "test3", create_test3_h5, mth5_target_dir, worker_id
    )


@pytest.fixture(scope="session")
def worker_safe_test12rr_h5(mth5_target_dir, worker_id):
    """Create test12rr.h5 in a worker-safe directory."""
    return _create_worker_safe_mth5(
        "test12rr", create_test12rr_h5, mth5_target_dir, worker_id
    )


# ============================================================================
# Parkfield Test Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def parkfield_paths():
    """Provide Parkfield test data paths."""
    from aurora.test_utils.parkfield.path_helpers import PARKFIELD_PATHS

    return PARKFIELD_PATHS


@pytest.fixture(scope="session")
def parkfield_h5_master(tmp_path_factory):
    """Create the master Parkfield MTH5 file once per test session.

    This downloads data from NCEDC and caches it in a persistent directory
    (.cache/aurora/parkfield) so it doesn't need to be re-downloaded for
    subsequent test runs. Only created once across all sessions.
    """
    from aurora.test_utils.parkfield.make_parkfield_mth5 import ensure_h5_exists

    # Use a persistent cache directory instead of temp
    # This way the file survives across test sessions
    cache_dir = Path.home() / ".cache" / "aurora" / "parkfield"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if file already exists in persistent cache
    cached_file = cache_dir / "parkfield.h5"
    if cached_file.exists():
        return cached_file

    # Check global cache first (for current session)
    cache_key = "parkfield_master"
    cached = _MTH5_GLOBAL_CACHE.get(cache_key)
    if cached:
        p = Path(cached)
        if p.exists():
            return p

    try:
        h5_path = ensure_h5_exists(target_folder=cache_dir)
        _MTH5_GLOBAL_CACHE[cache_key] = str(h5_path)
        return h5_path
    except IOError:
        pytest.skip("NCEDC data server not available")


@pytest.fixture(scope="session")
def parkfield_h5_path(parkfield_h5_master, tmp_path_factory, worker_id):
    """Copy master Parkfield MTH5 to worker-safe location.

    The master file is created once and cached persistently in
    ~/.cache/aurora/parkfield/ so it doesn't need to be re-downloaded.
    This fixture copies that cached file to a worker-specific temp
    directory to avoid file handle conflicts in pytest-xdist parallel execution.
    """
    import shutil

    cache_key = f"parkfield_h5_{worker_id}"

    # Check cache first
    cached = _MTH5_GLOBAL_CACHE.get(cache_key)
    if cached:
        p = Path(cached)
        if p.exists():
            return p

    # Create worker-safe directory and copy the master file
    target_dir = tmp_path_factory.mktemp(f"parkfield_{worker_id}")
    worker_h5_path = target_dir / parkfield_h5_master.name

    shutil.copy2(parkfield_h5_master, worker_h5_path)
    _MTH5_GLOBAL_CACHE[cache_key] = str(worker_h5_path)
    return worker_h5_path


@pytest.fixture
def parkfield_mth5(parkfield_h5_path):
    """Open and close MTH5 object for Parkfield data.

    This is a function-scoped fixture that ensures proper cleanup
    of MTH5 file handles after each test.
    """
    from mth5.mth5 import MTH5

    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(parkfield_h5_path, mode="r")
    yield mth5_obj
    mth5_obj.close_mth5()


@pytest.fixture
def parkfield_run_pkd(parkfield_mth5):
    """Get PKD station run 001 from Parkfield MTH5."""
    run_obj = parkfield_mth5.get_run("PKD", "001")
    return run_obj


@pytest.fixture
def parkfield_run_ts_pkd(parkfield_run_pkd):
    """Get RunTS object for PKD station."""
    return parkfield_run_pkd.to_runts()


@pytest.fixture(scope="class")
def parkfield_kernel_dataset_ss(parkfield_h5_path):
    """Create single-station KernelDataset for PKD."""
    from mth5.processing import KernelDataset, RunSummary

    run_summary = RunSummary()
    run_summary.from_mth5s([parkfield_h5_path])
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(run_summary=run_summary, local_station_id="PKD")
    return tfk_dataset


@pytest.fixture(scope="class")
def parkfield_kernel_dataset_rr(parkfield_h5_path):
    """Create remote-reference KernelDataset for PKD with SAO as RR."""
    from mth5.processing import KernelDataset, RunSummary

    run_summary = RunSummary()
    run_summary.from_mth5s([parkfield_h5_path])
    tfk_dataset = KernelDataset()
    tfk_dataset.from_run_summary(
        run_summary=run_summary, local_station_id="PKD", remote_station_id="SAO"
    )
    return tfk_dataset


@pytest.fixture
def disable_matplotlib_logging(request):
    """Disable noisy matplotlib logging for cleaner test output."""
    import logging

    loggers_to_disable = [
        "matplotlib.font_manager",
        "matplotlib.ticker",
    ]

    original_states = {}
    for logger_name in loggers_to_disable:
        logger_obj = logging.getLogger(logger_name)
        original_states[logger_name] = logger_obj.disabled
        logger_obj.disabled = True

    yield

    # Restore original states
    for logger_name, original_state in original_states.items():
        logging.getLogger(logger_name).disabled = original_state


# =============================================================================
# CAS04 FDSN Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def global_fdsn_miniseed_v010(tmp_path_factory):
    """Session-scoped CAS04 FDSN MTH5 file (v0.1.0) from mth5_test_data."""
    import obspy
    from mth5.clients.fdsn import FDSN
    from mth5_test_data import get_test_data_path

    # Get test data paths
    miniseed_path = get_test_data_path("miniseed")
    inventory_file = miniseed_path / "cas04_stationxml.xml"
    streams_file = miniseed_path / "cas_04_streams.mseed"

    # Verify files exist
    if not inventory_file.exists() or not streams_file.exists():
        pytest.skip(
            f"CAS04 test data not found in mth5_test_data. Expected:\n"
            f"  {inventory_file}\n"
            f"  {streams_file}"
        )

    # Load inventory and streams
    inventory = obspy.read_inventory(str(inventory_file))
    streams = obspy.read(str(streams_file))

    # Create temporary directory for this session
    session_dir = tmp_path_factory.mktemp("cas04_v010")

    # Create MTH5 from inventory and streams
    fdsn_client = FDSN(mth5_version="0.1.0")
    created_file = fdsn_client.make_mth5_from_inventory_and_streams(
        inventory, streams, save_path=session_dir
    )

    yield created_file

    # Cleanup
    if created_file.exists():
        created_file.unlink()


@pytest.fixture(scope="session")
def global_fdsn_miniseed_v020(tmp_path_factory):
    """Session-scoped CAS04 FDSN MTH5 file (v0.2.0) from mth5_test_data."""
    import obspy
    from mth5.clients.fdsn import FDSN
    from mth5_test_data import get_test_data_path

    # Get test data paths
    miniseed_path = get_test_data_path("miniseed")
    inventory_file = miniseed_path / "cas04_stationxml.xml"
    streams_file = miniseed_path / "cas_04_streams.mseed"

    # Verify files exist
    if not inventory_file.exists() or not streams_file.exists():
        pytest.skip(
            f"CAS04 test data not found in mth5_test_data. Expected:\n"
            f"  {inventory_file}\n"
            f"  {streams_file}"
        )

    # Load inventory and streams
    inventory = obspy.read_inventory(str(inventory_file))
    streams = obspy.read(str(streams_file))

    # Create temporary directory for this session
    session_dir = tmp_path_factory.mktemp("cas04_v020")

    # Create MTH5 from inventory and streams
    fdsn_client = FDSN(mth5_version="0.2.0")
    created_file = fdsn_client.make_mth5_from_inventory_and_streams(
        inventory, streams, save_path=session_dir
    )

    yield created_file

    # Cleanup
    if created_file.exists():
        created_file.unlink()
