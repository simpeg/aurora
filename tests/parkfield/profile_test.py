#!/usr/bin/env python
"""Profile the Parkfield calibration test."""
import cProfile
import pstats
import subprocess
import sys
from io import StringIO


# Run pytest with cProfile
prof = cProfile.Profile()
prof.enable()

# Run the test
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/parkfield/test_parkfield_pytest.py::TestParkfieldCalibration::test_calibration_sanity_check",
        "-v",
    ],
    cwd=".",
)

prof.disable()

# Print stats
s = StringIO()
ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
ps.print_stats(50)
print(s.getvalue())

sys.exit(result.returncode)
