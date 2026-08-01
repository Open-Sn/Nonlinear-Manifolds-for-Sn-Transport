import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_name",
    ["Transport_Driver_Benchmark_1D", "Nonlinear_Manifold_ROM"],
)
def test_production_module_import_assembles_globals_without_solve_or_io(
    module_name, tmp_path
):
    script = f"""
import numpy as np
import scipy.integrate

def forbidden(*args, **kwargs):
    raise AssertionError("production I/O or integration was called during import")

np.load = forbidden
np.save = forbidden
scipy.integrate.solve_ivp = forbidden
module = __import__({module_name!r})
driver = getattr(module, "transport_driver", module)
assert driver._PRODUCTION_INITIALIZED is True
assert driver.globalMM.shape == (6000, 6000)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []
