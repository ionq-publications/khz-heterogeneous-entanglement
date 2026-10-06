"""Load run data; set PRINT_OPENED_FILENAMES=1 to print absolute input paths.

Copyright (c) 2026 IonQ, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import os
from pathlib import Path

import xarray as xr

_REPO_ROOT = Path(__file__).parent.parent.parent
_RUNS_DIR = _REPO_ROOT / "data" / "runs"


def _run_dir(run_id: int) -> Path:
    path = _RUNS_DIR / str(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Run {run_id} not found in {_RUNS_DIR}")
    return path


def _run_file(run_id: int, filename: str) -> Path:
    path = (_run_dir(run_id) / filename).resolve()
    if os.environ.get("PRINT_OPENED_FILENAMES") == "1":
        print(path)
    return path


def load_dataset(run_id: int, filename: str) -> xr.Dataset:
    """Load an HDF5 dataset from a run folder.

    Args:
        run_id: Integer run ID.
        filename: File name including extension, e.g. ``"main.h5"``.
    """
    return xr.open_dataset(_run_file(run_id, filename), engine="h5netcdf")


def load_provenance(run_id: int) -> dict:
    """Load the provenance metadata for a run.

    Returns the contents of ``provenance.json`` as a dict.

    Args:
        run_id: Integer run ID.
    """
    return json.loads(_run_file(run_id, "provenance.json").read_text())
