"""Two-qubit counts, populations, and parity with Dirichlet uncertainties.

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

import numpy as np
import xarray as xr
from scipy.stats import dirichlet, norm

DIRICHLET_SAMPLES = 100_000
"""Number of posterior draws for each population or correlator estimate."""

UPPER_QUANTILE = float(norm.cdf(1))
"""Upper quantile of a central 68.27% interval; the lower is 1 minus this value."""


def make_two_qubit_state(
    state_1: xr.DataArray,
    state_1_high_value: int,
    state_2: xr.DataArray,
    state_2_high_value: int,
    name: str,
    label: str,
) -> xr.DataArray:
    """Encode outcomes as q1 + 2*q2, with NaN if either input is missing."""
    state = (state_1 == state_1_high_value) * 1 + (state_2 == state_2_high_value) * 2
    state = state.where(state_1.notnull() & state_2.notnull())
    state.attrs.update(
        name=f"{name}_state", label=f"{label} State", long_name=f"{label} State"
    )
    return state


def _errors(samples):
    mean = samples.mean(axis=0)
    lower, upper = np.quantile(samples, [1 - UPPER_QUANTILE, UPPER_QUANTILE], axis=0)
    return mean - lower, upper - mean


def _population(counts):
    samples = dirichlet.rvs(counts + 1, size=DIRICHLET_SAMPLES)
    minus, plus = _errors(samples)
    mean = counts / counts.sum() if counts.sum() else np.full(counts.shape, np.nan)
    return mean, minus, plus, np.cov(samples, rowvar=False)


def _correlator(counts):
    coefficients = np.array([1, -1, -1, 1])
    samples = np.einsum(
        "ij,j->i", dirichlet.rvs(counts + 1, size=DIRICHLET_SAMPLES), coefficients
    )
    minus, plus = _errors(samples)
    mean = np.sum(coefficients * (counts / counts.sum())) if counts.sum() else np.nan
    return mean, minus, plus


def calculate_histogram_population_and_correlator(
    data_array: xr.DataArray, label: str, prefix: str, rep_dim: str
) -> tuple[xr.DataArray, xr.Dataset, xr.Dataset]:
    """Count states 0..3 over rep_dim; return populations and parity."""
    state_dim = "state_idx"
    histogram = (data_array == xr.DataArray(np.arange(4), dims=state_dim)).sum(rep_dim)
    mean, minus, plus, covariance = xr.apply_ufunc(
        _population,
        histogram,
        input_core_dims=[[state_dim]],
        output_core_dims=[
            [state_dim],
            [state_dim],
            [state_dim],
            [state_dim, "state_idx_2"],
        ],
        vectorize=True,
        output_dtypes=[float, float, float, float],
    )
    population = xr.Dataset(
        {
            f"{prefix}_pop_mean": mean,
            f"{prefix}_pop_minus_error": minus,
            f"{prefix}_pop_plus_error": plus,
            f"{prefix}_pop_covariance": covariance,
        },
        attrs={"prefix": prefix},
    )
    mean, minus, plus = xr.apply_ufunc(
        _correlator,
        histogram,
        input_core_dims=[[state_dim]],
        output_core_dims=[[], [], []],
        vectorize=True,
        output_dtypes=[float, float, float],
    )
    correlator = xr.Dataset(
        {
            f"{prefix}_corr_mean": mean,
            f"{prefix}_corr_minus_error": minus,
            f"{prefix}_corr_plus_error": plus,
        },
        attrs={"prefix": prefix},
    )
    for dataset, suffix, title in (
        (population, "pop", "Population"),
        (correlator, "corr", "Correlator"),
    ):
        dataset[f"{prefix}_{suffix}_mean"].attrs.update(
            name=f"{prefix}_{suffix}_mean",
            label=f"{label} {title}",
            long_name=f"{label} {title}",
        )
    return histogram, population, correlator
