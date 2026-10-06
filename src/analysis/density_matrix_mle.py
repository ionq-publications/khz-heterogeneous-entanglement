"""Two-qubit maximum likelihood state reconstruction.

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

import warnings
from typing import NamedTuple

import cvxpy as cp
import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from papertools.formatting import format_uncertainty

MEASUREMENT_SETTINGS = tuple(first + second for first in "XYZ" for second in "XYZ")
"""The nine required two-qubit measurement settings."""

OUTCOME_LABELS = ("00", "01", "10", "11")
"""Joint outcomes, with 0 having eigenvalue +1 and 1 having eigenvalue -1."""

BELL_STATES = {
    "Phi+": np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0),
    "Phi-": np.array([1.0, 0.0, 0.0, -1.0], dtype=complex) / np.sqrt(2.0),
    "Psi+": np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2.0),
    "Psi-": np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0),
}
"""Bell-state targets for the fidelity."""

ONE_SIGMA_QUANTILES = (float(norm.cdf(-1)), float(norm.cdf(1)))
"""Quantiles of the central one-sigma (68.27%) interval."""

SOLVER = cp.SCS
"""Convex solver for the maximum likelihood fit."""

_PAULIS = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _projector(setting: str, outcome: str) -> npt.NDArray[np.complex128]:
    halves = [
        (np.eye(2) + (-1) ** int(bit) * _PAULIS[pauli]) / 2.0
        for pauli, bit in zip(setting, outcome, strict=True)
    ]
    return np.kron(*halves)


PROJECTORS = np.array(
    [
        [_projector(setting, outcome) for outcome in OUTCOME_LABELS]
        for setting in MEASUREMENT_SETTINGS
    ]
)
"""Measurement projectors, indexed by setting, outcome, row, column."""


class TomographyResult(NamedTuple):
    """Fitted state and bootstrap uncertainties.

    Errors are bootstrap standard deviations. Fidelity is <target|rho|target>;
    fidelity_interval contains the 68.27% bootstrap quantile bounds.
    """

    density_matrix: npt.NDArray[np.complex128]
    real_error: npt.NDArray[np.float64]
    imaginary_error: npt.NDArray[np.float64]
    fidelity: float
    fidelity_error: float
    fidelity_interval: tuple[float, float]
    bootstrap_density_matrices: npt.NDArray[np.complex128]

    def __str__(self) -> str:
        lines = []
        for part, values, errors in (
            ("Re", self.density_matrix.real, self.real_error),
            ("Im", self.density_matrix.imag, self.imaginary_error),
        ):
            lines.append(
                f"{part}[rho], as value(uncertainty) with both multiplied by 1e3:"
            )
            lines += [
                "   "
                + " ".join(
                    f"{format_uncertainty(1e3 * value, 1e3 * error):>14}"
                    for value, error in zip(row, row_errors, strict=True)
                )
                for row, row_errors in zip(values, errors, strict=True)
            ]
        lines.append(
            f"Fidelity: {format_uncertainty(self.fidelity, self.fidelity_error)}"
        )
        lines.append("68% interval: [{:.4f}, {:.4f}]".format(*self.fidelity_interval))
        return "\n".join(lines)


def counts_to_array(counts: dict[str, dict[str, int]]) -> npt.NDArray[np.float64]:
    """Validate counts and arrange them by setting and outcome."""
    unknown_settings = sorted(set(counts) - set(MEASUREMENT_SETTINGS))
    if unknown_settings:
        raise ValueError(
            f"Unknown settings {unknown_settings}; expected exactly {MEASUREMENT_SETTINGS}"
        )
    for setting in MEASUREMENT_SETTINGS:
        if setting not in counts:
            raise ValueError(
                f"Missing setting {setting}; all of {MEASUREMENT_SETTINGS} are required"
            )
        unknown_outcomes = sorted(set(counts[setting]) - set(OUTCOME_LABELS))
        if unknown_outcomes:
            raise ValueError(
                f"Setting {setting} has unknown outcomes {unknown_outcomes}; expected {OUTCOME_LABELS}"
            )
        missing = [
            outcome for outcome in OUTCOME_LABELS if outcome not in counts[setting]
        ]
        if missing:
            raise ValueError(f"Setting {setting} is missing outcomes {missing}")
    table = np.array(
        [[counts[s][o] for o in OUTCOME_LABELS] for s in MEASUREMENT_SETTINGS],
        dtype=float,
    )
    if not np.all(np.isfinite(table) & (table >= 0) & (table == np.floor(table))):
        raise ValueError(
            f"Counts must be finite non-negative whole numbers, got\n{table}"
        )
    empty = [
        s
        for s, total in zip(MEASUREMENT_SETTINGS, table.sum(axis=1), strict=True)
        if total == 0
    ]
    if empty:
        raise ValueError(f"Settings with no shots: {empty}")
    return table


def fit_maximum_likelihood_state(
    counts_table: npt.NDArray,
) -> npt.NDArray[np.complex128]:
    """Maximize the multinomial log likelihood with a physical density matrix."""
    flat_counts = counts_table.ravel()
    observed = flat_counts > 0
    weights = flat_counts[observed] / flat_counts.sum()
    density_matrix = cp.Variable((4, 4), hermitian=True)
    probabilities = cp.hstack(
        [
            cp.real(cp.trace(projector @ density_matrix))
            for projector in PROJECTORS.reshape(-1, 4, 4)[observed]
        ]
    )
    problem = cp.Problem(
        cp.Maximize(weights @ cp.log(probabilities)),
        [density_matrix >> 0, cp.real(cp.trace(density_matrix)) == 1],
    )
    problem.solve(solver=SOLVER)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            f"Maximum likelihood fit did not converge: status {problem.status}"
        )
    if problem.status == cp.OPTIMAL_INACCURATE:
        warnings.warn(
            "Maximum likelihood fit only reached an inaccurate solution.", stacklevel=2
        )
    return _project_to_physical(np.array(density_matrix.value))


def reconstruct_density_matrix(
    counts: dict[str, dict[str, int]],
    target_state: npt.NDArray[np.complex128] | None = None,
    num_bootstrap_samples: int = 500,
    seed: int = 0,
) -> TomographyResult:
    """Fit the state and bootstrap its errors; the default target is |Phi+>.

    The bootstrap uses its own seeded generator, independent of np.random.seed.
    """
    if target_state is None:
        target_state = BELL_STATES["Phi+"]
    if num_bootstrap_samples < 1:
        raise ValueError(
            "num_bootstrap_samples must be at least 1; call fit_maximum_likelihood_state for a bare fit"
        )
    counts_table = counts_to_array(counts)
    density_matrix = fit_maximum_likelihood_state(counts_table)
    shots_per_setting = counts_table.sum(axis=1).astype(int)
    probabilities = np.clip(
        np.einsum("skij,ji->sk", PROJECTORS, density_matrix).real, 0.0, None
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    generator = np.random.default_rng(seed)
    # Parametric bootstrap: assume the fitted state gives the true outcome
    # probabilities, with independent shots and measurement settings. Keep
    # each setting's shot count fixed and refit every multinomial resample.
    # This models sampling uncertainty, not calibration or readout systematics.
    bootstrap = np.array(
        [
            fit_maximum_likelihood_state(
                generator.multinomial(shots_per_setting, probabilities)
            )
            for _ in range(num_bootstrap_samples)
        ]
    )
    fidelities = _fidelity(bootstrap, target_state)
    # Central one-sigma interval, using the standard normal CDF at -1 and +1.
    low, high = np.quantile(fidelities, ONE_SIGMA_QUANTILES)
    return TomographyResult(
        density_matrix=density_matrix,
        real_error=bootstrap.real.std(axis=0),
        imaginary_error=bootstrap.imag.std(axis=0),
        fidelity=float(_fidelity(density_matrix, target_state)),
        fidelity_error=float(fidelities.std()),
        fidelity_interval=(float(low), float(high)),
        bootstrap_density_matrices=bootstrap,
    )


def _project_to_physical(
    matrix: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Remove negative eigenvalues from solver round-off and normalize the trace."""
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.conj().T) / 2.0)
    projected = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.conj().T
    return projected / np.trace(projected).real


def _fidelity(density_matrices: npt.NDArray, target_state: npt.NDArray) -> npt.NDArray:
    """Return <target|rho|target> for one state or a stack of states."""
    return np.einsum(
        "i,...ij,j->...", target_state.conj(), density_matrices, target_state
    ).real
