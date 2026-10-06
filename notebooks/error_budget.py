"""Calculate the predicted Ion-SiV fidelity from the estimated error budget.

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

# %% [markdown]
# # Error budget

# %%
import numpy as np

from papertools import FIGURES_DIR

CORRELATORS = ("XX", "YY", "ZZ")
# Error probabilities for the XX, YY, and ZZ correlators.
ERROR_PROBABILITIES = {
    "Ion": {
        "Pumping leakage after herald (MZ EOM)": (0.0087, 0.0087, 0.0145),
        "Ion state detection": (0.0036, 0.0036, 0.0036),
        "Double excitation": (0.0060, 0.0060, 0.0020),
        "Other (ion temperature, aberrations)": (0.0126, 0.0126, 0.0057),
    },
    "QFC": {
        "Signal to noise": (0.0185, 0.0185, 0.0185),
        "Bidirectional imbalance": (0.0010, 0.0010, 0.0010),
    },
    "Memory": {
        "Electron MW pulse errors": (0.0030, 0.0030, 0.0015),
        "Dephasing from environment": (0.0034, 0.0034, 0.0),
        "Electron fast reset": (0.0013, 0.0013, 0.0),
        "Nuclear spin flip during sequence": (0.0100, 0.0100, 0.0100),
        "Electron spin readout": (0.0004, 0.0004, 0.0004),
        "Finite SiV reflection contrast": (0.0100, 0.0100, 0.0100),
    },
    "Detection": {
        "Undetected heralding photons": (0.0192, 0.0192, 0.0),
    },
}


def combined_error(probabilities):
    """Return the probability that at least one independent error occurs.

    Args:
        probabilities: Error probabilities, with shape (sources, correlators).

    Returns:
        Combined error probability for each correlator.
    """
    return 1 - np.prod(1 - np.asarray(probabilities), axis=0)


# %% [markdown]
# ## Total error, correlator contrasts, and predicted fidelity

# %%
all_sources = [p for sources in ERROR_PROBABILITIES.values() for p in sources.values()]
assert all(0 <= p < 0.5 for row in all_sources for p in row)
total_error = combined_error(all_sources)
contrast = 1 - 2 * total_error
predicted_fidelity = (1 + contrast.sum()) / 4
subsystem_infidelity = {
    subsystem: combined_error(list(sources.values())).sum() / 2
    for subsystem, sources in ERROR_PROBABILITIES.items()
}

# %% [markdown]
# ## Supplementary error budget table

# %%
lines = [
    "Error budget",
    "",
    "Supplementary error budget",
    "Error probabilities in %",
    "",
    f"{'Subsystem':<10} {'Error source':<40}" + "".join(f"{c:>8}" for c in CORRELATORS),
]
for subsystem, sources in ERROR_PROBABILITIES.items():
    for source, p in sources.items():
        lines.append(
            f"{subsystem:<10} {source:<40}" + "".join(f"{100 * x:8.2f}" for x in p)
        )
lines.extend(
    [
        f"{'Total error probability':<51}"
        + "".join(f"{100 * x:8.2f}" for x in total_error),
        f"{'Correlator contrast':<51}" + "".join(f"{x:8.3f}" for x in contrast),
        f"Predicted fidelity: {100 * predicted_fidelity:.1f} %",
        "",
        "LaTeX error budget table",
        r"\begin{table}[tb]",
        r"  \caption{Estimated error budget.}",
        r"  \label{tab:SI:error_budget}",
        r"  \centering",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    Subsystem & Error source & XX (\%) & YY (\%) & ZZ (\%) \\",
        r"    \midrule",
    ]
)
for subsystem, sources in ERROR_PROBABILITIES.items():
    for index, (source, p) in enumerate(sources.items()):
        cells = [subsystem if index == 0 else "", source]
        cells += [f"{100 * x:.2f}" for x in p]
        lines.append("    " + " & ".join(cells) + r" \\")
lines.extend(
    [
        r"    \midrule",
        "    Total error probability & & "
        + " & ".join(f"{100 * x:.2f}" for x in total_error)
        + r" \\",
        "    Correlator contrast & & "
        + " & ".join(f"{x:.3f}" for x in contrast)
        + r" \\",
        rf"    Predicted fidelity (\%) & \multicolumn{{4}}{{c}}{{{100 * predicted_fidelity:.1f}}} \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
)

# %% [markdown]
# ## Subsystem infidelity table

# %%
lines.extend(
    ["", "Error budget by subsystem", "", f"{'Subsystem':<20} {'Infidelity (%)':>14}"]
)
lines.extend(
    f"{subsystem:<20} {100 * value:14.1f}"
    for subsystem, value in subsystem_infidelity.items()
)
lines.extend(
    [
        f"{'Predicted fidelity':<20} {100 * predicted_fidelity:14.1f}",
        "",
        "LaTeX subsystem table",
        r"\begin{table}[tb]",
        r"  \caption{Estimated infidelity contribution of each subsystem.}",
        r"  \label{tab:error_budget}",
        r"  \centering",
        r"  \begin{tabular}{lc}",
        r"    \toprule",
        r"    Subsystem & Infidelity (\%) \\",
        r"    \midrule",
    ]
)
lines.extend(
    f"    {subsystem} & {100 * value:.1f}" + r" \\"
    for subsystem, value in subsystem_infidelity.items()
)
lines.extend(
    [
        r"    \midrule",
        rf"    Predicted fidelity (\%) & {100 * predicted_fidelity:.1f} \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
)
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "error_budget.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
