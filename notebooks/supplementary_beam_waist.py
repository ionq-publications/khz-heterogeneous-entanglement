"""Plot the measured and fitted beam radius versus distance from the fiber facet.

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
# # Beam radius versus distance from the fiber facet
#
# The embedded measurements are 1/e² intensity radii from 2D Gaussian fits
# to camera images. All lengths are in metres.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from papertools import (
    BAR_EDGE,
    BAR_FILL,
    FIGURES_DIR,
    IONQ_ORANGE,
    IONQ_ORANGE_EDGE,
    MM,
    SINGLE_COLUMN_WIDTH_MM,
    SMALL_FONT_SIZE,
    apply_publication_style,
    save_figure,
)

apply_publication_style()

beam_data = {
    "positions_m": [0.005, 0.015, 0.02, 0.023, 0.025, 0.027],
    "radius_x_2d_m": [
        0.0024366420205840667,
        0.0016999001754581578,
        0.0013279782480936817,
        0.0011056771744002664,
        0.0009566483591494924,
        0.0008081315060874225,
    ],
    "radius_y_2d_m": [
        0.0024123921343115313,
        0.0017029915994599088,
        0.001332049046644656,
        0.001108875714402675,
        0.0009595461739912323,
        0.0008105237875835897,
    ],
    "facet_position_m": 0.0381507470465492,
    "wavelength_m": 4.93e-07,
    "matched_gaussian_waist_m": 1.6905405405405407e-06,
    "propagation_distance_m": [
        0.0,
        5e-06,
        1e-05,
        1.5e-05,
        2e-05,
        2.5e-05,
        3e-05,
        4e-05,
        5e-05,
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
    ],
    "propagated_lp01_radius_m": [
        1.7017736519650528e-06,
        1.6313242499839602e-06,
        1.9162474724795598e-06,
        2.30840972458337e-06,
        2.7040311018052564e-06,
        3.0954166397920753e-06,
        3.4805154343055005e-06,
        4.244686139860007e-06,
        5.005830462945491e-06,
        8.854560387147505e-06,
        1.6811886547404908e-05,
        4.125399338459694e-05,
        8.22670675620507e-05,
        0.0001644108814243499,
        0.0004109440414031616,
        0.0008218556084619822,
        0.0016437233531577697,
    ],
    "lp01_curve_parameters": {
        "waist_near_m": 1.7017736519650528e-06,
        "waist_far_m": 1.9094171763996614e-06,
        "crossover_distance_m": 9.952614886814886e-05,
        "sigmoid_width": 0.05000047334720049,
    },
    "manufacturer_mfd_m": 3.8e-06,
    "fiber_na": 0.12,
    "effective_na": 0.073,
}

# %% [markdown]
# ## Propagation curves
#
# The Gaussian reference curves use w(d) = w0*sqrt(1+(d/z_R)²), where
# z_R = pi*w0²/lambda. The manufacturer mode-field diameter (MFD) is divided
# by two to obtain w0. The NA references use w0 = lambda/(pi*arcsin(NA)),
# treating arcsin(NA) as the divergence half-angle in air.
#
# The LP01 curve approximates radii from Gaussian intensity fits to a
# numerically propagated fiber mode. Its effective waist is
# w0(d) = w_near + (w_far-w_near)*s(d), with
# s(d) = 1/(1+exp(-log(d/d0)/width)) and s(0)=0, inserted into the same
# Gaussian propagation formula. The embedded propagation points cover
# 0-20 mm; the curve beyond 20 mm is extrapolated. The inset shows model
# predictions, not measurements.


# %%
def gaussian_radius(distance, waist):
    """Return the 1/e² intensity radius; distance and waist are in metres."""
    rayleigh_range = np.pi * waist**2 / beam_data["wavelength_m"]
    return waist * np.sqrt(1 + (distance / rayleigh_range) ** 2)


def lp01_radius(distance):
    """Evaluate the saved distance-dependent-waist approximation in metres."""
    params = beam_data["lp01_curve_parameters"]
    argument = (
        np.log(np.maximum(distance, 1e-15) / params["crossover_distance_m"])
        / params["sigmoid_width"]
    )
    fraction = 1 / (1 + np.exp(-np.clip(argument, -500, 500)))
    waist = (
        params["waist_near_m"]
        + (params["waist_far_m"] - params["waist_near_m"]) * fraction
    )
    return gaussian_radius(distance, waist)


distance = beam_data["facet_position_m"] - np.array(beam_data["positions_m"])
radius_x = np.array(beam_data["radius_x_2d_m"])
radius_y = np.array(beam_data["radius_y_2d_m"])
propagation_distance = np.array(beam_data["propagation_distance_m"])
propagation_radius = np.array(beam_data["propagated_lp01_radius_m"])
assert distance.shape == radius_x.shape == radius_y.shape == (6,)
assert np.isfinite([distance, radius_x, radius_y]).all()
assert np.all(distance > 0) and np.all(radius_x > 0) and np.all(radius_y > 0)
assert propagation_distance.shape == propagation_radius.shape
assert np.isfinite([propagation_distance, propagation_radius]).all()
assert np.all(np.diff(propagation_distance) > 0) and np.all(propagation_radius > 0)
assert np.isclose(
    lp01_radius(0),
    beam_data["lp01_curve_parameters"]["waist_near_m"],
    rtol=1e-12,
    atol=0,
)

# %%
BEAM_FIGSIZE = (SINGLE_COLUMN_WIDTH_MM * MM, 75 * MM)
BEAM_DISTANCE_RANGE = (0, 0.035)
BEAM_RADIUS_RANGE = (0, 0.0026)
BEAM_INSET_POSITION = (0.63, 0.12, 0.34, 0.36)
BEAM_MARKER_SIZE = 4
BEAM_CURVE_POINTS = 50
BEAM_REFERENCES = (
    (
        "Fiber-matched Gaussian",
        beam_data["matched_gaussian_waist_m"],
        IONQ_ORANGE_EDGE,
        "--",
    ),
    ("Thorlabs MFD", beam_data["manufacturer_mfd_m"] / 2, IONQ_ORANGE_EDGE, ":"),
    (
        "Fiber NA 0.12",
        beam_data["wavelength_m"] / (np.pi * np.arcsin(beam_data["fiber_na"])),
        "0.65",
        "-",
    ),
    (
        "Effective NA 0.073",
        beam_data["wavelength_m"] / (np.pi * np.arcsin(beam_data["effective_na"])),
        "0.65",
        "--",
    ),
)

fig, ax = plt.subplots(figsize=BEAM_FIGSIZE)
inset = ax.inset_axes(BEAM_INSET_POSITION)
main_grid = np.linspace(*BEAM_DISTANCE_RANGE, BEAM_CURVE_POINTS)
inset_grid = np.linspace(0, 40e-6, BEAM_CURVE_POINTS)
for axis, grid, x_scale, y_scale in (
    (ax, main_grid, 1000, 1000),
    (inset, inset_grid, 1e6, 1e6),
):
    axis.plot(
        grid * x_scale,
        lp01_radius(grid) * y_scale,
        color=BAR_EDGE,
        lw=1.3,
        alpha=0.7,
        label=r"LP$_{01}$ mode",
        zorder=3,
    )
    for label, waist, color, linestyle in BEAM_REFERENCES:
        axis.plot(
            grid * x_scale,
            gaussian_radius(grid, waist) * y_scale,
            color=color,
            linestyle=linestyle,
            lw=0.8,
            label=label,
            zorder=4,
        )
ax.plot(
    distance * 1000,
    radius_x * 1000,
    "*",
    color=BAR_FILL,
    markeredgecolor=BAR_EDGE,
    markersize=BEAM_MARKER_SIZE,
    markeredgewidth=0.3,
    label="X axis",
    zorder=7,
)
ax.plot(
    distance * 1000,
    radius_y * 1000,
    "*",
    color=IONQ_ORANGE,
    markeredgecolor=IONQ_ORANGE_EDGE,
    markersize=BEAM_MARKER_SIZE,
    markeredgewidth=0.3,
    label="Y axis",
    zorder=6,
)
ax.set(
    xlim=np.array(BEAM_DISTANCE_RANGE) * 1000,
    ylim=np.array(BEAM_RADIUS_RANGE) * 1000,
    xlabel="Distance from the fiber facet (mm)",
    ylabel="Beam radius (mm)",
)
ax.tick_params(labelsize=6)
handles, labels = ax.get_legend_handles_labels()
heading = Line2D([], [], linestyle="none")
legend_handles = [
    heading,
    handles[5],
    handles[6],
    heading,
    handles[0],
    heading,
    *handles[1:5],
]
legend_labels = [
    r"$\mathbf{Measured}$ (2D Gaussian fit)",
    labels[5],
    labels[6],
    r"$\mathbf{Numerically\ propagated}$",
    labels[0],
    r"$\mathbf{Closed\ form}$",
    *labels[1:5],
]
ax.legend(
    legend_handles,
    legend_labels,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    frameon=False,
    fontsize=SMALL_FONT_SIZE,
    handlelength=2,
    labelspacing=0.3,
    borderaxespad=0.8,
)
inset.set(
    xlim=(0, 40), ylim=(0, 5), xticks=[0, 10, 20, 30, 40], yticks=[0, 1, 2, 3, 4, 5]
)
inset.set_xlabel("Distance (µm)", fontsize=SMALL_FONT_SIZE, labelpad=1)
inset.set_ylabel("Radius (µm)", fontsize=SMALL_FONT_SIZE, labelpad=1)
inset.tick_params(labelsize=SMALL_FONT_SIZE, length=2, pad=1)
save_figure(fig, "supplementary_beam_waist")

# %%
lines = [
    "Supplementary beam-radius propagation",
    "",
    "Measured: stage_position_m distance_from_facet_m radius_x_m radius_y_m",
]
for position, d, wx, wy in zip(beam_data["positions_m"], distance, radius_x, radius_y):
    lines.append(f"{position:.16g} {d:.16g} {wx:.16g} {wy:.16g}")
lines.extend(
    ["", "Numerical propagation: distance_m lp01_radius_m approximation_radius_m"]
)
lines.extend(
    f"{d:.16g} {w:.16g} {f:.16g}"
    for d, w, f in zip(
        propagation_distance,
        propagation_radius,
        lp01_radius(propagation_distance),
    )
)
for name, grid in (("Main plot", main_grid), ("Inset", inset_grid)):
    lines.extend(
        [
            "",
            name,
            "distance_m lp01_fit_m matched_gaussian_m manufacturer_mfd_m fiber_na_m effective_na_m",
        ]
    )
    curves = [lp01_radius(grid)] + [
        gaussian_radius(grid, waist) for _, waist, _, _ in BEAM_REFERENCES
    ]
    lines.extend(
        " ".join(f"{value:.16g}" for value in row) for row in zip(grid, *curves)
    )
(FIGURES_DIR / "txt").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "txt" / "supplementary_beam_waist.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
