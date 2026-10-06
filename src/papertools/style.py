"""Publication figure style defaults.

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

import matplotlib

FONT_SANS_SERIF = ["Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]

MM = 1 / 25.4
"""Convert millimeters to inches for Matplotlib figure sizes."""

SMALL_FONT_SIZE = 5
"""Text size in points for small panels and insets."""

BAR_FILL = "#598fff"
BAR_EDGE = "#2155b5"
IONQ_ORANGE = "#ff5000"
IONQ_ORANGE_EDGE = "#b53800"
BAR_LINEWIDTH = 0.6
SECONDARY_TEXT = "#303030"
DENSITY_MATRIX_COLORS = ("#bfd3ff", "#1260ff")
FLOOR_COLOR = "#f1f2f4"

BAR_STYLE = {"facecolor": BAR_FILL, "edgecolor": BAR_EDGE, "linewidth": BAR_LINEWIDTH}
"""Shared appearance for bars and filled histograms; pass with **BAR_STYLE."""

ERRORBAR_STYLE = {"ecolor": "black", "elinewidth": 1.0, "capsize": 0}
"""Uncapped error bars; pass with **ERRORBAR_STYLE."""

SINGLE_COLUMN_WIDTH_MM = 89
"""Single-column figure width in millimeters."""

DOUBLE_COLUMN_WIDTH_MM = 183
"""Double-column figure width in millimeters."""


def apply_publication_style(
    font_size: float = 7,
    figsize: tuple[float, float] = (6.5, 5.0),
    line_width: float = 1.5,
    marker_size: float = 7,
    marker_edge_width: float = 1.5,
) -> None:
    """Apply rcParams suitable for publication figures.

    Call once at the top of a notebook before any plotting.
    """
    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_SANS_SERIF,
            "mathtext.fontset": "stixsans",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "figure.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "figure.figsize": figsize,
            "figure.constrained_layout.use": True,
            "figure.max_open_warning": 100,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "savefig.format": "png",
            "savefig.bbox": None,
            "image.cmap": "inferno",
            "image.interpolation": "nearest",
            "legend.fontsize": 0.9 * font_size,
            "legend.handlelength": 0.8,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.0,
            "legend.borderaxespad": 0.2,
            "legend.borderpad": 0.3,
            "axes.ymargin": 0.02,
            "axes.axisbelow": False,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "errorbar.capsize": 0,
            "lines.linewidth": line_width,
            "lines.markersize": marker_size,
            "lines.markeredgewidth": marker_edge_width,
            "lines.markeredgecolor": "black",
            "scatter.edgecolors": "black",
        }
    )
