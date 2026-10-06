"""Data loading, figure style, and figure export helpers for the paper figures.

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

from papertools.export import FIGURES_DIR, save_figure
from papertools.load import (
    load_dataset,
    load_provenance,
)
from papertools.style import (
    BAR_EDGE,
    BAR_FILL,
    BAR_LINEWIDTH,
    BAR_STYLE,
    DENSITY_MATRIX_COLORS,
    DOUBLE_COLUMN_WIDTH_MM,
    ERRORBAR_STYLE,
    FLOOR_COLOR,
    IONQ_ORANGE,
    IONQ_ORANGE_EDGE,
    MM,
    SECONDARY_TEXT,
    SINGLE_COLUMN_WIDTH_MM,
    SMALL_FONT_SIZE,
    apply_publication_style,
)

__all__ = [
    "BAR_EDGE",
    "BAR_FILL",
    "BAR_LINEWIDTH",
    "BAR_STYLE",
    "DENSITY_MATRIX_COLORS",
    "ERRORBAR_STYLE",
    "FIGURES_DIR",
    "FLOOR_COLOR",
    "IONQ_ORANGE",
    "IONQ_ORANGE_EDGE",
    "MM",
    "SECONDARY_TEXT",
    "SMALL_FONT_SIZE",
    "save_figure",
    "DOUBLE_COLUMN_WIDTH_MM",
    "SINGLE_COLUMN_WIDTH_MM",
    "load_dataset",
    "load_provenance",
    "apply_publication_style",
]
