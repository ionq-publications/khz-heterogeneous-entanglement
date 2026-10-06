"""Format values with bracketed uncertainties, for example 0.912(14) = 0.912 ± 0.014.

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

import math
import re


def format_uncertainty(value, error, upper_error=None):
    """Use two error digits for leading 1, otherwise one; preserve unequal errors."""
    upper_error = error if upper_error is None else upper_error
    if (
        not all(math.isfinite(x) for x in (value, error, upper_error))
        or min(error, upper_error) < 0
    ):
        raise ValueError("Values must be finite and uncertainties nonnegative")
    largest = max(error, upper_error)
    if largest == 0:
        return f"{value:g}(0)"
    exponent = math.floor(math.log10(largest))
    precision = -exponent + int(largest / 10**exponent < 2)
    decimals = max(0, precision)
    lower = round(round(error, precision) * 10**decimals)
    upper = round(round(upper_error, precision) * 10**decimals)
    number = round(value, precision)
    number = 0.0 if number == 0 else number
    uncertainty = str(lower) if lower == upper else f"-{lower},+{upper}"
    return f"{number:.{decimals}f}({uncertainty})"


def format_fit_report(result):
    """Keep lmfit diagnostics while formatting parameter uncertainties consistently."""
    return re.sub(
        r"(?m)^(\s*\w+:\s*)([-+\d.eE]+) \+/- ([-+\d.eE]+)",
        lambda match: match[1] + format_uncertainty(float(match[2]), float(match[3])),
        result.fit_report(),
    )
