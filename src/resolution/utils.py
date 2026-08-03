import math


def rounded(n: float) -> int:
    """Round `n` to nearest integer using round-half-up."""
    if n >= 0:
        return math.floor(n + 0.5)
    else:
        return math.ceil(n - 0.5)


def round_to_nearest_multiple(n: float, multiple: float = 1):
    """Round `n` to the nearest multiple of `multiple` using round-half-up."""
    return rounded(n / multiple) * multiple
