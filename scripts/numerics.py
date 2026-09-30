"""Explicit scalar arithmetic for reproducible generated tables."""


def sequential_sum(values):
    """Preserve the left-to-right float addition used by the published build.

    Python 3.12 changed built-in sum() to use compensated float summation.
    Explicit additions keep the exported last digits identical on 3.9–3.12.
    """
    total = 0.0
    for value in values:
        total += float(value)
    return total
