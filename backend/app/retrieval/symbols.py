"""Detect formulas that are a single symbol."""

import re

_SINGLE = re.compile(r"[A-Za-z0-9]|\\[A-Za-z]+")


def is_single_symbol(latex: str) -> bool:
    """Tells whether LaTeX is one letter, digit or backslash command, ignoring wrapping braces.

    Args:
        latex: LaTeX source.

    Returns:
        True for e.g. `m`, `{m}` or `\\omega`; false for `t_a`, `\\mathbf{M}` or anything longer.
    """
    s = latex.strip()
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    return _SINGLE.fullmatch(s) is not None
