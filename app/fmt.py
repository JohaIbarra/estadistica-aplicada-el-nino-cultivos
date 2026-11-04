"""Formato numérico en español (coma decimal, punto de miles)."""

from __future__ import annotations


def num(x: float, dec: int = 0) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, dec: int = 1, sign: bool = True) -> str:
    s = f"{x:+,.{dec}f}" if sign else f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") + " %"


def share(x: float, dec: int = 0) -> str:
    """Fracción 0-1 como porcentaje."""
    return pct(100 * x, dec, sign=False)


def ci(lo: float, hi: float, dec: int = 1) -> str:
    return f"{pct(lo, dec)} a {pct(hi, dec)}"
