from __future__ import annotations

from itertools import product

from sncosmo.photdata import PHOTDATA_ALIASES as SNCOSMO_ALIAS_KEY

ALL_SNCOSMO_ALIASES = {
    alias for aliases in SNCOSMO_ALIAS_KEY.values() for alias in aliases
}


def are_sncosmo_aliases(labels: set[str]) -> bool:
    return labels.issubset(ALL_SNCOSMO_ALIASES)


def resolve_sncosmo_label(label: str) -> str:
    for key, aliases in SNCOSMO_ALIAS_KEY.items():
        if label in aliases:
            return key
    else:
        raise ValueError(f"{label} is not a known alias of any sncosmo column label")


def stitch(
    *args: list[str] | set[str] | str,
    tight: list[bool] | bool = False,
    echo: str | None = None,
    sep: str = "_",
    _to_echo: set[str] | None = None,
) -> set[str]:
    assert len(args) == 2
    assert echo is None or echo in {"left", "right", "both"}
    assert _to_echo is None or echo is None

    args = tuple({arg} if isinstance(arg, str) else arg for arg in args)
    tight = True if isinstance(tight, list) and all(tight) else tight  # type: ignore
    tight = False if isinstance(tight, list) and not any(tight) else tight  # type: ignore
    to_echo: set[str] = set() if _to_echo is None else _to_echo

    assert isinstance(tight, bool) or any([isinstance(arg, list) for arg in args])
    assert isinstance(tight, bool) or any([len(arg) == len(tight) for arg in args])

    larg, rarg = args
    to_echo |= set() if echo is None or echo == "right" else set(larg)
    if echo is not None and echo in {"right", "both"}:
        to_echo |= set(rarg)

    if isinstance(tight, list):
        seps = ["" if val else sep for val in tight]  # type: ignore
        if isinstance(larg, list):
            larg = set(map("".join, zip(larg, seps)))
        else:
            rarg = set(map("".join, zip(seps, rarg)))
        return stitch(larg, rarg, tight=True, _to_echo=to_echo)

    return to_echo | set(map("".join if tight else sep.join, product(larg, rarg)))
