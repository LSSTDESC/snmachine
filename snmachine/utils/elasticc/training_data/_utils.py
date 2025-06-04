from itertools import product
from pathlib import Path
from typing import Sequence

FNAME_TMPL = ("ELASTICC2_TRAIN_02", "NONIaMODEL0-00", "FITS.gz")
FNAME_BASE = "_".join(FNAME_TMPL[:2])
StrSpec = Sequence[str] | set[str] | str


def resolve_spec(spec: StrSpec) -> set[str]:
    if isinstance(spec, set):
        return spec
    return {spec} if isinstance(spec, str) else set(spec)


def src_class_dir(root_dir: Path, src_class: str) -> Path:
    return root_dir / f"{FNAME_TMPL[0]}_{src_class}"


def fits_path(src_class_dir: Path, icore: int, key: str) -> Path:
    return src_class_dir / f"{FNAME_BASE}{icore:02d}_{key.upper()}.{FNAME_TMPL[2]}"


def key_to_rename_spec(key_dict: dict[str, str]) -> dict[str, tuple[str]]:
    return dict(zip(["names", "new_names"], zip(*key_dict.items())))


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
    tight = True if isinstance(tight, list) and all(tight) else tight
    tight = False if isinstance(tight, list) and not any(tight) else tight
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
