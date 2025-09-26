from __future__ import annotations

from collections import ChainMap
from pathlib import Path
from typing import NamedTuple

from astropy.table import Table, vstack

FNAME_TMPL = ("ELASTICC2_TRAIN_02", "NONIaMODEL0-00", "FITS.gz")
FNAME_BASE = "_".join(FNAME_TMPL[:2])


class DataBundle(NamedTuple):
    head: Table
    data: dict[str, Table]
    excl: Table | None

    @classmethod
    def from_bundles(cls, bundles: list[DataBundle]) -> DataBundle:
        excls = [bundle.excl for bundle in bundles if bundle.excl is not None]
        return cls(
            head=vstack([bundle.head for bundle in bundles]),
            data=dict(ChainMap(*[bundle.data for bundle in bundles])),
            excl=vstack(excls) if excls else None,
        )


def src_class_dir(root_dir: Path, src_class: str) -> Path:
    return root_dir / f"{FNAME_TMPL[0]}_{src_class}"


def fits_path(src_class_dir: Path, icore: int, key: str) -> Path:
    return src_class_dir / f"{FNAME_BASE}{icore:02d}_{key.upper()}.{FNAME_TMPL[2]}"
