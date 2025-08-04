from __future__ import annotations

from collections import ChainMap
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, NamedTuple
from warnings import warn

import numpy as np
from astropy.table import Table, vstack
from numpy.typing import NDArray
from tqdm import tqdm

from ..utils import StrSpec, key_to_rename_spec, resolve_spec
from .metadata import BAND_LABELS, RENAMED_DATA_COLS, RENAMED_METADATA_COLS
from .utils import fits_path, src_class_dir

read_table = partial(Table.read, character_as_bytes=False, memmap=False)
BANDS_KEY: dict[str, str] = {f"{band} ": f"lsst{band.lower()}" for band in BAND_LABELS}


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


def load_training_data(src_classes: set[str], **kwargs) -> DataBundle:
    tqdm_spec: dict[str, Any] = dict(desc="Classes loaded", leave=False)
    tqdm_keys: set[str] = {"leave"}
    tqdm_spec |= {key: kwargs.get(key) for key in set(kwargs) & tqdm_keys}
    tqdm_spec["disable"] = tqdm_spec.get("disable", False) or len(src_classes) < 2

    src_class_bundles: list[DataBundle] = [
        load_src_class(src_class, **kwargs)
        for src_class in tqdm(src_classes, **tqdm_spec)
    ]
    return DataBundle.from_bundles(src_class_bundles)


def load_src_class(
    src_class: str,
    root_dir: Path,
    add_src_class_col: bool = True,
    num_workers: int = 1,
    chunksize: int = 1,
    **kwargs,
) -> DataBundle:
    if not (cores_dir := src_class_dir(root_dir, src_class)).is_dir():
        raise FileNotFoundError(f"src_class_dir for {src_class} not found.")
    assert num_workers > 0
    assert chunksize > 0

    tqdm_spec: dict[str, Any] = dict(desc="Core files loaded", total=40, leave=False)
    tqdm_keys: set[str] = {"leave", "disable"}
    tqdm_spec |= {key: kwargs.get(key) for key in set(kwargs) & tqdm_keys}

    bundler: Callable[[tuple[Table, Table]], DataBundle] = partial(
        bundle_core_tbls, **kwargs
    )
    core_tables: Iterator[tuple[Table, Table]] = tqdm(
        load_core_tables(cores_dir), **tqdm_spec
    )  # type: ignore

    with Pool(num_workers) as pool:
        core_bundles: list[DataBundle] = list(
            pool.imap(bundler, core_tables, chunksize)
        )
    out_bundle: DataBundle = DataBundle.from_bundles(core_bundles)
    if add_src_class_col:
        for tbl in [out_bundle.head, out_bundle.excl]:
            if tbl is not None:
                tbl.add_column(src_class, name="src_class")
    return out_bundle


def bundle_core_tbls(tbls: tuple[Table, Table], **kwargs) -> DataBundle:
    head: Table
    phot: Table
    head, phot = tbls
    format_head(head, **kwargs)
    format_phot(phot, **kwargs)
    src_phots: list[Table | None] = [
        parse_src_phot(src_phot, **kwargs) for src_phot in devour_phot(phot, head)
    ]
    excl_idxs: list[int] = [
        idx for idx, src_phot in enumerate(src_phots) if src_phot is None
    ]
    excl: Table | None = head[excl_idxs] if excl_idxs else None  # type: ignore
    if excl_idxs:
        head.remove_columns(excl_idxs)
        for idx in reversed(excl_idxs):
            src_phots.pop(idx)

    assert None not in src_phots
    assert len(src_phots) == len(head)

    data: dict[str, Table] = dict(zip(head.columns["object_id"], src_phots))  # type: ignore
    return DataBundle(head, data, excl)


def format_head(head: Table, drop_head_cols: StrSpec | None = None, **_) -> None:
    head.rename_columns(**key_to_rename_spec(RENAMED_METADATA_COLS))
    drop_cols: set[str] | None = resolve_drop_cols(
        drop_head_cols, base={"PTROBS_MIN", "PTROBS_MAX"}, protected={"object_id"}
    )
    assert drop_cols is not None
    head.remove_columns(drop_cols & set(head.columns))
    oids: Iterator[str] = map(str.strip, head.columns["object_id"])
    head.replace_column("object_id", list(oids))


def format_phot(phot: Table, drop_phot_cols: StrSpec | None = None, **_) -> None:
    phot.rename_columns(**key_to_rename_spec(RENAMED_DATA_COLS))
    drop_cols: set[str] | None = resolve_drop_cols(
        drop_phot_cols, protected={"band", "detected"}
    )
    if drop_cols is not None:
        phot.remove_columns(drop_cols)
    phot.remove_rows(phot.columns["band"] == "- ")
    phot.replace_column("band", list(map(BANDS_KEY.get, phot.columns["band"])))
    phot.replace_column("detected", phot.columns["detected"].astype(bool).astype(int))  # type: ignore


# NOTE: The 'MJD_DETECT_FIRST' and 'MJD_TRIGGER' fields in head files can't be trusted.
def parse_src_phot(
    src_phot: Table,
    only_detected: bool = False,
    zeroed: bool = True,
    min_incl_obs: int = 1,
    **_,
) -> Table | None:
    assert min_incl_obs > 0
    detections: NDArray[np.int_]
    detections = np.nonzero(np.array(src_phot.columns["detected"]))[0]
    n_incl_obs: int = len(detections if only_detected else src_phot)
    if n_incl_obs < min_incl_obs:
        return None
    if not (only_detected or zeroed):
        return src_phot
    if not zeroed:
        return src_phot[detections]  # type: ignore

    mjds_detect = np.array(src_phot.columns["mjd"][detections])
    if only_detected:
        src_phot = src_phot[detections]  # type: ignore
    src_phot.add_column(src_phot["mjd"] - mjds_detect[0], name="days_since_detect")
    return src_phot


# NOTE: The hack-ey slicing of phot prevents the yielded Table from referencing phot.
#       (Passing such referenced copies causes memory meltdowns.)
#       Unfortunately, astropy is a hot mess, so there's no knowing why this works, and
#       no sensible type hinting.
def devour_phot(phot: Table, head: Table) -> Generator[Table, None, None]:
    for (nobs,) in head.iterrows("NOBS"):
        yield phot[list(range(nobs))]  # type: ignore
        phot.remove_rows(slice(nobs))


def load_core_tables(src_class_dir: Path) -> Generator[tuple[Table, Table], None, None]:
    fits_path_ = partial(fits_path, src_class_dir)
    for icore in range(1, 41):
        yield (
            read_table(fits_path_(icore, "head")),
            read_table(fits_path_(icore, "phot")),
        )


def resolve_drop_cols(
    drop_cols: StrSpec | None,
    base: set[str] | None = None,
    protected: set[str] | None = None,
) -> set[str] | None:
    if drop_cols is None and base is None:
        return None
    all_drop_cols: set[str] = set() if base is None else base
    if drop_cols is not None:
        all_drop_cols |= resolve_spec(drop_cols)
    if protected is None:
        return all_drop_cols
    if __debug__:
        if base is not None and protected is not None:
            assert not base & protected
    for colname in protected:
        if colname in all_drop_cols:
            warn(f'Ignoring "{colname}" in drop_cols')
    return all_drop_cols - protected
