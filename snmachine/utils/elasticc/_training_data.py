from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Generator, Iterator
from warnings import warn

import numpy as np
from astropy.table import Table
from numpy.typing import NDArray
from tqdm import tqdm

from ...sndata import default_pb_wavelengths
from .._utils import are_sncosmo_aliases
from ._training_metadata import BAND_LABELS
from ._typing import DataBundle, StrSpec, resolve_spec

_read_table = partial(Table.read, character_as_bytes=False, memmap=False)

FNAME_TMPL = ("ELASTICC2_TRAIN_02", "NONIaMODEL0-00", "FITS.gz")
FNAME_BASE = "_".join(FNAME_TMPL[:2])
BANDS_KEY: dict[str, str] = {f"{band} ": f"lsst{band.lower()}" for band in BAND_LABELS}

_renamed_data_cols = {
    "MJD": "mjd",
    "BAND": "band",
    "FLUXCAL": "flux",
    "FLUXCALERR": "fluxerr",
    "ZEROPT": "zp",
}
assert are_sncosmo_aliases(set(_renamed_data_cols.values()))
_renamed_data_cols |= {"PHOTFLAG": "detected", "ZEROPT_ERR": "zp_error"}
RENAMED_DATA_COLS: dict[str, tuple] = dict(
    zip(["names", "new_names"], zip(*_renamed_data_cols.items()))
)

_renamed_mdata_cols = {"SNID": "object_id"}
RENAMED_MDATA_COLS: dict[str, tuple] = dict(
    zip(["names", "new_names"], zip(*_renamed_mdata_cols.items()))
)


class ElasticcTrainingData:
    SURVEY_NAME = "lsst"
    FILTER_SET = tuple(default_pb_wavelengths[SURVEY_NAME])
    from ._training_metadata import (
        ALL_DATA_COLS,
        ALL_METADATA_COLS,
        ALL_SRC_CLASSES,
        SRC_CLASS_TAXONOMY,
    )

    data_cols_key: dict[str, str] = {
        "MJD": "mjd",
        "BAND": "band",
        "FLUXCAL": "flux",
        "FLUXCALERR": "fluxerr",
        "ZEROPT": "zp",
    }
    assert are_sncosmo_aliases(set(data_cols_key.values()))
    data_cols_key |= {"PHOTFLAG": "detected", "ZEROPT_ERR": "zp_error"}
    base_data_cols = set(data_cols_key.values())
    derived_data_cols = {"days_since_detect", "src_class"}
    assert set(data_cols_key.keys()).issubset(ALL_DATA_COLS)

    def __init__(
        self,
        src_classes: StrSpec,
        root_dir: str | Path,
        add_data_cols: StrSpec = "none",
        quiet_load: bool = False,
        **kwargs,
    ) -> None:
        use_all: bool = isinstance(src_classes, str) and src_classes.lower() == "all"
        self.src_classes: set[str] = (
            self.ALL_SRC_CLASSES if use_all else self._parse_src_classes(src_classes)
        )
        self.root_dir: Path = Path(root_dir) if isinstance(root_dir, str) else root_dir
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Specified root_dir does not exist:\n{root_dir}")

        drop_phot_cols: set[str] = self.ALL_DATA_COLS - (
            set(self.data_cols_key.keys()) | self._parse_add_cols_spec(add_data_cols)
        )

        # FIXME: The len(thing) < 2 check(s) are missing for all the tqdm calls.
        _bundle = _load_training_data(
            self.src_classes,
            root_dir=self.root_dir,
            drop_phot_cols=drop_phot_cols,
            disable=quiet_load,
            **kwargs,
        )
        self.metadata: Table = _bundle.head
        self.data: dict[str, Table] = _bundle.data
        self.excluded_srcs: Table | None = _bundle.excl

    def __len__(self):
        return len(self.data)

    def _parse_src_classes(self, spec: StrSpec) -> set[str]:
        if not isinstance(spec, set):
            spec = {spec} if isinstance(spec, str) else set(spec)

        src_classes: set[str] = spec & self.ALL_SRC_CLASSES
        for spec_str in spec - self.ALL_SRC_CLASSES:
            src_classes |= self._resolve_spec_str(spec_str.lower())
        return src_classes

    # TODO: Make these checks case insensitive.
    def _resolve_spec_str(self, spec_str: str) -> set[str]:
        if spec_str in self.SRC_CLASS_TAXONOMY.keys():
            supset_dict = self.SRC_CLASS_TAXONOMY[spec_str]
            src_classes = set()
            for supset in supset_dict.values():
                src_classes |= supset
            return src_classes
        for supset_dict in self.SRC_CLASS_TAXONOMY.values():
            if spec_str in supset_dict.keys():
                return supset_dict[spec_str]
        else:
            supset_keys = []
            loop_keys = sorted(self.SRC_CLASS_TAXONOMY.keys())
            for key in loop_keys:
                supset_keys += sorted(self.SRC_CLASS_TAXONOMY[key].keys())
            raise ValueError(
                f"Invalid spec_str: '{spec_str}'. Value must be (equivalent to) 'all' or in:\n"
                f"{loop_keys}\n"
                f"or\n{supset_keys}\n"
                f"or\n{sorted(self.ALL_SRC_CLASSES)}"
            )

    def _parse_add_cols_spec(self, add_cols: StrSpec) -> set[str]:
        if isinstance(add_cols, str):
            add_cols = add_cols.lower()
            if add_cols == "none":
                return self.base_data_cols
            elif add_cols == "all":
                return self.ALL_DATA_COLS
            else:
                return self.base_data_cols & {add_cols}
        if not isinstance(add_cols, set):
            add_cols = set(add_cols)

        # TODO: Add check and warn for cols in data_cols_key.keys(); also kinda ignored.
        bad_cols: set[str] = add_cols - (
            self.ALL_DATA_COLS
            | set(self.data_cols_key.values())
            | self.derived_data_cols
        )
        if bad_cols:
            warn(
                f"Ignoring invalid data column labels in add_data_cols:\n{bad_cols}\nValid labels are those in:\n"
                f"{set(self.data_cols_key.values())}\nand/or\n{self.ALL_DATA_COLS}"
            )
        return self.base_data_cols & add_cols - bad_cols


def _load_training_data(src_classes: set[str], **kwargs) -> DataBundle:
    tqdm_spec: dict[str, Any] = dict(desc="Classes loaded", leave=False)
    tqdm_keys: set[str] = {"leave"}
    tqdm_spec |= {key: kwargs.get(key) for key in set(kwargs) & tqdm_keys}
    tqdm_spec["disable"] = tqdm_spec.get("disable", False) or len(src_classes) < 2

    src_class_bundles: list[DataBundle] = [
        load_src_class(src_class, **kwargs)
        for src_class in tqdm(src_classes, **tqdm_spec)
    ]
    return DataBundle.from_bundles(src_class_bundles)


def _add_src_class_col(bundle: DataBundle, src_class: str) -> None:
    for tbl in [bundle.head, bundle.excl]:
        if tbl is not None:
            tbl.add_column(src_class, name="src_class")


def _src_class_dir(root_dir: Path, src_class: str) -> Path:
    return root_dir / f"{FNAME_TMPL[0]}_{src_class}"


def load_src_class(
    src_class: str,
    root_dir: Path,
    add_src_class_col: bool = True,
    num_workers: int = 1,
    chunksize: int = 1,
    **kwargs,
) -> DataBundle:
    if not (src_class_dir := _src_class_dir(root_dir, src_class)).is_dir():
        raise FileNotFoundError(f"src_class_dir for {src_class} not found.")
    assert num_workers > 0
    assert chunksize > 0

    tqdm_spec: dict[str, Any] = dict(desc="Core files loaded", total=40, leave=False)
    tqdm_keys: set[str] = {"leave", "disable"}
    tqdm_spec |= {key: kwargs.get(key) for key in set(kwargs) & tqdm_keys}

    bundler: Callable[[tuple[Table, Table]], DataBundle] = partial(
        _bundle_core_tbls, **kwargs
    )
    core_tables: Iterator[tuple[Table, Table]] = tqdm(
        _load_core_tables(src_class_dir), **tqdm_spec
    )  # type: ignore

    with Pool(num_workers) as pool:
        core_bundles: list[DataBundle] = list(
            pool.imap(bundler, core_tables, chunksize)
        )
    out_bundle: DataBundle = DataBundle.from_bundles(core_bundles)
    if add_src_class_col:
        _add_src_class_col(out_bundle, src_class)
    return out_bundle


def _bundle_core_tbls(tbls: tuple[Table, Table], **kwargs) -> DataBundle:
    head: Table
    phot: Table
    head, phot = tbls
    _format_head(head, **kwargs)
    _format_phot(phot, **kwargs)
    src_phots: list[Table | None] = [
        _parse_src_phot(src_phot, **kwargs) for src_phot in _devour_phot(phot, head)
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


# NOTE: The 'MJD_DETECT_FIRST' and 'MJD_TRIGGER' fields in head files can't be trusted.
def _parse_src_phot(
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
def _devour_phot(phot: Table, head: Table) -> Generator[Table, None, None]:
    for (nobs,) in head.iterrows("NOBS"):
        yield phot[list(range(nobs))]  # type: ignore
        phot.remove_rows(slice(nobs))


def _load_core_tables(
    src_class_dir: Path,
) -> Generator[tuple[Table, Table], None, None]:
    fits_path = partial(_fits_path, src_class_dir)
    for icore in range(1, 41):
        yield _read_table(fits_path(icore, "head")), _read_table(
            fits_path(icore, "phot")
        )


def _fits_path(src_class_dir: Path, icore: int, key: str) -> Path:
    return src_class_dir / f"{FNAME_BASE}{icore:02d}_{key.upper()}.{FNAME_TMPL[2]}"


def _format_head(head: Table, drop_head_cols: StrSpec | None = None, **_) -> None:
    head.rename_columns(**RENAMED_MDATA_COLS)
    drop_cols_: set[str] | None = _resolve_drop_cols(
        drop_head_cols, base={"PTROBS_MIN", "PTROBS_MAX"}, protected={"object_id"}
    )
    assert drop_cols_ is not None
    head.remove_columns(drop_cols_)
    oids: Iterator[str] = map(str.strip, head.columns["object_id"])
    head.replace_column("object_id", list(oids))


def _format_phot(phot: Table, drop_phot_cols: StrSpec | None = None, **_) -> None:
    phot.rename_columns(**RENAMED_DATA_COLS)
    drop_cols_: set[str] | None = _resolve_drop_cols(
        drop_phot_cols, protected={"band", "detected"}
    )
    if drop_cols_ is not None:
        phot.remove_columns(drop_cols_)
    phot.remove_rows(phot.columns["band"] == "- ")
    phot.replace_column("band", list(map(BANDS_KEY.get, phot.columns["band"])))
    phot.replace_column("detected", phot.columns["detected"].astype(bool).astype(int))  # type: ignore


def _resolve_drop_cols(
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
