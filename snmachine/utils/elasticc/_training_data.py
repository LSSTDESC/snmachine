from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Set, Tuple, Union
from warnings import warn

import tqdm
from astropy.table import Table
from pandas import DataFrame, Series, concat

from snmachine.sndata import default_pb_wavelengths

FNAME_TMPL = ("ELASTICC2_TRAIN_02", "NONIaMODEL0-00", "FITS.gz")
BANDS_KEY: Dict[bytes, str] = {
    bytes(f"{band} ", encoding="utf-8"): f"lsst{band.lower()}"
    for band in ["u", "g", "r", "i", "z", "Y", "-"]
}
# NOTE: This dictionary should be higher up in the snmachine somewhere.
SNCOSMO_COLS = {
    "time": "mjd",
    "band": "band",
    "flux": "flux",
    "fluxerr": "flux_error",
    "zp": "zp",
    # "zpsys": "zpsys",
    # "fluxcov": "covar",
}
detected_label = "detected"
zeroed_mjds_label = "days_since_detect"
src_class_label = "sim_src_class"
data_cols_key: Dict[str, str] = {
    "MJD": SNCOSMO_COLS["time"],
    "BAND": SNCOSMO_COLS["band"],
    "FLUXCAL": SNCOSMO_COLS["flux"],
    "FLUXCALERR": SNCOSMO_COLS["fluxerr"],
    "ZEROPT": SNCOSMO_COLS["zp"],
    "PHOTFLAG": detected_label,
}
data_cols_key["ZEROPT_ERR"] = "zp_error"
base_data_cols = set(data_cols_key.values())
derived_data_cols = {zeroed_mjds_label, src_class_label}

# FIXME: Using 'Dict' and 'List' etc. is deprecated. Use `type` from 3.12 onward…
StrSpec = Union[List[str], Set[str], str]
DFDict = Dict[str, DataFrame]
DFList = List[DataFrame]
DFTuple = Tuple[DataFrame, ...]
TableDict = Dict[str, Table]


class ElasticcTrainingData:
    SURVEY_NAME = "lsst"
    PB_WAVELENGTHS: Mapping = default_pb_wavelengths[SURVEY_NAME]
    FILTER_SET = tuple(PB_WAVELENGTHS)
    from ._training_metadata import ALL_DATA_COLS, ALL_SRC_CLASSES, SRC_CLASS_TAXONOMY

    assert set(data_cols_key.keys()).issubset(ALL_DATA_COLS)

    def __init__(
        self,
        src_classes: StrSpec,
        root_dir: str | Path,
        add_data_cols: StrSpec = "none",
        min_incl_obs: int = 1,
        only_detected: bool = False,
        add_zeroed_mjds: bool = True,
        sort_data_cols: bool = False,
        add_src_class_col: bool = True,
        quiet_load: bool = False,
    ) -> None:
        self.root_dir: Path = Path(root_dir) if isinstance(root_dir, str) else root_dir
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Specified root_dir does not exist:\n{root_dir}")

        use_all = isinstance(src_classes, str) and src_classes.lower() == "all"
        self.src_classes: Set[str] = (
            self.ALL_SRC_CLASSES if use_all else self._parse_src_classes(src_classes)
        )

        self.dropped_data_cols: Set[str] = self.ALL_DATA_COLS - (
            set(data_cols_key.keys()) | self._parse_add_cols_spec(add_data_cols)
        )
        self.dropped_metadata_cols = {"NOBS", "PTROBS_MIN", "PTROBS_MAX"}

        self.min_incl_obs: int = min_incl_obs
        self.only_detected: bool = only_detected
        self.zeroed: bool = add_zeroed_mjds
        self._sorted_data_cols: bool = sort_data_cols
        self._add_src_class_col: bool = add_src_class_col
        self._quiet_load: bool = quiet_load

        self.metadata: DataFrame
        self.excluded_srcs: Dict[str, Set[str]] = {}
        self.data: TableDict = {}
        self._load_data()

    def __len__(self):
        return len(self.data)

    def _load_data(self) -> None:
        heads: DFList = []
        src_classes = tqdm.tqdm(
            self.src_classes,
            desc="Classes loaded",
            leave=False,
            disable=any([len(self.src_classes) < 2, self._quiet_load]),
        )
        for src_class in src_classes:
            self._load_class(src_class, heads)
        self.metadata = concat(heads)

    def _load_class(self, src_class: str, heads: DFList) -> None:
        src_class_dir: Path = self.root_dir / f"{FNAME_TMPL[0]}_{src_class}"
        if not src_class_dir.is_dir():
            raise FileNotFoundError(
                f"Specified src_class_dir does not exist:\n{src_class_dir}"
            )
        core_head: DataFrame
        core_phot: DataFrame
        core_heads: DFList = []
        excl_srcs: Set[str] = set()
        core_nums = range(1, 41)
        core_nums_ = tqdm.tqdm(
            core_nums, desc=src_class, leave=False, disable=self._quiet_load
        )
        for icore in core_nums_:
            core_head, core_phot = self._load_core(icore, src_class_dir)
            self.data.update(self._parse_core(core_head, core_phot, excl_srcs))
            core_head.drop(columns=self.dropped_metadata_cols, inplace=True)
            core_heads.append(core_head)
        if excl_srcs:
            self.excluded_srcs[src_class] = excl_srcs
        full_core_head = concat(core_heads)
        if self._add_src_class_col:
            full_core_head.insert(
                loc=len(full_core_head.columns),
                column=src_class_label,
                value=[src_class] * len(full_core_head),
            )
        heads.append(full_core_head)

    def _load_core(self, icore: int, src_class_dir: Path) -> DFTuple:
        fname_core_tmpl: str = f"{FNAME_TMPL[0]}_{FNAME_TMPL[1]}{icore:02d}"
        core_fpaths: Dict[str, Path] = {
            key: src_class_dir / f"{fname_core_tmpl}_{key.upper()}.{FNAME_TMPL[2]}"
            for key in ["head", "phot"]
        }
        assert all([core_fpath.is_file() for core_fpath in core_fpaths.values()])
        return self._parse_core_dfs(
            **{
                key: Table.read(core_fpath).to_pandas()
                for key, core_fpath in core_fpaths.items()
            }
        )

    def _parse_core_dfs(self, head: DataFrame, phot: DataFrame) -> DFTuple:
        head.rename(columns={"SNID": "object_id"}, inplace=True)
        object_ids: List[str] = [
            snid.decode().strip() for snid in head.pop("object_id")
        ]
        head.insert(loc=0, column="object_id", value=object_ids)
        head.set_index("object_id", inplace=True)

        phot.rename(columns=data_cols_key, inplace=True)
        bands: List[str] = [BANDS_KEY[band] for band in phot.pop(SNCOSMO_COLS["band"])]
        detecteds: Series = (phot.pop(detected_label) > 0).astype(int)
        phot.insert(loc=1, column=SNCOSMO_COLS["band"], value=bands)
        phot.insert(loc=2, column=detected_label, value=detecteds)
        if self._sorted_data_cols:
            phot.insert(
                loc=3, column=SNCOSMO_COLS["flux"], value=phot.pop(SNCOSMO_COLS["flux"])
            )
            phot.insert(
                loc=4,
                column=SNCOSMO_COLS["fluxerr"],
                value=phot.pop(SNCOSMO_COLS["fluxerr"]),
            )

        if self.dropped_data_cols:
            phot.drop(columns=self.dropped_data_cols, inplace=True)
        return head, phot

    def _parse_core(
        self, core_head: DataFrame, core_phot: DataFrame, excl_srcs: Set[str]
    ) -> TableDict:
        core_data: TableDict = {}
        core_excl_srcs: Set[str] = set()
        src_head: Series
        for object_id, src_head in core_head.iterrows():
            assert isinstance(object_id, str)
            nobs: int = src_head.loc["NOBS"]
            src_phot_view = core_phot[:nobs]
            assert isinstance(src_phot_view, DataFrame)
            src_phot: DataFrame | None = (
                self._extract_src_phot(src_phot_view)
                if len(src_phot_view) >= self.min_incl_obs
                else None
            )
            if src_phot is not None:
                core_data[object_id] = Table.from_pandas(src_phot)
            else:
                core_excl_srcs.add(object_id)
            core_phot.drop(index=core_phot.index[: nobs + 1], inplace=True)
        if core_excl_srcs:
            core_head.drop(index=core_excl_srcs, inplace=True)
            excl_srcs.update(core_excl_srcs)
        return core_data

    def _extract_src_phot(self, src_phot_view: DataFrame) -> DataFrame | None:
        src_phot_detected = src_phot_view.query(f"{detected_label} == 1")
        assert isinstance(src_phot_detected, DataFrame)
        if self.only_detected and len(src_phot_detected) < self.min_incl_obs:
            return None
        src_phot = src_phot_detected if self.only_detected else src_phot_view.copy()
        assert isinstance(src_phot, DataFrame)

        if self.zeroed:
            self._insert_days_since_detection(src_phot, src_phot_detected)
        return src_phot

    # NOTE: The 'MJD_DETECT_FIRST' and 'MJD_TRIGGER' fields in head files can't be trusted.
    def _insert_days_since_detection(
        self, src_phot: DataFrame, src_phot_only_detected: DataFrame
    ) -> None:
        mjds_detected = src_phot_only_detected[SNCOSMO_COLS["time"]]
        mjds_all = src_phot[SNCOSMO_COLS["time"]]
        assert isinstance(mjds_detected, Series) and isinstance(mjds_all, Series)
        mjd_diffs: Series = mjds_all - mjds_detected.min()
        src_phot.insert(loc=1, column=zeroed_mjds_label, value=mjd_diffs.round(4))

    def _parse_src_classes(self, spec: list[str] | set[str] | str) -> set[str]:
        if not isinstance(spec, set):
            spec = {spec} if isinstance(spec, str) else {*spec}

        src_classes: Set[str] = spec & self.ALL_SRC_CLASSES
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

    def _parse_add_cols_spec(self, add_cols: StrSpec) -> Set[str]:
        if isinstance(add_cols, str):
            add_cols = add_cols.lower()
            if add_cols == "none":
                return base_data_cols
            elif add_cols == "all":
                return self.ALL_DATA_COLS
            else:
                return base_data_cols & {add_cols}
        if not isinstance(add_cols, set):
            add_cols = set(add_cols)

        # TODO: Add check and warn for cols in data_cols_key.keys(); also kinda ignored.
        bad_cols: Set[str] = add_cols - (
            self.ALL_DATA_COLS | set(data_cols_key.values()) | derived_data_cols
        )
        if bad_cols:
            warn(
                f"Ignoring invalid data column labels in add_data_cols:\n{bad_cols}\nValid labels are those in:\n"
                f"{set(data_cols_key.values())}\nand/or\n{self.ALL_DATA_COLS}"
            )
        return base_data_cols & add_cols - bad_cols
