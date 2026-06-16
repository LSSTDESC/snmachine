from itertools import chain
from pathlib import Path
from warnings import warn

from astropy.table import Table

from ...utils import StrSpec, resolve_spec
from .loading import load_training_data
from .metadata import (
    ALL_DATA_COLS,
    ALL_SRC_CLASSES,
    RENAMED_DATA_COLS,
    SRC_CLASS_TAXONOMY,
)

ALL_DATA_COLS_RENAMED = ALL_DATA_COLS - set(RENAMED_DATA_COLS.keys()) | set(
    RENAMED_DATA_COLS.values()
)


class TrainingData:
    data_cols_base = ["mjd", "band", "detected", "flux", "fluxerr", "zp", "zp_error"]
    data_cols_derived = {"days_since_detect"}

    def __init__(
        self,
        src_classes: StrSpec,
        root_dir: str | Path,
        zeroed: bool = True,
        add_data_cols: StrSpec = "none",
        add_src_class_col: bool = True,
        quiet_load: bool = False,
        **kwargs,
    ) -> None:
        self.src_classes: set[str] = self._parse_src_classes(src_classes)
        self.root_dir: Path = Path(root_dir) if isinstance(root_dir, str) else root_dir
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Specified root_dir does not exist:\n{root_dir}")
        self.data_cols: set[str] = set(self.data_cols_base) | self._parse_add_cols_spec(
            add_data_cols
        )

        drop_phot_cols: set[str] = ALL_DATA_COLS_RENAMED - self.data_cols

        # FIXME: The len(thing) < 2 check(s) are missing for all the tqdm calls.
        _bundle = load_training_data(
            self.src_classes,
            root_dir=self.root_dir,
            zeroed=zeroed,
            drop_phot_cols=drop_phot_cols,
            add_src_class_col=add_src_class_col,
            disable=quiet_load,
            **kwargs,
        )
        self.metadata: Table = _bundle.head
        self.data: dict[str, Table] = _bundle.data
        self.excluded_srcs: Table | None = _bundle.excl

    def __len__(self):
        return len(self.data)

    def _parse_src_classes(self, spec: StrSpec) -> set[str]:
        if isinstance(spec, str) and spec == "all":
            return ALL_SRC_CLASSES
        spec = resolve_spec(spec)
        src_classes: set[str] = spec & ALL_SRC_CLASSES
        for spec_str in spec - ALL_SRC_CLASSES:
            src_classes |= self._resolve_spec_str(spec_str.lower())
        return src_classes

    # TODO: Make these checks case insensitive.
    def _resolve_spec_str(self, spec_str: str) -> set[str]:
        if spec_str in SRC_CLASS_TAXONOMY:
            return set(chain(*SRC_CLASS_TAXONOMY[spec_str]))
        for supset_dict in SRC_CLASS_TAXONOMY.values():
            if spec_str in supset_dict:
                return supset_dict[spec_str]
        raise ValueError(
            f"Invalid spec_str: '{spec_str}'. Value must be (equivalent to) 'all' or in:\n"
            f"{sorted(SRC_CLASS_TAXONOMY)}\n"
            f"or\n{sorted(chain(*SRC_CLASS_TAXONOMY.values()))}\n"
            f"or\n{sorted(ALL_SRC_CLASSES)}"
        )

    def _parse_add_cols_spec(self, add_cols: StrSpec) -> set[str]:
        if isinstance(add_cols, str):
            if add_cols.lower() in set(self.data_cols_base) | {"none"}:
                return set()
            if add_cols.lower() == "all":
                return ALL_DATA_COLS_RENAMED - set(self.data_cols_base)
            if add_cols.upper() in ALL_DATA_COLS:
                old_name, add_cols = add_cols, RENAMED_DATA_COLS[add_cols]
                warn(f"{old_name} added to data cols but renamed to {add_cols}.")
            if add_cols.lower() in ALL_DATA_COLS_RENAMED:
                return {add_cols.lower()}
        add_cols = resolve_spec(add_cols)

        assert isinstance(add_cols, set)
        assert all(isinstance(val, str) for val in add_cols)

        if "days_since_detect" in add_cols:
            warn("Ignoring 'days_since_detect' in add_cols_spec. Pass 'zeroed=True'.")
            add_cols.remove("days_since_detect")

        bad_cols: set[str]
        if bad_cols := add_cols - (ALL_DATA_COLS | ALL_DATA_COLS_RENAMED):
            warn(
                f"Ignoring invalid data column labels in add_data_cols:\n{bad_cols}\nValid labels are those in:\n"
                f"{sorted(ALL_DATA_COLS | ALL_DATA_COLS_RENAMED)}"
            )
        return add_cols - (bad_cols | set(self.data_cols_base))
