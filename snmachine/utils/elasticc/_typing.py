from __future__ import annotations

from astropy.table import Table
from pandas import DataFrame

StrSpec = list[str] | set[str] | str
# DFDict = dict[str, DataFrame]
DFList = list[DataFrame]
DFTuple = tuple[DataFrame, ...]
TableDict = dict[str, Table]
