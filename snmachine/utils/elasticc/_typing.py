from __future__ import annotations

from typing import ChainMap, NamedTuple, Sequence

from astropy.table import Table, vstack

StrSpec = Sequence[str] | set[str] | str


def resolve_spec(spec: StrSpec) -> set[str]:
    if isinstance(spec, set):
        return spec
    return {spec} if isinstance(spec, str) else set(spec)


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
