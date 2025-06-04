
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
