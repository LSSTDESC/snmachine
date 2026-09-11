from collections import ChainMap
from copy import copy
from itertools import chain

from ....extra_utils import are_sncosmo_aliases
from ...utils import stitch

BAND_LABELS = ["u", "g", "r", "i", "z", "Y"]


def per_band(stems: list[str] | set[str] | str, **stitch_kwargs) -> set[str]:
    return stitch(stems, set(BAND_LABELS), **stitch_kwargs)


SRC_CLASS_TAXONOMY: dict[str, dict[str, set[str]]] = {
    "Non-Recurring": {
        "SN-like": {"SNIax", "SNII-NMF", "SNIIn-MOSFIT"}
        | {f"SNIa-{val}" for val in ["91bg", "SALT3"]}
        | {f"{val}+HostXT_V19" for val in ["SNIcBL", "SNIIb", "SNIIn"]}
        | stitch(["SNIb", "SNIc", "SNII"], ["-Templates", "+HostXT_V19"], tight=True),
        "Fast": {"Mdwarf-flare", "dwarf-nova"}
        | {f"KN_{val}" for val in ["B19", "K17"]}
        | {f"uLens-{val}" for val in ["Binary", "Single-GenLens", "Single_PyLIMA"]},
        "Long": {"SLSN-I+host", "SLSN-I_no_host", "TDE", "CART", "ILOT", "PISN"},
    },
    "Recurring": {
        "Periodic": {"Cepheid", "d-Sct", "EB", "RRL"},
        "Non-Periodic": {"CLAGN"},
    },
}
ALL_SRC_CLASSES: set[str] = set(chain(*ChainMap(*SRC_CLASS_TAXONOMY.values()).values()))

ALL_DATA_COLS: set[str] = (
    {"BAND", "CCDNUM", "FIELD", "GAIN", "MJD", "RDNOISE", "SKY_SIG", "SKY_SIG_T"}
    | {f"{val}PIX" for val in ["X", "Y"]}
    | {f"PSF_{val}" for val in ["RATIO", "SIG1", "SIG2"]}
    | {f"SIM_{val}" for val in ["MAGOBS", "FLUXCAL_HOSTERR"]}
    | {f"PHOT{val}" for val in ["FLAG", "PROB"]}
    | stitch(["FLUXCAL", "ZEROPT"], "ERR", tight=[True, False], echo="left")
)
RENAMED_DATA_COLS = {
    "MJD": "mjd",
    "BAND": "band",
    "FLUXCAL": "flux",
    "FLUXCALERR": "fluxerr",
    "ZEROPT": "zp",
}
assert are_sncosmo_aliases(set(RENAMED_DATA_COLS.values()))
RENAMED_DATA_COLS |= {"PHOTFLAG": "detected", "ZEROPT_ERR": "zp_error"}

MIXIN_METADATA_COLS: dict[tuple[str, ...], set[str]] = {
    tuple(
        f"AGN_PARAM({val})"
        for val in sorted(
            ["M_BH", "Mi", "cl_flag", "edd_ratio", "edd_ratio2", "t_transition"]
        )
    ): SRC_CLASS_TAXONOMY["Recurring"]["Non-Periodic"],
    tuple(
        f"SIM_SALT2{val}"
        for val in sorted(["alpha", "beta", "c", "gammaDM", "mB", "x0", "x1"])
    ): {"SNIa-SALT3"},
    tuple(
        f"SIM_HOSTLIB({val})" for val in sorted(["LOGMASS_TRUE", "LOG_SFR"])
    ): SRC_CLASS_TAXONOMY["Non-Recurring"]["SN-like"]
    | {"TDE", "SLSN-I+host"},
    tuple(f"SIM_HOSTLIB({val}_obs)" for val in sorted(["g", "i", "r"])): {
        f"KN_{val}" for val in ["B19", "K17"]
    },
    tuple(sorted(per_band("SIM_TEMPLATEMAG"))): (
        SRC_CLASS_TAXONOMY["Non-Recurring"]["Fast"]
        - {f"KN_{val}" for val in ["B19", "K17"]}
        | set(chain(*SRC_CLASS_TAXONOMY["Recurring"].values()))
    ),
    tuple(
        sorted(
            per_band("SIM_GALFRAC")
            | stitch(
                ["HOSTGAL", "HOSTGAL2"],
                [f"ZPHOT_Q{num:03}" for num in range(0, 101, 10)],
            )
        )
    ): SRC_CLASS_TAXONOMY["Non-Recurring"]["SN-like"]
    | {"TDE", "SLSN-I+host", "PISN", "CART", "CLAGN", "ILOT"}
    | {f"KN_{val}" for val in ["B19", "K17"]},
}

_mdata_gal2_has_err: list[str] = ["COLOR", "PHOTOZ", "SPECZ"] + [
    f"LOG{val}" for val in ["MASS", "SFR", "sSFR"]
]
_mdata_gal2: list[str] = (
    ["FLAG", "RA", "DEC", "SNSEP", "DDLR", "ELLIPTICITY", "SQRADIUS"]
    + ["OBJID", "OBJID2", "OBJID_UNIQUE"]
    + list(stitch(_mdata_gal2_has_err, "ERR", echo="left"))
    + list(per_band(stitch("MAG", "ERR", tight=True, echo="left")))
)
_mdata_gal1: list[str] = (
    _mdata_gal2 + ["CONFUSION", "NMATCH", "NMATCH2"] + list(per_band("SB_FLUXCAL"))
)
_mdata_sim: list[str] = (
    ["RA", "DEC", "AV", "RV", "VPEC", "HOSTLIB_GALID", "NOBS_UNDEFINED"]
    + ["DLMU", "LENSDMU", "MAGSMEAR_COH", "MWEBV", "SEARCHEFF_MASK"]
    + ["MJD_EXPLODE", "PEAKMJD", "LIBID", "NGEN_LIBID"]
    + [f"{val}_INDEX" for val in ["TEMPLATE", "SUBSAMPLE"]]
    + [f"REDSHIFT_{val}" for val in ["HELIO", "CMB", "HOST", "FLAG"]]
    + list(stitch(["MODEL", "TYPE"], ["NAME", "INDEX"]))
    + list(per_band(["PEAKMAG", "EXPOSURE"]))
)
_mdata_has_err: list[str] = ["MWEBV", "VPEC"] + [
    f"REDSHIFT_{val}" for val in ["HELIO", "FINAL"]
]
DOOMED_METADATA_COLS = {"NOBS", "PTROBS_MIN", "PTROBS_MAX", "SNID"}
COMMON_METADATA_COLS: set[str] = (
    DOOMED_METADATA_COLS
    | {"IAUC", "FAKE", "RA", "DEC", "PIXSIZE", "SNTYPE", "SEARCH_TYPE"}
    | {"PEAKMJD", "MJD_TRIGGER"}
    | {f"MJD_DETECT_{val}" for val in ["FIRST", "LAST"]}
    | {f"N{val}PIX" for val in ["X", "Y"]}
    | {f"HOSTGAL_{val}" for val in _mdata_gal1}
    | {f"HOSTGAL2_{val}" for val in _mdata_gal2}
    | {f"SIM_{val}" for val in _mdata_sim}
    | stitch(_mdata_has_err, "ERR", echo="left")
)
ALL_METADATA_COLS: set[str] = COMMON_METADATA_COLS | set(chain(*MIXIN_METADATA_COLS))
RENAMED_METADATA_COLS = {"SNID": "object_id"}


def resolve_metadata_cols(src_class: str) -> set[str]:
    mdata_cols = copy(COMMON_METADATA_COLS)
    for mixin, src_classes in MIXIN_METADATA_COLS.items():
        if src_class in src_classes:
            mdata_cols |= set(mixin)
    return mdata_cols - DOOMED_METADATA_COLS
