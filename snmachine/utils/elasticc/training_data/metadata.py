from collections import ChainMap
from itertools import chain

from ..._utils import are_sncosmo_aliases
from ._utils import stitch

BAND_LABELS = ["u", "g", "r", "i", "z", "Y"]


def per_band(stems: set[str] | str, **stitch_kwargs) -> set[str]:
    return stitch(stems, set(BAND_LABELS), **stitch_kwargs)


SRC_CLASS_TAXONOMY: dict[str, dict[str, set[str]]] = {
    "Non-Recurring": {
        "SN-like": {
            *{"SNIax", "SNII-NMF", "SNIIn-MOSFIT"},
            *stitch("SNIa", {"91bg", "SALT3"}, sep="-"),
            *stitch({"SNIb", "SNIc", "SNII"}, "Templates", sep="-"),
            *stitch({"SNIb", "SNIc", "SNII"}, "HostXT_V19", sep="+"),
            *stitch({"SNIcBL", "SNIIb", "SNIIn"}, "HostXT_V19", sep="+"),
        },
        "Fast": {
            *{"Mdwarf-flare", "dwarf-nova"},
            *stitch("KN", {"B19", "K17"}),
            *stitch("uLens", {"Binary", "Single-GenLens", "Single_PyLIMA"}, sep="-"),
        },
        "Long": {"SLSN-I+host", "SLSN-I_no_host", "TDE", "CART", "ILOT", "PISN"},
    },
    "Recurring": {
        "Periodic": {"Cepheid", "d-Sct", "EB", "RRL"},
        "Non-Periodic": {"CLAGN"},
    },
}
ALL_SRC_CLASSES: set[str] = set(chain(*ChainMap(*SRC_CLASS_TAXONOMY.values()).values()))

ALL_DATA_COLS: set[str] = {
    *{"BAND", "CCDNUM", "FIELD", "GAIN", "MJD", "RDNOISE"},
    *stitch({"X", "Y"}, "PIX", tight=True),
    *stitch(["FLUXCAL", "ZEROPT"], "ERR", tight=[True, False], echo="left"),
    *stitch("PSF", {"RATIO", "SIG1", "SIG2"}),
    *stitch("SKY_SIG", "T", echo="left"),
    *stitch("SIM", {"MAGOBS", "FLUXCAL_HOSTERR"}),
    *stitch("PHOT", {"FLAG", "PROB"}, tight=True),
}
RENAMED_DATA_COLS = {
    "MJD": "mjd",
    "BAND": "band",
    "FLUXCAL": "flux",
    "FLUXCALERR": "fluxerr",
    "ZEROPT": "zp",
}
assert are_sncosmo_aliases(set(RENAMED_DATA_COLS.values()))
RENAMED_DATA_COLS |= {"PHOTFLAG": "detected", "ZEROPT_ERR": "zp_error"}

_mdata_gal2_has_err: set[str] = {
    *stitch("LOG", {"MASS", "SFR", "sSFR"}, tight=True),
    *{"COLOR", *stitch({"PHOTO", "SPEC"}, "Z", tight=True)},
}
_mdata_gal2: set[str] = {
    *{"FLAG", "RA", "DEC", "SNSEP", "DDLR", "ELLIPTICITY", "SQRADIUS"},
    *stitch(_mdata_gal2_has_err, "ERR", echo="left"),
    *stitch("OBJID", ["2", "UNIQUE"], tight=[True, False], echo="left"),
    *per_band(stitch("MAG", "ERR", tight=True, echo="left")),
}
_mdata_gal1: set[str] = {
    *_mdata_gal2,
    *{"CONFUSION", *stitch("NMATCH", "2", tight=True, echo="left")},
    *per_band("SB_FLUXCAL"),
}
_mdata_sim: set[str] = {
    *{"RA", "DEC", "AV", "RV", "VPEC", "HOSTLIB_GALID", "NOBS_UNDEFINED"},
    *{"DLMU", "LENSDMU", "MAGSMEAR_COH", "MWEBV", "SEARCHEFF_MASK"},
    *{"MJD_EXPLODE", "PEAKMJD"},
    *stitch("NGEN", "LIBID", echo="right"),
    *stitch({"MODEL", "TYPE"}, {"NAME", "INDEX"}),
    *stitch({"TEMPLATE", "SUBSAMPLE"}, "INDEX"),
    *stitch("REDSHIFT", {"HELIO", "CMB", "HOST", "FLAG"}),
    *per_band({"PEAKMAG", "TEMPLATEMAG", "EXPOSURE"}),
}
_mdata_has_err: set[str] = {"MWEBV", "VPEC", *stitch("REDSHIFT", {"HELIO", "FINAL"})}
ALL_METADATA_COLS: set[str] = {
    *{"IAUC", "FAKE", "RA", "DEC", "PIXSIZE", "PEAKMJD"},
    *stitch(_mdata_has_err, "ERR", echo="left"),
    *stitch(["SN", "SEARCH"], "TYPE", tight=[True, False]),
    *stitch("N", stitch({"X", "Y"}, "PIX", tight=True), tight=True),
    *stitch("MJD", {"TRIGGER", *stitch("DETECT", {"FIRST", "LAST"})}),
    *stitch("HOSTGAL", _mdata_gal1),
    *stitch("HOSTGAL2", _mdata_gal2),
    *stitch("SIM", _mdata_sim),
}
RENAMED_MDATA_COLS = {"SNID": "object_id"}
