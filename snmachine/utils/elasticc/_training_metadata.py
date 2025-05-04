from __future__ import annotations

from .._utils import stitch

SetDict = dict[str, set[str]]
BAND_LABELS = ["u", "g", "r", "i", "z", "Y"]


def per_band(stems: set[str] | str, **stitch_kwargs) -> set[str]:
    return stitch(stems, set(BAND_LABELS), **stitch_kwargs)


SRC_CLASS_TAXONOMY: dict[str, SetDict] = {
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
ALL_SRC_CLASSES = set()
for supset_dict in SRC_CLASS_TAXONOMY.values():
    for src_class_supset in supset_dict.values():
        ALL_SRC_CLASSES |= src_class_supset


ALL_DATA_COLS = {
    *{"BAND", "CCDNUM", "FIELD", "GAIN", "MJD", "RDNOISE"},
    *stitch({"X", "Y"}, "PIX", tight=True),
    *stitch(["FLUXCAL", "ZEROPT"], "ERR", tight=[True, False], echo="left"),
    *stitch("PSF", {"RATIO", "SIG1", "SIG2"}),
    *stitch("SKY_SIG", "T", echo="left"),
    *stitch("SIM", {"MAGOBS", "FLUXCAL_HOSTERR"}),
    *stitch("PHOT", {"FLAG", "PROB"}, tight=True),
}

_mdata_gal2_has_err = {
    *stitch("LOG", {"MASS", "SFR", "sSFR"}, tight=True),
    *{"COLOR", *stitch({"PHOTO", "SPEC"}, "Z", tight=True)},
}
_mdata_gal2 = {
    *{"FLAG", "RA", "DEC", "SNSEP", "DDLR", "ELLIPTICITY", "SQRADIUS"},
    *stitch(_mdata_gal2_has_err, "ERR", echo="left"),
    *stitch("OBJID", ["2", "UNIQUE"], tight=[True, False], echo="left"),
    *per_band(stitch("MAG", "ERR", tight=True, echo="left")),
}
_mdata_gal1 = {
    *_mdata_gal2,
    *{"CONFUSION", *stitch("NMATCH", "2", tight=True, echo="left")},
    *per_band("SB_FLUXCAL"),
}

_mdata_sim = {
    *{"RA", "DEC", "AV", "RV", "VPEC", "HOSTLIB_GALID", "NOBS_UNDEFINED"},
    *{"DLMU", "LENSDMU", "MAGSMEAR_COH", "MWEBV", "SEARCHEFF_MASK"},
    *{"MJD_EXPLODE", "PEAKMJD"},
    *stitch("NGEN", "LIBID", echo="right"),
    *stitch({"MODEL", "TYPE"}, {"NAME", "INDEX"}),
    *stitch({"TEMPLATE", "SUBSAMPLE"}, "INDEX"),
    *stitch("REDSHIFT", {"HELIO", "CMB", "HOST", "FLAG"}),
    *per_band({"PEAKMAG", "TEMPLATEMAG", "EXPOSURE"}),
}

_mdata_has_err = {"MWEBV", "VPEC", *stitch("REDSHIFT", {"HELIO", "FINAL"})}
ALL_METADATA_COLS = {
    *{"IAUC", "FAKE", "RA", "DEC", "PIXSIZE", "PEAKMJD"},
    *stitch(_mdata_has_err, "ERR", echo="left"),
    *stitch(["SN", "SEARCH"], "TYPE", tight=[True, False]),
    *stitch("N", stitch({"X", "Y"}, "PIX", tight=True), tight=True),
    *stitch("MJD", {"TRIGGER", *stitch("DETECT", {"FIRST", "LAST"})}),
    *stitch("HOSTGAL", _mdata_gal1),
    *stitch("HOSTGAL2", _mdata_gal2),
    *stitch("SIM", _mdata_sim),
}
