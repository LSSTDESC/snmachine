from __future__ import annotations

from .._utils import stitch

SetDict = dict[str, set[str]]
BAND_LABELS = ["u", "g", "r", "i", "z", "Y"]
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
