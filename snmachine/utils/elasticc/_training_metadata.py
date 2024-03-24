from __future__ import annotations
from typing import Dict, Set

SetDict = Dict[str, Set[str]]

src_class_taxonomy: Dict[str, SetDict] = {
    "Non-Recurring": {
        "SN-like": {
            "SNIa-91bg",
            "SNIa-SALT3",
            "SNIax",
            "SNIb+HostXT_V19",
            "SNIb-Templates",
            "SNIcBL+HostXT_V19",
            "SNIc+HostXT_V19",
            "SNIc-Templates",
            "SNIIb+HostXT_V19",
            "SNII+HostXT_V19",
            "SNIIn+HostXT_V19",
            "SNII-NMF",
            "SNIIn-MOSFIT",
            "SNII-Templates",
        },
        "Fast": {
            "KN_B19",
            "KN_K17",
            "Mdwarf-flare",
            "dwarf-nova",
            "uLens-Binary",
            "uLens-Single-GenLens",
            "uLens-Single_PyLIMA",
        },
        "Long": {"SLSN-I+host", "SLSN-I_no_host", "TDE", "CART", "ILOT", "PISN"},
    },
    "Recurring": {
        "Periodic": {"Cepheid", "d-Sct", "EB", "RRL"},
        "Non-Periodic": {"CLAGN"},
    },
}
all_src_classes = set()
for supset_dict in src_class_taxonomy.values():
    for src_class_supset in supset_dict.values():
        all_src_classes |= src_class_supset

all_data_cols = {
    "MJD",
    "BAND",
    "PHOTFLAG",
    "FLUXCAL",
    "FLUXCALERR",
    "CCDNUM",
    "ZEROPT_ERR",
    "SIM_MAGOBS",
    "YPIX",
    "FIELD",
    "SIM_FLUXCAL_HOSTERR",
    "ZEROPT",
    "RDNOISE",
    "SKY_SIG",
    "PHOTPROB",
    "SKY_SIG_T",
    "GAIN",
    "PSF_SIG2",
    "PSF_SIG1",
    "XPIX",
    "PSF_RATIO",
}
