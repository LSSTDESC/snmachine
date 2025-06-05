from pathlib import Path

FNAME_TMPL = ("ELASTICC2_TRAIN_02", "NONIaMODEL0-00", "FITS.gz")
FNAME_BASE = "_".join(FNAME_TMPL[:2])


def src_class_dir(root_dir: Path, src_class: str) -> Path:
    return root_dir / f"{FNAME_TMPL[0]}_{src_class}"


def fits_path(src_class_dir: Path, icore: int, key: str) -> Path:
    return src_class_dir / f"{FNAME_BASE}{icore:02d}_{key.upper()}.{FNAME_TMPL[2]}"
