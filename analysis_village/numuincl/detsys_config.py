from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from naming import PAND_CUTS_CONT
from sbnd.general.utils import read_hdf

PDS_VARS = ["pmtgain", "pmtqe", "pmtspe"]
SCE_VARS = ["nosce", "twicesce"]
WIREMOD_VARS = ["wiremodxtheta", "wiremodyz"]
CALO_SUFFIXES = [
    "_alpha_embm1",
    "_beta_90m1",
    "_R_embm1",
    "_alpha_embp1",
    "_beta_90p1",
    "_R_embp1",
    "_c_cal_fracp1",
    "_c_cal_fracm1",
]
CALO_VARS = [s[1:] for s in CALO_SUFFIXES]
DET_VARS_ALL = PDS_VARS + SCE_VARS + WIREMOD_VARS
CALO_CUTS = frozenset({"muon", "cont_full", "cont"})


@dataclass(frozen=True)
class DetsysConfig:
    day: str = "checkpoint7_test2"
    data_dir: str = "/exp/sbnd/data/users/brindenc/analyze_sbnd/numu/v10_06_00_validation/pandora"
    version: str = "v8"
    contained: bool = True
    small: bool = False
    tiny: bool = False
    ncpu: int = 1
    chunk_nfiles: int = 8

    @property
    def save_dir(self) -> str:
        return f"{self.data_dir}/data/{self.day}/syst"

    @property
    def universe_dir(self) -> str:
        return f"{self.save_dir}/universes"

    @property
    def plot_dir(self) -> str:
        return f"{self.data_dir}/plots/{self.day}/syst"

    @property
    def cuts(self) -> list[str]:
        return ["precut", *PAND_CUTS_CONT]


def build_config(
    *,
    small: bool = False,
    tiny: bool = False,
    day: str = "checkpoint7",
    chunk_nfiles: int = 8,
    ncpu: int = 1,
    data_dir: str | None = None,
) -> DetsysConfig:
    kwargs: dict[str, Any] = {
        "small": small,
        "tiny": tiny,
        "day": day,
        "chunk_nfiles": chunk_nfiles,
        "ncpu": ncpu,
    }
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    return DetsysConfig(**kwargs)


def _glob_det_var(data_dir: str, version: str, subdir: str, var: str) -> list[str]:
    return sorted(glob.glob(f"{data_dir}/det_var/{subdir}/{version}/*{var}*/*.df"))


def build_det_lists(cfg: DetsysConfig) -> tuple[list[str], list[list[str]]]:
    d, v = cfg.data_dir, cfg.version
    det_vars: list[str] = []
    det_fnames: list[list[str]] = []
    for var in PDS_VARS:
        det_vars.append(var)
        det_fnames.append(_glob_det_var(d, v, "pds", var))
    for var in SCE_VARS:
        det_vars.append(var)
        det_fnames.append(_glob_det_var(d, v, "sce", var))
    for var in WIREMOD_VARS:
        det_vars.append(var)
        det_fnames.append(_glob_det_var(d, v, "wiremod", var))
    return det_vars, det_fnames


def _slice_head(fnames: list[str], div: int) -> list[str]:
    if div <= 1:
        return fnames[:1]
    return fnames[: max(1, len(fnames) // div)]


_AUX_FILE_KEYS = ("OFFBEAM_FNAMES", "MC_LOWE_FNAMES", "DATA_OFFBEAM_FNAMES")


def subsample_file_map(
    file_map: dict[str, Any],
    *,
    slim_div: int | None = None,
    sample_div: int | None = None,
    aux_div: int | None = None,
) -> None:
    """
    Subsample file lists in place. None means leave that group at full file count.

    slim_div: MC_SLIM_FNAMES
    sample_div: nominal + det vars (small/tiny). When aux_div is also set, aux keys
        are not sliced here (avoids double subsampling).
    aux_div: MC offbeam, lowe, data offbeam (shared aux/cosmic background sample)
    """
    if slim_div is not None:
        file_map["MC_SLIM_FNAMES"] = _slice_head(file_map["MC_SLIM_FNAMES"], slim_div)
    if sample_div is not None:
        file_map["MC_NOMINAL_FNAMES"] = _slice_head(file_map["MC_NOMINAL_FNAMES"], sample_div)
        if "DET_FNAMES" in file_map:
            file_map["DET_FNAMES"] = [
                _slice_head(fl, sample_div) for fl in file_map["DET_FNAMES"]
            ]
        if aux_div is None:
            for key in _AUX_FILE_KEYS:
                file_map[key] = _slice_head(file_map[key], sample_div)
    if aux_div is not None:
        for key in _AUX_FILE_KEYS:
            file_map[key] = _slice_head(file_map[key], aux_div)


def build_file_map(cfg: DetsysConfig) -> dict[str, Any]:
    d = cfg.data_dir
    v = cfg.version
    det_vars, det_fnames = build_det_lists(cfg)

    file_map: dict[str, Any] = {
        "MC_SLIM_FNAMES": sorted(glob.glob(f"{d}/mc_syst/{v}/*_slimsyst*/*.df")),
        "MC_NOMINAL_FNAMES": sorted(glob.glob(f"{d}/det_var/nominal/{v}/*_nominal*/*.df")),
        "OFFBEAM_FNAMES": sorted(glob.glob(f"{d}/offbeam/{v}/*mcoffbeam*/*.df")),
        "DATA_OFFBEAM_FNAMES": sorted(glob.glob(f"{d}/offbeam/{v}/*data*/*.df")),
        "DATA_FNAMES": sorted(glob.glob(f"{d}/data/{v}/*dataonbeam*/*.df")),
        "MC_LOWE_FNAMES": sorted(glob.glob(f"{d}/mc_lowe/{v}/*_mclowe_*/*.df")),
        "DET_VARS": det_vars,
        "DET_FNAMES": det_fnames,
        "CALO_VARS": CALO_VARS,
        "CALO_SUFFIXES": CALO_SUFFIXES,
    }
    if cfg.tiny:
        subsample_file_map(file_map, slim_div=1, sample_div=1)
    elif cfg.small:
        subsample_file_map(file_map, slim_div=400, sample_div=5, aux_div=10)
    else:
        subsample_file_map(file_map, aux_div=4)
    return file_map


def chunk_file_list(fnames: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [fnames[i : i + chunk_size] for i in range(0, len(fnames), chunk_size)]


def compute_normalization(
    file_map: dict[str, Any],
    *,
    ncpu: int = 1,
    pot_key: str = "histpotdf*",
    hdr_key: str = "hdr*",
) -> dict[str, Any]:
    norm: dict[str, Any] = {}
    if not file_map["MC_NOMINAL_FNAMES"]:
        raise ValueError("MC_NOMINAL_FNAMES is empty")
    if not file_map["MC_SLIM_FNAMES"]:
        raise ValueError("MC_SLIM_FNAMES is empty")
    if not file_map["OFFBEAM_FNAMES"]:
        raise ValueError("OFFBEAM_FNAMES is empty")

    norm["POT_NOMINAL"] = float(
        read_hdf(file_map["MC_NOMINAL_FNAMES"], key=pot_key, ncpu=ncpu, show_progress=False).TotalPOT.sum()
    )
    norm["POT_MC_SLIM_FULL"] = float(
        read_hdf(file_map["MC_SLIM_FNAMES"], key=pot_key, ncpu=ncpu, show_progress=False).TotalPOT.sum()
    )
    norm["POT_MC_LOWE_FULL"] = float(
        read_hdf(file_map["MC_LOWE_FNAMES"], key=pot_key, ncpu=ncpu, show_progress=False).TotalPOT.sum()
    )

    # Set min for data only since it has a small number of files
    n_data = len(file_map["DATA_FNAMES"])
    hdr_data = read_hdf(
        file_map["DATA_FNAMES"],
        key=hdr_key,
        ncpu=min(max(n_data, 1), ncpu),
        show_progress=False,
    )
    norm["POT_DATA"] = float(hdr_data.pot.values.sum())
    norm["LIVETIME_DATA"] = 9.51e5

    offbeam_hdr = read_hdf(file_map["DATA_OFFBEAM_FNAMES"], key=hdr_key, ncpu=ncpu, show_progress=False)
    norm["LIVETIME_OFFBEAM_DATA_FULL"] = float(offbeam_hdr.noffbeambnb.values.sum())

    genevt_key = "histgenevtdf*"
    offbeam_mc = read_hdf(file_map["OFFBEAM_FNAMES"], key=genevt_key, ncpu=ncpu, show_progress=False)
    norm["LIVETIME_OFFBEAM_FULL"] = float(offbeam_mc.TotalGenEvents.sum())

    norm["POT_DET"] = {}
    for var, flist in zip(file_map["DET_VARS"], file_map["DET_FNAMES"]):
        if not flist:
            raise ValueError(f"DET file list empty for variation {var}")
        norm["POT_DET"][var] = float(
            read_hdf(flist, key=pot_key, ncpu=ncpu, show_progress=False).TotalPOT.sum()
        )
    return norm


def normalization_json_path(save_dir: str) -> Path:
    return Path(save_dir) / "metadata" / "normalization.json"


def write_normalization_json(save_dir: str, normalization: dict[str, Any]) -> str:
    metadata_dir = normalization_json_path(save_dir).parent
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output = normalization_json_path(save_dir)
    with output.open("w", encoding="utf-8") as f:
        json.dump(normalization, f, indent=2, sort_keys=True, default=str)
    return str(output)


def load_normalization_json(save_dir: str) -> dict[str, Any]:
    path = normalization_json_path(save_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run the build once to create normalization.json"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dirs(cfg: DetsysConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.universe_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)
