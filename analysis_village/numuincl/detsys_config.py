from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from naming import PAND_CUTS_CONT
from sbnd.general.utils import read_hdf

DETSYS_CUTS_ALL = ("precut", *PAND_CUTS_CONT)
DETSYS_CUT_NAMES = frozenset(DETSYS_CUTS_ALL)
_MUON_IDX = PAND_CUTS_CONT.index("muon")
CUTS_BEFORE_MUON = tuple(PAND_CUTS_CONT[:_MUON_IDX])
CUTS_FROM_MUON = tuple(PAND_CUTS_CONT[_MUON_IDX:])

# Slim HDF: bundled multisim + bundled multisigma (MvA ZExp per-knob).
SLIM_HDF_SCHEMA = "ar23p_slim_bundle_v1"
SIGNAL_CATEGORIES_XSEC = [0]


def save_stages_for(cut: str | None) -> tuple[str, ...]:
    """Output directories to write. None = full build (all stages)."""
    if cut is None:
        return DETSYS_CUTS_ALL
    if cut not in DETSYS_CUT_NAMES:
        raise ValueError(
            f"cut must be one of {sorted(DETSYS_CUT_NAMES)}, got {cut!r}"
        )
    return (cut,)


def cut_chain_for_detsys(cut: str) -> list[str]:
    """Apply-cut chain for save stage cut: all PAND_CUTS_CONT entries through cut."""
    if cut == "precut":
        return []
    if cut not in PAND_CUTS_CONT:
        raise ValueError(
            f"cut {cut!r} must be 'precut' or one of {list(PAND_CUTS_CONT)}"
        )
    return list(PAND_CUTS_CONT[: PAND_CUTS_CONT.index(cut) + 1])


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

BUILD_MODES = frozenset(
    {
        "default",
        "small",
        "tiny",
        "full_slim",
        "full_det",
        "full_cosmic",
        "full_slim_test",
        "full_det_test",
        "full_cosmic_test",
    }
)

# Full production divisors (tuned for ~64 GB RAM, three-pass build).
# div == 1: full list. N>=2: head slice len // N.
_FULL_SLIM_DIV = 1
_FULL_NOMINAL_DIV = 1
_FULL_DET_DIV = 1
# lowe + data-offbeam for full-det and full-slim only
_FULL_PARTIAL_AUX_DIV = 10
# full-cosmic: chunked loads are concat'd in memory per cut (worst at precut)
# div=7 -> ~33 GB inflated peak (div=6 was ~37 GB; extra headroom on 64 GB nodes)
_FULL_COSMIC_NOMINAL_DIV = 7
_FULL_COSMIC_OFFBEAM_MC_DIV = 7
_FULL_COSMIC_LOWE_DIV = 7
_FULL_COSMIC_DATA_OFFBEAM_DIV = 7
# --full-*-test modes: same layout as full-* but 1/100 file count on loaded groups
_FULL_TEST_DIV = 100

_SMALL_SLIM_DIV = 100
_SMALL_SAMPLE_DIV = 5
_SMALL_PARTIAL_AUX_DIV = 10
_DEFAULT_SLIM_DIV = 50
_DEFAULT_SAMPLE_DIV = 4
_DEFAULT_AUX_DIV = 12

_PARTIAL_AUX_KEYS = ("MC_LOWE_FNAMES", "DATA_OFFBEAM_FNAMES")
_AUX_FILE_KEYS = ("OFFBEAM_FNAMES", "MC_LOWE_FNAMES", "DATA_OFFBEAM_FNAMES")
_EMPTY_DIV = -1
_SINGLE_FILE_DIV = 0


@dataclass(frozen=True)
class DetsysConfig:
    day: str = "checkpoint10_test"
    data_dir: str = "/exp/sbnd/data/users/brindenc/analyze_sbnd/numu/v10_06_00_validation/pandora"
    # Written artifacts (det CSVs, pot_scaling, universes, plots). None -> data_dir.
    out_dir: str | None = None
    # Slim GENIE HDF with full AR23+ knobs + per-knob multisigma
    version: str = "v9"
    contained: bool = True
    build_mode: str = "default"
    ncpu: int = 1
    chunk_nfiles: int = 8
    show_progress: bool = True
    cut: str | None = None

    def __post_init__(self) -> None:
        if self.out_dir is None:
            object.__setattr__(self, "out_dir", self.data_dir)

    @property
    def small(self) -> bool:
        return self.build_mode == "small"

    @property
    def tiny(self) -> bool:
        return self.build_mode == "tiny"

    @property
    def full_slim(self) -> bool:
        return self.build_mode in ("full_slim", "full_slim_test")

    @property
    def full_det(self) -> bool:
        return self.build_mode in ("full_det", "full_det_test")

    @property
    def full_cosmic(self) -> bool:
        return self.build_mode in ("full_cosmic", "full_cosmic_test")

    @property
    def full_test(self) -> bool:
        return self.build_mode.endswith("_test")

    @property
    def save_build_mode(self) -> str:
        if self.build_mode.endswith("_test"):
            return self.build_mode[: -len("_test")]
        return self.build_mode

    @property
    def runs_slim(self) -> bool:
        return self.build_mode in (
            "default",
            "small",
            "tiny",
            "full_slim",
            "full_slim_test",
        )

    @property
    def runs_det(self) -> bool:
        return self.build_mode in (
            "default",
            "small",
            "tiny",
            "full_det",
            "full_det_test",
        )

    @property
    def runs_cosmic(self) -> bool:
        return self.build_mode in (
            "default",
            "small",
            "tiny",
            "full_cosmic",
            "full_cosmic_test",
        )

    @property
    def save_dir(self) -> str:
        return f"{self.out_dir}/data/{self.day}/syst"

    @property
    def universe_dir(self) -> str:
        return f"{self.save_dir}/universes"

    @property
    def plot_dir(self) -> str:
        return f"{self.out_dir}/plots/{self.day}/syst"

    @property
    def cuts(self) -> list[str]:
        return list(save_stages_for(self.cut))


def build_config(
    *,
    build_mode: str = "default",
    day: str = "checkpoint10_test",
    chunk_nfiles: int = 8,
    ncpu: int = 1,
    show_progress: bool = True,
    cut: str | None = None,
    data_dir: str | None = None,
    out_dir: str | None = None,
    version: str | None = None,
    small: bool | None = None,
    tiny: bool | None = None,
) -> DetsysConfig:
    if build_mode not in BUILD_MODES:
        raise ValueError(f"build_mode must be one of {sorted(BUILD_MODES)}, got {build_mode!r}")
    if tiny:
        build_mode = "tiny"
    elif small:
        build_mode = "small"
    kwargs: dict[str, Any] = {
        "build_mode": build_mode,
        "day": day,
        "chunk_nfiles": chunk_nfiles,
        "ncpu": ncpu,
        "show_progress": show_progress,
    }
    if cut is not None and cut not in DETSYS_CUT_NAMES:
        raise ValueError(
            f"cut must be one of {sorted(DETSYS_CUT_NAMES)}, got {cut!r}"
        )
    kwargs["cut"] = cut
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    if out_dir is not None:
        kwargs["out_dir"] = out_dir
    if version is not None:
        kwargs["version"] = version
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


def _apply_group_div(fnames: list[str], div: int | None) -> list[str]:
    """
    Apply a divisor to a file list.

    div is None: unchanged (full list)
    div == _EMPTY_DIV (-1): empty list
    div == _SINGLE_FILE_DIV (0): one file
    div == 1: full list (full-* primary groups use this global)
    div >= 2: head slice len // div
    """
    if div is None:
        return fnames
    if div == _EMPTY_DIV:
        return []
    if not fnames:
        return []
    if div == _SINGLE_FILE_DIV:
        return fnames[:1]
    if div == 1:
        return fnames
    return fnames[: max(1, len(fnames) // div)]


def _apply_partial_aux_div(file_map: dict[str, Any], div: int) -> None:
    for key in _PARTIAL_AUX_KEYS:
        file_map[key] = _apply_group_div(file_map[key], div)


def subsample_file_map(
    file_map: dict[str, Any],
    *,
    slim_div: int | None = None,
    sample_div: int | None = None,
    aux_div: int | None = None,
    partial_aux_div: int | None = None,
) -> None:
    """
    Subsample file lists in place. None means leave that group at full file count.

    slim_div: MC_SLIM_FNAMES
    sample_div: nominal + det vars. When aux_div is also set, aux keys
        are not sliced here (avoids double subsampling).
    aux_div: all three aux keys (offbeam MC, lowe, data offbeam)
    partial_aux_div: MC_LOWE_FNAMES and DATA_OFFBEAM_FNAMES only
    """
    if slim_div is not None:
        file_map["MC_SLIM_FNAMES"] = _apply_group_div(file_map["MC_SLIM_FNAMES"], slim_div)
    if sample_div is not None:
        file_map["MC_NOMINAL_FNAMES"] = _apply_group_div(
            file_map["MC_NOMINAL_FNAMES"], sample_div
        )
        if "DET_FNAMES" in file_map:
            file_map["DET_FNAMES"] = [
                _apply_group_div(fl, sample_div) for fl in file_map["DET_FNAMES"]
            ]
        if aux_div is None and partial_aux_div is None:
            for key in _AUX_FILE_KEYS:
                file_map[key] = _apply_group_div(file_map[key], sample_div)
    if aux_div is not None:
        for key in _AUX_FILE_KEYS:
            file_map[key] = _apply_group_div(file_map[key], aux_div)
    if partial_aux_div is not None:
        _apply_partial_aux_div(file_map, partial_aux_div)


def _apply_full_slim_splits(
    file_map: dict[str, Any], *, slim_div: int, partial_aux_div: int
) -> None:
    file_map["MC_SLIM_FNAMES"] = _apply_group_div(file_map["MC_SLIM_FNAMES"], slim_div)
    file_map["MC_NOMINAL_FNAMES"] = []
    file_map["OFFBEAM_FNAMES"] = []
    file_map["DET_FNAMES"] = [[] for _ in file_map["DET_FNAMES"]]
    _apply_partial_aux_div(file_map, partial_aux_div)


def _apply_full_det_splits(
    file_map: dict[str, Any], *, nominal_div: int, det_div: int, partial_aux_div: int
) -> None:
    file_map["MC_SLIM_FNAMES"] = []
    file_map["MC_NOMINAL_FNAMES"] = _apply_group_div(
        file_map["MC_NOMINAL_FNAMES"], nominal_div
    )
    file_map["DET_FNAMES"] = [
        _apply_group_div(fl, det_div) for fl in file_map["DET_FNAMES"]
    ]
    file_map["OFFBEAM_FNAMES"] = []
    _apply_partial_aux_div(file_map, partial_aux_div)


def _apply_full_cosmic_splits(
    file_map: dict[str, Any],
    *,
    nominal_div: int,
    offbeam_mc_div: int,
    lowe_div: int,
    data_offbeam_div: int,
) -> None:
    file_map["MC_SLIM_FNAMES"] = []
    file_map["MC_NOMINAL_FNAMES"] = _apply_group_div(
        file_map["MC_NOMINAL_FNAMES"], nominal_div
    )
    file_map["DET_FNAMES"] = [[] for _ in file_map["DET_FNAMES"]]
    file_map["OFFBEAM_FNAMES"] = _apply_group_div(
        file_map["OFFBEAM_FNAMES"], offbeam_mc_div
    )
    file_map["MC_LOWE_FNAMES"] = _apply_group_div(file_map["MC_LOWE_FNAMES"], lowe_div)
    file_map["DATA_OFFBEAM_FNAMES"] = _apply_group_div(
        file_map["DATA_OFFBEAM_FNAMES"], data_offbeam_div
    )


def apply_build_mode_splits(file_map: dict[str, Any], mode: str) -> None:
    """Apply per-mode file list divisors (see scripts/README.md table)."""
    if mode == "full_slim":
        _apply_full_slim_splits(
            file_map, slim_div=_FULL_SLIM_DIV, partial_aux_div=_FULL_PARTIAL_AUX_DIV
        )
        return

    if mode == "full_slim_test":
        _apply_full_slim_splits(
            file_map, slim_div=_FULL_TEST_DIV, partial_aux_div=_FULL_TEST_DIV
        )
        return

    if mode == "full_det":
        _apply_full_det_splits(
            file_map,
            nominal_div=_FULL_NOMINAL_DIV,
            det_div=_FULL_DET_DIV,
            partial_aux_div=_FULL_PARTIAL_AUX_DIV,
        )
        return

    if mode == "full_det_test":
        _apply_full_det_splits(
            file_map,
            nominal_div=_FULL_TEST_DIV,
            det_div=_FULL_TEST_DIV,
            partial_aux_div=_FULL_TEST_DIV,
        )
        return

    if mode == "full_cosmic":
        _apply_full_cosmic_splits(
            file_map,
            nominal_div=_FULL_COSMIC_NOMINAL_DIV,
            offbeam_mc_div=_FULL_COSMIC_OFFBEAM_MC_DIV,
            lowe_div=_FULL_COSMIC_LOWE_DIV,
            data_offbeam_div=_FULL_COSMIC_DATA_OFFBEAM_DIV,
        )
        return

    if mode == "full_cosmic_test":
        _apply_full_cosmic_splits(
            file_map,
            nominal_div=_FULL_TEST_DIV,
            offbeam_mc_div=_FULL_TEST_DIV,
            lowe_div=_FULL_TEST_DIV,
            data_offbeam_div=_FULL_TEST_DIV,
        )
        return

    if mode == "tiny":
        subsample_file_map(
            file_map,
            slim_div=_SINGLE_FILE_DIV,
            sample_div=_SINGLE_FILE_DIV,
        )
        return

    if mode == "small":
        subsample_file_map(
            file_map,
            slim_div=_SMALL_SLIM_DIV,
            sample_div=_SMALL_SAMPLE_DIV,
            partial_aux_div=_SMALL_PARTIAL_AUX_DIV,
        )
        return

    # default
    subsample_file_map(
        file_map,
        slim_div=_DEFAULT_SLIM_DIV,
        sample_div=_DEFAULT_SAMPLE_DIV,
        aux_div=_DEFAULT_AUX_DIV,
    )


def _base_file_map(cfg: DetsysConfig) -> dict[str, Any]:
    d = cfg.data_dir
    v = cfg.version
    det_vars, det_fnames = build_det_lists(cfg)
    return {
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


def build_file_map(cfg: DetsysConfig) -> dict[str, Any]:
    file_map = _base_file_map(cfg)
    apply_build_mode_splits(file_map, cfg.build_mode)
    return file_map


def _file_group_pct(used: int, full: int) -> str:
    if full <= 0:
        return "n/a" if used == 0 else "new"
    return f"{100.0 * used / full:5.1f}%"


_GB = 1024**3


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / _GB:8.2f} GB"


def _collect_file_map_paths(file_map: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for key in (
        "MC_SLIM_FNAMES",
        "MC_NOMINAL_FNAMES",
        "OFFBEAM_FNAMES",
        "MC_LOWE_FNAMES",
        "DATA_OFFBEAM_FNAMES",
        "DATA_FNAMES",
    ):
        paths.update(file_map[key])
    for flist in file_map["DET_FNAMES"]:
        paths.update(flist)
    return paths


def _build_file_size_cache(paths: set[str]) -> dict[str, int]:
    cache: dict[str, int] = {}
    for path in paths:
        try:
            cache[path] = os.path.getsize(path)
        except OSError:
            cache[path] = 0
    return cache


def _file_list_bytes(fnames: list[str], size_cache: dict[str, int]) -> int:
    return sum(size_cache.get(path, 0) for path in fnames)


_NODE_RAM_GB = 64
_RAM_PANDAS_INFLATE = 4.0
_PEAK_RAM_WARN_GB = 58


def _max_chunk_bytes(
    fnames: list[str], chunk_nfiles: int, size_cache: dict[str, int]
) -> tuple[int, int]:
    """Largest on-disk chunk and number of chunks (file-size proxy for load RAM)."""
    if not fnames:
        return 0, 0
    chunks = chunk_file_list(fnames, chunk_nfiles)
    return max(_file_list_bytes(chunk, size_cache) for chunk in chunks), len(chunks)


def _file_map_load_roles(cfg: DetsysConfig) -> dict[str, str]:
    """How each group is loaded: chunked, concat (chunk then hold full), resident, or skip."""
    cosmic_only = cfg.build_mode in ("full_cosmic", "full_cosmic_test")
    return {
        "MC_SLIM_FNAMES": "chunked" if cfg.runs_slim else "skip",
        "MC_NOMINAL_FNAMES": (
            "concat"
            if cosmic_only
            else "chunked"
            if cfg.runs_det
            else "skip"
        ),
        "OFFBEAM_FNAMES": "concat" if cosmic_only else "chunked" if cfg.runs_cosmic else "skip",
        "MC_LOWE_FNAMES": "resident",
        "DATA_OFFBEAM_FNAMES": "resident",
        "DATA_FNAMES": "resident",
    }


def _format_peak_chunk_ram(
    role: str,
    fnames: list[str],
    chunk_nfiles: int,
    size_cache: dict[str, int],
) -> str:
    if role == "concat" and fnames:
        total_bytes = _file_list_bytes(fnames, size_cache)
        return f"{total_bytes / _GB:8.2f} GB  (concat)"
    if role != "chunked" or not fnames:
        return f"{'':>12}"
    chunk_bytes, n_chunks = _max_chunk_bytes(fnames, chunk_nfiles, size_cache)
    return f"{chunk_bytes / _GB:8.2f} GB  ({n_chunks} ch)"


def format_file_map_summary(
    cfg: DetsysConfig,
    file_map: dict[str, Any],
    *,
    chunk_nfiles: int | None = None,
) -> str:
    """Used/full counts, on-disk read size, and peak chunk RAM for chunked groups."""
    chunk_nfiles = cfg.chunk_nfiles if chunk_nfiles is None else chunk_nfiles
    full_map = _base_file_map(cfg)
    size_cache = _build_file_size_cache(_collect_file_map_paths(full_map))
    load_roles = _file_map_load_roles(cfg)
    lines = [
        f"Build mode={cfg.build_mode}  chunk_nfiles={chunk_nfiles}",
        f"  {'group':14} {'load':>10} {'files':>13}  {'disk read':>27}  {'peak chunk':>18}",
    ]
    list_groups = (
        ("MC_SLIM_FNAMES", "slim MC"),
        ("MC_NOMINAL_FNAMES", "nominal MC"),
        ("OFFBEAM_FNAMES", "offbeam MC"),
        ("MC_LOWE_FNAMES", "lowe MC"),
        ("DATA_OFFBEAM_FNAMES", "data offbeam"),
        ("DATA_FNAMES", "data onbeam"),
    )
    total_used_files = 0
    total_full_files = 0
    total_used_bytes = 0
    total_full_bytes = 0
    resident_bytes = 0
    max_chunk_bytes = 0
    concat_bytes = 0
    cosmic_only = cfg.build_mode in ("full_cosmic", "full_cosmic_test")
    for key, label in list_groups:
        used = len(file_map[key])
        full = len(full_map[key])
        pct = _file_group_pct(used, full)
        used_bytes = _file_list_bytes(file_map[key], size_cache)
        full_bytes = _file_list_bytes(full_map[key], size_cache)
        role = load_roles[key] if used else "skip"
        total_used_files += used
        total_full_files += full
        total_used_bytes += used_bytes
        total_full_bytes += full_bytes
        if role == "resident" and used:
            resident_bytes += used_bytes
        elif role == "concat" and used:
            concat_bytes += used_bytes
        elif role == "chunked" and used:
            chunk_bytes, _ = _max_chunk_bytes(file_map[key], chunk_nfiles, size_cache)
            max_chunk_bytes = max(max_chunk_bytes, chunk_bytes)
        peak_col = _format_peak_chunk_ram(role, file_map[key], chunk_nfiles, size_cache)
        lines.append(
            f"  {label:14} {role:>10} {used:5d} / {full:5d}  ({pct})"
            f"  {_format_gb(used_bytes)} / {_format_gb(full_bytes)}  {peak_col}"
        )

    det_vars = file_map["DET_VARS"]
    used_det = file_map["DET_FNAMES"]
    full_det = full_map["DET_FNAMES"]
    det_used_total = sum(len(fl) for fl in used_det)
    det_full_total = sum(len(fl) for fl in full_det)
    det_pct = _file_group_pct(det_used_total, det_full_total)
    det_used_bytes = sum(_file_list_bytes(fl, size_cache) for fl in used_det)
    det_full_bytes = sum(_file_list_bytes(fl, size_cache) for fl in full_det)
    total_used_files += det_used_total
    total_full_files += det_full_total
    total_used_bytes += det_used_bytes
    total_full_bytes += det_full_bytes
    det_role = "chunked" if cfg.runs_det else "skip"
    det_peak_col = f"{'':>12}"
    if det_role == "chunked" and det_used_total:
        det_chunk_peak = 0
        det_n_chunks = 0
        for flist in used_det:
            if not flist:
                continue
            chunk_bytes, n_chunks = _max_chunk_bytes(flist, chunk_nfiles, size_cache)
            det_chunk_peak = max(det_chunk_peak, chunk_bytes)
            det_n_chunks = max(det_n_chunks, n_chunks)
        max_chunk_bytes = max(max_chunk_bytes, det_chunk_peak)
        det_peak_col = f"{det_chunk_peak / _GB:8.2f} GB  ({det_n_chunks} ch)"
    lines.append(
        f"  {'det (all vars)':14} {det_role:>10} {det_used_total:5d} / {det_full_total:5d}  ({det_pct})"
        f"  {_format_gb(det_used_bytes)} / {_format_gb(det_full_bytes)}  {det_peak_col}"
    )
    for var, used_list, full_list in zip(det_vars, used_det, full_det):
        used = len(used_list)
        full = len(full_list)
        pct = _file_group_pct(used, full)
        used_bytes = _file_list_bytes(used_list, size_cache)
        full_bytes = _file_list_bytes(full_list, size_cache)
        var_role = det_role if used else "skip"
        peak_col = _format_peak_chunk_ram(var_role, used_list, chunk_nfiles, size_cache)
        lines.append(
            f"    {var:12} {var_role:>10} {used:5d} / {full:5d}  ({pct})"
            f"  {_format_gb(used_bytes)} / {_format_gb(full_bytes)}  {peak_col}"
        )

    total_file_pct = _file_group_pct(total_used_files, total_full_files)
    total_size_pct = _file_group_pct(total_used_bytes, total_full_bytes)
    if cosmic_only and concat_bytes:
        peak_bytes = resident_bytes + concat_bytes
        peak_terms = (
            f"resident {_format_gb(resident_bytes)} + concat {_format_gb(concat_bytes)}"
        )
    else:
        peak_bytes = resident_bytes + max_chunk_bytes
        peak_terms = (
            f"resident {_format_gb(resident_bytes)} + max chunk {_format_gb(max_chunk_bytes)}"
        )
    peak_inflated_bytes = int(peak_bytes * _RAM_PANDAS_INFLATE)
    lines.append(
        f"  {'TOTAL disk':14} {'':>10} {total_used_files:5d} / {total_full_files:5d}  ({total_file_pct})"
        f"  {_format_gb(total_used_bytes)} / {_format_gb(total_full_bytes)} ({total_size_pct})"
    )
    lines.append(
        f"  {'PEAK RAM est.':14} {'':>10} {peak_terms}"
        f" = {_format_gb(peak_bytes)} disk proxy"
    )
    lines.append(
        f"  {'PEAK x infl.':14} {'':>10} x{_RAM_PANDAS_INFLATE:g} pandas load"
        f" ~ {_format_gb(peak_inflated_bytes)} on {_NODE_RAM_GB} GB node"
    )
    if peak_inflated_bytes / _GB > _PEAK_RAM_WARN_GB:
        lines.append(
            f"  WARNING: inflated peak ~{_format_gb(peak_inflated_bytes)} exceeds"
            f" {_PEAK_RAM_WARN_GB} GB budget — raise cosmic divisors in detsys_config.py"
            " or lower chunk_nfiles."
        )
    return "\n".join(lines)


def chunk_file_list(fnames: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not fnames:
        return []
    return [fnames[i : i + chunk_size] for i in range(0, len(fnames), chunk_size)]


def _slim_pot_fnames(file_map: dict[str, Any]) -> list[str]:
    return list(file_map.get("MC_SLIM_FNAMES") or [])


def _normalization_step_count(mode: str, file_map: dict[str, Any]) -> int:
    n = 3  # lowe MC, data onbeam, data offbeam
    if mode in ("default", "small", "tiny", "full_slim", "full_slim_test"):
        n += 1
    if mode in (
        "default",
        "small",
        "tiny",
        "full_det",
        "full_det_test",
        "full_cosmic",
        "full_cosmic_test",
    ):
        n += 1
    if mode in ("default", "small", "tiny", "full_cosmic", "full_cosmic_test"):
        n += 1
    if mode in ("default", "small", "tiny", "full_det", "full_det_test"):
        n += sum(1 for flist in file_map["DET_FNAMES"] if flist)
    return n


def compute_normalization(
    file_map: dict[str, Any],
    *,
    mode: str = "default",
    ncpu: int = 1,
    pot_key: str = "histpotdf*",
    hdr_key: str = "hdr*",
    show_progress: bool = True,
) -> dict[str, Any]:
    if mode not in BUILD_MODES:
        raise ValueError(f"mode must be one of {sorted(BUILD_MODES)}, got {mode!r}")

    norm: dict[str, Any] = {}
    pbar = tqdm(
        total=_normalization_step_count(mode, file_map),
        desc="normalization",
        unit="scan",
        disable=not show_progress,
    )

    def _step(label: str, fn):
        pbar.set_postfix_str(label, refresh=False)
        result = fn()
        pbar.update(1)
        return result

    try:
        if mode in ("default", "small", "tiny", "full_slim", "full_slim_test"):
            slim_fnames = _slim_pot_fnames(file_map)
            if not slim_fnames:
                raise ValueError("MC slim file list is empty (required for slim POT)")
            norm["POT_MC_SLIM_FULL"] = float(
                _step(
                    "slim POT",
                    lambda: read_hdf(
                        slim_fnames, key=pot_key, ncpu=ncpu, show_progress=False
                    ).TotalPOT.sum(),
                )
            )

        if mode in (
            "default",
            "small",
            "tiny",
            "full_det",
            "full_det_test",
            "full_cosmic",
            "full_cosmic_test",
        ):
            if not file_map["MC_NOMINAL_FNAMES"]:
                raise ValueError("MC_NOMINAL_FNAMES is empty")
            norm["POT_NOMINAL"] = float(
                _step(
                    "nominal POT",
                    lambda: read_hdf(
                        file_map["MC_NOMINAL_FNAMES"],
                        key=pot_key,
                        ncpu=ncpu,
                        show_progress=False,
                    ).TotalPOT.sum(),
                )
            )

        if mode in ("default", "small", "tiny", "full_cosmic", "full_cosmic_test"):
            if not file_map["OFFBEAM_FNAMES"]:
                raise ValueError("OFFBEAM_FNAMES is empty")
            genevt_key = "histgenevtdf*"

            def _offbeam_mc_livetime():
                offbeam_mc = read_hdf(
                    file_map["OFFBEAM_FNAMES"],
                    key=genevt_key,
                    ncpu=ncpu,
                    show_progress=False,
                )
                return offbeam_mc.TotalGenEvents.sum()

            norm["LIVETIME_OFFBEAM_FULL"] = float(
                _step("offbeam MC livetime", _offbeam_mc_livetime)
            )

        if mode in ("default", "small", "tiny", "full_det", "full_det_test"):
            norm["POT_DET"] = {}
            for var, flist in zip(file_map["DET_VARS"], file_map["DET_FNAMES"]):
                if not flist:
                    raise ValueError(f"DET file list empty for variation {var}")
                norm["POT_DET"][var] = float(
                    _step(
                        f"{var} POT",
                        lambda fl=flist: read_hdf(
                            fl, key=pot_key, ncpu=ncpu, show_progress=False
                        ).TotalPOT.sum(),
                    )
                )

        if not file_map["MC_LOWE_FNAMES"]:
            raise ValueError("MC_LOWE_FNAMES is empty")
        if not file_map["DATA_FNAMES"]:
            raise ValueError("DATA_FNAMES is empty")
        if not file_map["DATA_OFFBEAM_FNAMES"]:
            raise ValueError("DATA_OFFBEAM_FNAMES is empty")

        norm["POT_MC_LOWE_FULL"] = float(
            _step(
                "lowe POT",
                lambda: read_hdf(
                    file_map["MC_LOWE_FNAMES"],
                    key=pot_key,
                    ncpu=ncpu,
                    show_progress=False,
                ).TotalPOT.sum(),
            )
        )

        n_data = len(file_map["DATA_FNAMES"])
        norm["POT_DATA"] = float(
            _step(
                "data POT",
                lambda: read_hdf(
                    file_map["DATA_FNAMES"],
                    key=hdr_key,
                    ncpu=min(max(n_data, 1), ncpu),
                    show_progress=False,
                ).pot.values.sum(),
            )
        )
        norm["LIVETIME_DATA"] = 9.51e5

        norm["LIVETIME_OFFBEAM_DATA_FULL"] = float(
            _step(
                "data offbeam livetime",
                lambda: read_hdf(
                    file_map["DATA_OFFBEAM_FNAMES"],
                    key=hdr_key,
                    ncpu=ncpu,
                    show_progress=False,
                ).noffbeambnb.values.sum(),
            )
        )
    finally:
        pbar.close()

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
