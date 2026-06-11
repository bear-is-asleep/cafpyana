"""Chi2 with data histogram from CAF + MC pred/cov from saved systematics."""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from detsys_config import DetsysConfig, cut_chain_for_output
from naming import PAND_CUTS_CONT, PAND_KEY
from sbnd.cafclasses.slice import CAFSlice
from sbnd.numu.numu_constants import (
    BCFM_BINS,
    COSTHETA_BINS,
    FLASHPE_BINS,
    MAX_PMOM,
    MOMENTUM_BINS,
    TRACK_LENGTH_BINS,
    TRACK_PHI_BINS,
    TRACK_THETA_BINS,
    TRACK_XY_BINS,
    TRACK_Z_BINS,
)
from sbnd.stats.stats import calc_chi2

DEFAULT_DATA_DIR = DetsysConfig().data_dir
DEFAULT_VERSION = DetsysConfig().version

VARIABLE_RECO_COLS: dict[str, str] = {
    "costheta": "mu.pfp.trk.costheta",
    "momentum": "mu.pfp.trk.P.p_muon",
    "differential": "bin.differential",
    "bcfm": "slc.barycenterFM.score",
    "flashpe": "slc.barycenterFM.flashPEs",
    "startx": "mu.pfp.trk.start.x",
    "starty": "mu.pfp.trk.start.y",
    "startz": "mu.pfp.trk.start.z",
    "endx": "mu.pfp.trk.end.x",
    "endy": "mu.pfp.trk.end.y",
    "endz": "mu.pfp.trk.end.z",
    "phi": "mu.pfp.trk.phi",
    "theta": "mu.pfp.trk.theta",
    "length": "mu.pfp.trk.len",
    "vtxx": "slc.vertex.x",
    "vtyy": "slc.vertex.y",
    "vtzx": "slc.vertex.z",
}


@dataclass
class CafDataContext:
    slc_data: CAFSlice
    cut_chain: list[str]


def cut_chain_for_syst_dir(syst_path: Path) -> list[str]:
    cut_name = syst_path.name
    if cut_name == "full":
        return list(PAND_CUTS_CONT)
    if cut_name in PAND_CUTS_CONT:
        idx = PAND_CUTS_CONT.index(cut_name)
        return list(PAND_CUTS_CONT[: idx + 1])
    return list(cut_chain_for_output(cut_name, PAND_CUTS_CONT))


def total_cov_dir(syst_path: Path) -> Path:
    total = syst_path / "total"
    if total.is_dir():
        return total
    raise FileNotFoundError(f"Missing total covariance directory under {syst_path}")


def bins_for_variable(var_name: str, syst_path: Path, metadata_dir: str) -> np.ndarray:
    meta_bins = syst_path / metadata_dir / f"{var_name}_bins.csv"
    if meta_bins.is_file():
        bins = np.loadtxt(meta_bins)
    elif var_name == "costheta":
        bins = np.array(COSTHETA_BINS, dtype=float)
    elif var_name == "momentum":
        bins = np.array(MOMENTUM_BINS, dtype=float)
        bins[-1] = MAX_PMOM
    elif var_name == "differential":
        from sbnd.cafclasses.binning import Binning2D

        bins = np.array(Binning2D().differential_edges, dtype=float)
    elif var_name == "bcfm":
        bins = np.array(BCFM_BINS, dtype=float)
    elif var_name == "flashpe":
        bins = np.array(FLASHPE_BINS, dtype=float)
    elif var_name in {"startx", "starty", "endx", "endy", "vtxx", "vtyy"}:
        bins = np.array(TRACK_XY_BINS, dtype=float)
    elif var_name in {"startz", "endz", "vtzx"}:
        bins = np.array(TRACK_Z_BINS, dtype=float)
    elif var_name == "phi":
        bins = np.array(TRACK_PHI_BINS, dtype=float)
    elif var_name == "theta":
        bins = np.array(TRACK_THETA_BINS, dtype=float)
    elif var_name == "length":
        bins = np.array(TRACK_LENGTH_BINS, dtype=float)
    else:
        raise KeyError(f"No bin definition for variable {var_name!r}")
    return bins


def _col_values(slc: CAFSlice, col_path: str) -> pd.Series:
    col = slc.get_key(col_path)[0]
    return slc.data[col]


def _data_counts(series: pd.Series, bins: np.ndarray) -> np.ndarray:
    grouped = series.groupby(pd.cut(series, bins=bins)).count()
    return np.asarray(grouped.values, dtype=float)


def _pred_from_metadata(syst_path: Path, var_name: str, metadata_dir: str) -> np.ndarray:
    meta = syst_path / metadata_dir
    sel = np.loadtxt(meta / f"{var_name}_sel.csv")
    sel_background = np.loadtxt(meta / f"{var_name}_sel_background.csv")
    return sel + sel_background


def load_caf_data(
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    version: str = DEFAULT_VERSION,
    cut_chain: list[str],
    ncpu: int = 1,
) -> CafDataContext:
    """Load on-beam data only, apply cut chain, print cuts used."""
    data_fnames = sorted(glob.glob(f"{data_dir}/data/{version}/*dataonbeam*/*.df"))
    if not data_fnames:
        raise FileNotFoundError(f"No on-beam data files under {data_dir}/data/{version}")

    print(f"Loading data from {len(data_fnames)} file(s) ({data_dir}/data/{version})")
    print(f"Cut chain ({len(cut_chain)} cuts): {cut_chain}")

    slc_data = CAFSlice.load(
        data_fnames, key=PAND_KEY, ncpu=ncpu, cuts=None, show_progress=False
    )
    slc_data.cut_is_cont(cut=False)
    print(f"  precut: {len(slc_data.data)} events")

    for cut in cut_chain:
        n_before = len(slc_data.data)
        slc_data.apply_cut(cut)
        n_after = len(slc_data.data)
        print(f"  cut {cut}: {n_before} -> {n_after}")

    print(f"Selected data events: {len(slc_data.data)}")
    return CafDataContext(slc_data=slc_data, cut_chain=cut_chain)


def compute_chi2_from_caf(
    syst_path: Path,
    var_name: str,
    ctx: CafDataContext,
    *,
    metadata_dir: str,
    cov_dir: Path | None = None,
) -> dict:
    if var_name not in VARIABLE_RECO_COLS:
        raise KeyError(f"Unknown variable {var_name!r}")
    reco_col = VARIABLE_RECO_COLS[var_name]
    bins = bins_for_variable(var_name, syst_path, metadata_dir)

    pred = _pred_from_metadata(syst_path, var_name, metadata_dir)
    data_series = _col_values(ctx.slc_data, reco_col)
    true = _data_counts(data_series, bins)

    cov_root = cov_dir or total_cov_dir(syst_path)
    cov_path = cov_root / f"{var_name}_event_cov.csv"
    if not cov_path.is_file():
        raise FileNotFoundError(f"Missing covariance matrix: {cov_path}")
    cov = np.loadtxt(cov_path)

    chi2, dof, pvalue = calc_chi2(pred, true, cov, filter_min=1.0)
    return {
        "keys": ["total"],
        "chi2": float(chi2),
        "dof": int(dof),
        "pvalue": float(pvalue),
        "pred": pred,
        "true": true,
        "cov": cov,
    }
