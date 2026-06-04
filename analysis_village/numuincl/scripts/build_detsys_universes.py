#!/usr/bin/env python3
"""Chunked build of slim RW, detector, CALO, and cosmic systematics universes."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
    module="numpy.lib.histograms",
)

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
CAFPYANA_WD = REPO_ROOT.parents[1]
for p in (str(REPO_ROOT), str(CAFPYANA_WD)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detsys_config import (
    CALO_CUTS,
    CALO_SUFFIXES,
    CALO_VARS,
    build_config,
    build_file_map,
    chunk_file_list,
    compute_normalization,
    load_normalization_json,
    normalization_json_path,
    write_normalization_json,
)
from detsys_det_match import (
    filter_slice_to_events,
    load_variation_events,
    require_artifacts,
)
from detsys_pot_debug import ChunkStageContext, DetPotDebug  # DEBUG-POT
from sbnd.cafclasses.binning import Binning2D
from sbnd.cafclasses.slice import CAFSlice
from sbnd.detector.definitions import F_SCALE, NUMBER_TARGETS_FV
from sbnd.flux.constants import NUMU_INTEGRATED_FLUX
from sbnd.numu.numu_constants import (
    BCFM_BINS,
    COSTHETA_BINS,
    DIFF_MOMENTUM_BINS_2D,
    FLASHPE_BINS,
    MOMENTUM_BINS,
    TRACK_LENGTH_BINS,
    TRACK_PHI_BINS,
    TRACK_THETA_BINS,
    TRACK_XY_BINS,
    TRACK_Z_BINS,
)
from sbnd.stats.systematics import (
    Systematics,
    compute_efficiency,
    compute_sigma_tilde,
    convert_smearing_to_response,
)


SLIM_KEYS = ["xsec", "flux", "g4"]
FLUX_FILE = "/exp/sbnd/data/users/munjung/xsec/flux_closure/Gen1FV_flux.root"
PAND_KEY = "evt_pand*"
OFFBEAM_COMBINE_OFFSET = int(1e7)
LOWE_COMBINE_OFFSET = int(2e7)
COSMIC_NT_LO = int(1e7)


def _preprocess(slc: CAFSlice) -> None:
    """
    Precompute containment columns (cut.cont / cut.cont_full) for det systematics.

    Pure column definition (no row filtering); does not depend on the cut chain.
    """
    slc.cut_is_cont(cut=False)


def _rename_slim_columns(slc: CAFSlice) -> None:
    cols = slc.data.columns
    new_cols = [tuple("g4" if x == "slim" else x for x in c) for c in cols]
    new_cols = [tuple("xsec" if x == "GENIE" else x for x in c) for c in new_cols]
    new_cols = [tuple("flux" if x == "Flux" else x for x in c) for c in new_cols]
    slc.data.columns = slc.data.columns.from_tuples(new_cols, names=cols.names)


@dataclass
class AuxBases:
    lowe: CAFSlice
    offbeam_data: CAFSlice

    def apply_cuts_and_combine(
        self,
        target: CAFSlice,
        cut_chain: list[str],
        *,
        verbose: bool = False,
    ) -> None:
        aux_offbeam = self.offbeam_data.copy()
        aux_lowe = self.lowe.copy()
        _preprocess(aux_offbeam)
        _preprocess(aux_lowe)
        aux_offbeam.apply_cut_chain(cut_chain, verbose=verbose)
        aux_lowe.apply_cut_chain(cut_chain, verbose=verbose)
        target.combine(aux_offbeam, duplicate_ok=True, offset=OFFBEAM_COMBINE_OFFSET)
        target.combine(aux_lowe, duplicate_ok=True, offset=LOWE_COMBINE_OFFSET)


def _load_slice_chunk(
    cfg,
    fnames,
    cut_chain: list[str] | None,
    *,
    norm,
    sample_scale,
    scale_mode: str = "pot",
    file_index_offset: int = 0,
    rename_slim: bool = False,
    aux_bases: AuxBases | None = None,
    verbose: bool = False,
) -> CAFSlice:
    slc = CAFSlice.load(
        fnames,
        key=PAND_KEY,
        file_index_offset=file_index_offset,
        ncpu=cfg.ncpu,
        verbose=verbose,
    )
    _preprocess(slc)
    if rename_slim:
        _rename_slim_columns(slc)
    if scale_mode == "pot":
        slc.scale_to_pot(
            norm["POT_DATA"], sample_pot=sample_scale, overwrite=True, verbose=verbose
        )
        if aux_bases is not None:
            aux_bases.apply_cuts_and_combine(slc, cut_chain or [], verbose=verbose)
            slc.pot = float(norm["POT_DATA"])
    elif scale_mode == "livetime":
        slc.scale_to_livetime(
            norm["LIVETIME_DATA"],
            sample_livetime=sample_scale,
            overwrite=True,
            verbose=verbose,
        )
    elif scale_mode == "none":
        if aux_bases is not None:
            aux_bases.apply_cuts_and_combine(slc, cut_chain or [], verbose=verbose)
    else:
        raise ValueError(f"unknown scale_mode: {scale_mode}")
    if cut_chain:
        slc.apply_cut_chain(cut_chain, verbose=verbose)
    return slc


def _split_signal_background(slc: CAFSlice, categories: list[int]) -> tuple[CAFSlice, CAFSlice]:
    truth_event_type_col = slc.get_key("truth.event_type")[0]
    sig_mask = np.isin(slc.data[truth_event_type_col].values, categories)
    bg_mask = ~sig_mask
    return CAFSlice(slc.data[sig_mask], pot=slc.pot), CAFSlice(slc.data[bg_mask], pot=slc.pot)


def _ntuple_level(index) -> np.ndarray:
    if isinstance(index, pd.MultiIndex):
        if "__ntuple" in index.names:
            return index.get_level_values("__ntuple").to_numpy()
        return index.get_level_values(-1).to_numpy()
    return np.asarray(index)


def _concat_arrays(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate(chunks)


def _compute_xsec_unit(pot_data: float) -> float:
    try:
        import uproot

        flux = uproot.open(FLUX_FILE)
        numu_flux = flux["flux_sbnd_numu"].to_numpy()
        integrated_flux = numu_flux[0].sum() / 1e4
        integrated_flux *= pot_data / 1e6
    except Exception:
        integrated_flux = NUMU_INTEGRATED_FLUX * (pot_data / 1e6)
    return 1.0 / (NUMBER_TARGETS_FV * integrated_flux)


def _variable_specs(differential_edges, xsec_unit):
    return [
        ("costheta", COSTHETA_BINS, "mu.pfp.trk.costheta", "truth.mu.dir.z", xsec_unit),
        ("momentum", MOMENTUM_BINS, "mu.pfp.trk.P.p_muon", "truth.mu.totp", xsec_unit),
        ("differential", differential_edges, "bin.differential", "true_bin.differential", xsec_unit),
        ("bcfm", BCFM_BINS, "slc.barycenterFM.score", None, None),
        ("flashpe", FLASHPE_BINS, "slc.barycenterFM.flashPEs", None, None),
        ("startx", TRACK_XY_BINS, "mu.pfp.trk.start.x", None, None),
        ("starty", TRACK_XY_BINS, "mu.pfp.trk.start.y", None, None),
        ("startz", TRACK_Z_BINS, "mu.pfp.trk.start.z", None, None),
        ("endx", TRACK_XY_BINS, "mu.pfp.trk.end.x", None, None),
        ("endy", TRACK_XY_BINS, "mu.pfp.trk.end.y", None, None),
        ("endz", TRACK_Z_BINS, "mu.pfp.trk.end.z", None, None),
        ("phi", TRACK_PHI_BINS, "mu.pfp.trk.phi", None, None),
        ("theta", TRACK_THETA_BINS, "mu.pfp.trk.theta", None, None),
        ("length", TRACK_LENGTH_BINS, "mu.pfp.trk.len", None, None),
        ("vtxx", TRACK_XY_BINS, "slc.vertex.x", None, None),
        ("vtyy", TRACK_XY_BINS, "slc.vertex.y", None, None),
        ("vtzx", TRACK_Z_BINS, "slc.vertex.z", None, None),
    ]


def _get_values(slc: CAFSlice, key: str | None) -> np.ndarray | None:
    if key is None:
        return None
    col = slc.get_key(key)[0]
    return slc.data[col].values


def _prepare_aux(cfg, file_map, norm, verbose: bool) -> AuxBases:
    slc_lowe = CAFSlice.load(
        file_map["MC_LOWE_FNAMES"], key=PAND_KEY, ncpu=cfg.ncpu, verbose=verbose
    )
    _preprocess(slc_lowe)
    slc_lowe.scale_to_pot(
        norm["POT_DATA"], sample_pot=norm["POT_MC_LOWE_FULL"], overwrite=True, verbose=verbose
    )
    slc_lowe.pot = float(norm["POT_DATA"])
    slc_offbeam_data = CAFSlice.load(
        file_map["DATA_OFFBEAM_FNAMES"], key=PAND_KEY, ncpu=cfg.ncpu, verbose=verbose
    )
    _preprocess(slc_offbeam_data)
    slc_offbeam_data.scale_to_livetime(
        norm["LIVETIME_DATA"],
        sample_livetime=norm["LIVETIME_OFFBEAM_DATA_FULL"],
        overwrite=True,
        f=F_SCALE,
        verbose=verbose,
    )
    slc_offbeam_data.livetime = float(norm["LIVETIME_DATA"])
    return AuxBases(lowe=slc_lowe, offbeam_data=slc_offbeam_data)


def _accumulate_offbeam_mc_at_cut(
    cfg, file_map, norm, cut_chain, verbose: bool
) -> CAFSlice:
    parts: list[pd.DataFrame] = []
    file_base = 0
    for chunk_files in chunk_file_list(file_map["OFFBEAM_FNAMES"], cfg.chunk_nfiles):
        slc = _load_slice_chunk(
            cfg,
            chunk_files,
            cut_chain,
            norm=norm,
            sample_scale=norm["LIVETIME_OFFBEAM_FULL"],
            scale_mode="livetime",
            file_index_offset=file_base,
            verbose=verbose,
        )
        file_base += len(chunk_files)
        parts.append(slc.data)
    return CAFSlice(
        pd.concat(parts) if len(parts) > 1 else parts[0],
        livetime=norm["LIVETIME_OFFBEAM_FULL"],
        duplicate_ok=True,
    )


def _accumulate_nominal_cosmic_signal_at_cut(
    cfg,
    file_map,
    norm,
    cut_chain,
    categories: list[int],
    verbose: bool,
) -> CAFSlice:
    """MC nominal at cut, signal category only (shared cosmic signal leg; no lowe)."""
    parts: list[pd.DataFrame] = []
    file_base = 0
    for chunk_files in chunk_file_list(file_map["MC_NOMINAL_FNAMES"], cfg.chunk_nfiles):
        slc = _load_slice_chunk(
            cfg,
            chunk_files,
            cut_chain,
            norm=norm,
            sample_scale=norm["POT_NOMINAL"],
            scale_mode="pot",
            file_index_offset=file_base,
            verbose=verbose,
        )
        file_base += len(chunk_files)
        if len(slc.data) > 0:
            parts.append(slc.data)
    if not parts:
        return CAFSlice(pd.DataFrame(), pot=float(norm["POT_NOMINAL"]))
    data = pd.concat(parts) if len(parts) > 1 else parts[0]
    slc = CAFSlice(data, pot=float(norm["POT_DATA"]), duplicate_ok=True)
    slc_sig, _ = _split_signal_background(slc, categories)
    return slc_sig


def _accumulate_nominal_slice(
    cfg,
    file_map,
    norm,
    cut_chain,
    aux_bases: AuxBases,
    verbose: bool,
) -> CAFSlice:
    parts: list[pd.DataFrame] = []
    file_base = 0
    for chunk_files in chunk_file_list(file_map["MC_NOMINAL_FNAMES"], cfg.chunk_nfiles):
        slc = _load_slice_chunk(
            cfg,
            chunk_files,
            cut_chain,
            norm=norm,
            sample_scale=norm["POT_NOMINAL"],
            scale_mode="pot",
            file_index_offset=file_base,
            verbose=verbose,
        )
        file_base += len(chunk_files)
        parts.append(slc.data)
    if not parts:
        raise ValueError("MC_NOMINAL_FNAMES produced no data")
    nominal = CAFSlice(
        pd.concat(parts) if len(parts) > 1 else parts[0],
        pot=norm["POT_NOMINAL"],
        duplicate_ok=True,
    )
    aux_bases.apply_cuts_and_combine(nominal, cut_chain, verbose=verbose)
    nominal.pot = float(norm["POT_DATA"])
    return nominal


def _build_calo_slices_at_muon(nominal_at_muon: CAFSlice) -> list[CAFSlice]:
    slcs_calo = [nominal_at_muon.copy(duplicate_ok=True) for _ in CALO_SUFFIXES]
    base_has_muon = nominal_at_muon.get_key("has_muon")[0]
    for slc, suffix in zip(slcs_calo, CALO_SUFFIXES):
        has_muon_key = f"has_muon{suffix}"
        mask = _ntuple_level(slc.data.index) >= COSMIC_NT_LO
        if int(mask.sum()) > 0 and base_has_muon in slc.data.columns:
            values = np.array(slc.data.loc[mask, base_has_muon].values, dtype=bool)
        else:
            values = np.zeros(int(mask.sum()), dtype=bool)
        slc.add_cols(has_muon_key, values, conditions=mask, fill=False, verbose=False)
        has_muon_col = slc.get_key(has_muon_key)[0]
        slc.data = slc.data[slc.data[has_muon_col].astype(bool)]
    return slcs_calo


def _init_det_var_systems(specs, var: str) -> dict[str, Systematics]:
    systems_var: dict[str, Systematics] = {}
    for name, bins, _reco_key, _true_key, xsec in specs:
        systems_var[name] = Systematics.for_chunked_build(
            name,
            bins,
            keys=[var],
            xsec_unit=xsec,
            stype="Det",
            pattern=None,
        )
    return systems_var


def _apply_chunk_cv_truth(sys_obj: Systematics) -> None:
    """
    Promote filtered-nominal CV from _chunk_acc so process_det_systematics
    can build xsec smearing (needs eff_truth / sel_truth on self).
    """
    cv = sys_obj._chunk_acc["cv"]
    sys_obj.sel = cv["sel"]
    sys_obj.sel_background = cv["sel_background"]
    if not cv["has_truth"]:
        return
    sys_obj.sig_truth = cv["sig_truth"]
    sys_obj.sel_truth = cv["sel_truth"]
    sys_obj.sel_background_truth = cv["sel_background_truth"]
    sys_obj.eff_truth = compute_efficiency(sys_obj.sig_truth, sys_obj.sel_truth)
    with np.errstate(divide="ignore", invalid="ignore"):
        sys_obj.eff_reco = np.where(sys_obj.sig_truth > 0, sys_obj.sel / sys_obj.sig_truth, 0.0)
    sys_obj.smearing = cv["smearing"]
    sys_obj.response = convert_smearing_to_response(sys_obj.smearing, sys_obj.eff_truth)
    sys_obj.sigma_tilde = compute_sigma_tilde(
        sys_obj.response, sys_obj.sig_truth, sys_obj.sel_background, sys_obj.xsec_unit
    )


def _sync_cv_acc_from_promoted(sys_obj: Systematics) -> None:
    """Mirror promoted nominal CV into _chunk_acc so finalize_chunked_build preserves it."""
    cv = sys_obj._chunk_acc["cv"]
    cv["sel"] = np.copy(sys_obj.sel)
    cv["sel_background"] = np.copy(sys_obj.sel_background)
    if sys_obj.sig_truth is None:
        return
    cv["has_truth"] = True
    cv["sig_truth"] = np.copy(sys_obj.sig_truth)
    cv["sel_truth"] = np.copy(sys_obj.sel_truth)
    cv["sel_background_truth"] = np.copy(sys_obj.sel_background_truth)
    cv["smearing"] = np.copy(sys_obj.smearing)


def _promote_nominal_cv_truth(dst: Systematics, src: Systematics) -> None:
    """Copy filtered-nominal CV (metadata) from src onto dst."""
    dst.sel = np.copy(src.sel)
    dst.sel_background = np.copy(src.sel_background)
    if src.sig_truth is None:
        dst.sig_truth = None
        dst.sel_truth = None
        dst.sel_background_truth = None
        dst.eff_truth = None
        dst.eff_reco = None
        dst.smearing = None
        dst.response = None
        dst.sigma_tilde = None
        return
    dst.sig_truth = np.copy(src.sig_truth)
    dst.sel_truth = np.copy(src.sel_truth)
    dst.sel_background_truth = np.copy(src.sel_background_truth)
    dst.eff_truth = np.copy(src.eff_truth)
    dst.eff_reco = np.copy(src.eff_reco)
    dst.smearing = np.copy(src.smearing)
    dst.response = np.copy(src.response)
    dst.sigma_tilde = np.copy(src.sigma_tilde)


def _pin_slim_cv_sel(
    sys_obj: Systematics, keys: tuple[str, ...] = ("xsec", "flux", "g4")
) -> None:
    """Pin slim CV on RW groups so nominal promote does not steal their reference."""
    slim_cv = sys_obj.sel + sys_obj.sel_background
    for key in keys:
        if key not in sys_obj.systematics:
            continue
        sd = sys_obj.systematics[key]
        if sd.get("cv_sel") is None:
            sd["cv_sel"] = np.copy(slim_cv)
        if sys_obj.sigma_tilde is not None and sd.get("cv_sigma_tilde") is None:
            sd["cv_sigma_tilde"] = np.copy(sys_obj.sigma_tilde)


def _first_det_var_with_files(file_map) -> tuple[str | None, list[str] | None]:
    for var, flist in zip(file_map["DET_VARS"], file_map["DET_FNAMES"]):
        if flist:
            return var, flist
    return None, None


def _accumulate_shared_nominal_cv(
    systems: dict[str, Systematics],
    specs,
    file_map,
    norm,
    cut_chain,
    categories,
    cfg,
    aux_bases: AuxBases,
    pot_scaling: dict,
    verbose: bool,
    *,
    pot_debug: DetPotDebug | None = None,
) -> None:
    """Build metadata CV once from filtered nominal MC + aux (same sample as det variations)."""
    ref_var, _ = _first_det_var_with_files(file_map)
    if ref_var is None:
        return

    dbg = pot_debug or DetPotDebug.for_build(False)
    nom_fnames = file_map["MC_NOMINAL_FNAMES"]
    var_scale = pot_scaling["variations"][ref_var]
    base_pot_nom = float(norm["POT_NOMINAL"])
    ratio_nom = float(var_scale["event_ratio_nominal"])
    events = load_variation_events(cfg, ref_var)
    systems_cv = _init_det_var_systems(specs, ref_var)
    cv_stats = DetPotChunkStats()

    file_base = 0
    for cv_chunk_idx, chunk_files in enumerate(
        chunk_file_list(nom_fnames, cfg.chunk_nfiles)
    ):
        slc = _load_slice_chunk(
            cfg,
            chunk_files,
            cut_chain,
            norm=norm,
            sample_scale=0.0,
            scale_mode="none",
            aux_bases=None,
            file_index_offset=file_base,
            verbose=verbose,
        )
        file_base += len(chunk_files)
        _accumulate_det_chunk(
            slc,
            events,
            categories,
            specs,
            systems_cv,
            "cv",
            norm=norm,
            sample_pot=base_pot_nom,
            cut_chain=cut_chain,
            aux_bases=aux_bases,
            chunk_idx=cv_chunk_idx,
            var=ref_var,
            event_ratio=ratio_nom,
            verbose=verbose,
            pot_debug=dbg,
            chunk_stats=cv_stats,
        )

    for name in systems_cv:
        _apply_chunk_cv_truth(systems_cv[name])
        if name in systems:
            _promote_nominal_cv_truth(systems[name], systems_cv[name])


def _merge_det_var_systematics(main_sys: Systematics, var_sys: Systematics, var: str) -> None:
    if var not in var_sys.systematics:
        return
    main_sys.systematics[var] = copy.deepcopy(var_sys.systematics[var])


def _init_systems(specs, slc_data, truth_keys) -> dict[str, Systematics]:
    """RW-only systematics; det vars are merged later, CALO initialized before calo step."""
    systems: dict[str, Systematics] = {}
    for name, bins, reco_key, _true_key, xsec in specs:
        data_vals = _get_values(slc_data, reco_key)
        systems[name] = Systematics.for_chunked_build(
            name,
            bins,
            truth_keys,
            xsec_unit=xsec,
            pattern=SLIM_KEYS,
            stype="RW",
            data=data_vals,
            genweights_data=np.ones(len(data_vals), dtype=float),
        )
    return systems


def _ensure_calo_systematics(systems: dict[str, Systematics]) -> None:
    for sys_obj in systems.values():
        sys_obj._initialize_from_keys(CALO_VARS, stype="Det", pattern=None)


@dataclass
class DetPotChunkStats:
    n_before: int = 0
    n_after: int = 0
    n_aux_before: int = 0
    n_aux_after: int = 0
    genweight_sum_before: float = 0.0
    genweight_sum_after: float = 0.0

    def add(self, other: DetPotChunkStats) -> None:
        self.n_before += other.n_before
        self.n_after += other.n_after
        self.n_aux_before += other.n_aux_before
        self.n_aux_after += other.n_aux_after
        self.genweight_sum_before += other.genweight_sum_before
        self.genweight_sum_after += other.genweight_sum_after


def _mc_row_mask(index) -> np.ndarray:
    return _ntuple_level(index) < OFFBEAM_COMBINE_OFFSET


def _slice_mc_only(slc: CAFSlice) -> CAFSlice:
    """MC nominal/variation rows only (exclude combined offbeam/lowe aux)."""
    mask = _mc_row_mask(slc.data.index)
    if mask.all():
        return slc
    return CAFSlice(slc.data.loc[mask], pot=slc.pot)


def _count_aux_rows(slc: CAFSlice) -> int:
    return int(np.sum(~_mc_row_mask(slc.data.index)))


def _mc_genweight_sum(slc: CAFSlice, categories: list[int]) -> float:
    if len(slc.data) == 0:
        return 0.0
    truth_event_type_col = slc.get_key("truth.event_type")[0]
    mc_mask = _mc_row_mask(slc.data.index)
    if not mc_mask.any():
        return 0.0
    gen_col = slc.get_key("genweight")[0]
    types = slc.data.loc[mc_mask, truth_event_type_col].values
    sig_mask = np.isin(types, categories)
    weights = slc.data.loc[mc_mask, gen_col].astype(float).values
    return float(weights[sig_mask].sum())


def _scale_det_slice_after_filter(
    slc: CAFSlice,
    norm: dict,
    sample_pot: float,
    *,
    n_before: int,
    verbose: bool = False,
    pot_debug: DetPotDebug | None = None,
) -> float:
    """
    Scale det chunk to data POT after common-event filter (detsys.ipynb style).

    sample_pot is full POT_NOMINAL or POT_DET[var] from normalization.json;
    pot_eff = sample_pot * (n_after / n_before). event_ratio from pot_scaling.json
    is not applied here (only used for event lists).
    """
    n_after = len(slc.data)
    if n_after == 0:
        return 0.0
    pot_eff = sample_pot * (n_after / n_before) if n_before > 0 else sample_pot
    if pot_eff <= 0:
        return 0.0
    dbg = pot_debug or DetPotDebug.for_build(False)
    dbg.log_scale_apply(
        n_rows_scaled=n_after,
        n_aux_rows=int(np.sum(~_mc_row_mask(slc.data.index))),
        sample_pot=sample_pot,
        n_before=n_before,
        n_after=n_after,
        pot_eff=pot_eff,
    )
    slc.scale_to_pot(
        norm["POT_DATA"], sample_pot=pot_eff, overwrite=True, verbose=verbose
    )
    slc.pot = float(norm["POT_DATA"])
    return pot_eff


def _warn_pot_scaling_mismatch(
    var: str, norm: dict, var_scale: dict, *, cfg
) -> None:
    """Warn when full-build POT in pot_scaling.json disagrees with current file_map."""
    if cfg.small or cfg.tiny:
        return
    pot_det = norm.get("POT_DET", {}).get(var)
    pot_det_full = var_scale.get("POT_DET_FULL")
    if pot_det is None or pot_det_full is None or pot_det_full <= 0:
        return
    ratio = pot_det / float(pot_det_full)
    if ratio < 0.1 or ratio > 10.0:
        warnings.warn(
            f"det var {var}: on-the-fly POT_DET ({pot_det:.3g}) vs pot_scaling "
            f"POT_DET_FULL ({pot_det_full:.3g}) ratio {ratio:.3g}; "
            "re-run build_det_event_lists.py or check det file lists",
            stacklevel=2,
        )


def _accumulate_det_chunk(
    slc: CAFSlice,
    events,
    categories: list[int],
    specs,
    systems_var: dict[str, Systematics],
    mode: str,
    *,
    norm: dict,
    sample_pot: float,
    cut_chain: list[str],
    aux_bases: AuxBases | None = None,
    chunk_idx: int = 0,
    var: str | None = None,
    event_ratio: float | None = None,
    verbose: bool = False,
    pot_debug: DetPotDebug | None = None,
    chunk_stats: DetPotChunkStats | None = None,
) -> None:
    dbg = pot_debug or DetPotDebug.for_build(False)
    stage_ctx = ChunkStageContext(var=var, mode=mode, chunk_idx=chunk_idx, event_ratio=event_ratio)
    entry_snap = DetPotDebug.snapshot(slc, categories)

    n_aux_before = _count_aux_rows(slc)
    mc_slc = _slice_mc_only(slc)
    n_before = len(mc_slc.data)
    gen_before = _mc_genweight_sum(mc_slc, categories)
    mc_slc = filter_slice_to_events(mc_slc, events)
    n_after = len(mc_slc.data)
    if n_after == 0:
        dbg.trace_det_chunk_empty_after_filter(
            ctx=stage_ctx, entry=entry_snap, n_mc_before_filter=n_before
        )
        return
    post_filter_snap = DetPotDebug.snapshot(mc_slc, categories)
    pot_eff = _scale_det_slice_after_filter(
        mc_slc,
        norm,
        sample_pot,
        n_before=n_before,
        verbose=verbose,
        pot_debug=dbg,
    )
    post_scale_snap = DetPotDebug.snapshot(mc_slc, categories)
    if aux_bases is not None and chunk_idx == 0:
        aux_bases.apply_cuts_and_combine(mc_slc, cut_chain, verbose=verbose)
        mc_slc.pot = float(norm["POT_DATA"])
    slc = mc_slc
    post_aux_snap = DetPotDebug.snapshot(slc, categories)
    n_aux_after = _count_aux_rows(slc)
    gen_after = _mc_genweight_sum(slc, categories)
    if chunk_stats is not None:
        chunk_stats.n_before += n_before
        chunk_stats.n_after += n_after
        chunk_stats.n_aux_before += n_aux_before
        chunk_stats.n_aux_after += n_aux_after
        chunk_stats.genweight_sum_before += gen_before
        chunk_stats.genweight_sum_after += gen_after
    dbg.trace_det_chunk(
        ctx=stage_ctx,
        entry=entry_snap,
        post_filter=post_filter_snap,
        post_scale=post_scale_snap,
        post_aux=post_aux_snap,
        sample_pot=sample_pot,
        pot_eff=pot_eff,
        n_mc_before_filter=n_before,
        pot_data=float(norm["POT_DATA"]),
    )

    if mode == "cv":
        slc_sig_full, _ = _split_signal_background(slc, categories)
        slc_sig, slc_bg = _split_signal_background(slc, categories)
        gen_sel = _get_values(slc_sig, "genweight")
        gen_bg = _get_values(slc_bg, "genweight")
        gen_sig_full = _get_values(slc_sig_full, "genweight")
        for name, _bins, reco_key, true_key, _xsec in specs:
            if name not in systems_var:
                continue
            systems_var[name].accumulate_cv_chunk(
                reco_sel=_get_values(slc_sig, reco_key),
                reco_sel_background=_get_values(slc_bg, reco_key),
                genweights_sel=gen_sel,
                genweights_sel_background=gen_bg,
                true_sig=_get_values(slc_sig_full, true_key) if true_key else None,
                true_sel=_get_values(slc_sig, true_key) if true_key else None,
                true_sel_background=_get_values(slc_bg, true_key) if true_key else None,
                genweights_sig=gen_sig_full,
            )
        return

    if mode == "variation":
        if var is None:
            raise ValueError("var required for variation det chunk mode")
        slc_sig, slc_bg = _split_signal_background(slc, categories)
        for name, _bins, reco_key, true_key, _xsec in specs:
            if name not in systems_var:
                continue
            sys_obj = systems_var[name]
            reco_sel = _get_values(slc_sig, reco_key)
            reco_bg = _get_values(slc_bg, reco_key)
            gen_sel = _get_values(slc_sig, "genweight")
            gen_bg = _get_values(slc_bg, "genweight")
            true_sig = _get_values(slc_sig, true_key) if true_key else None
            true_sel = true_sig
            true_bg = _get_values(slc_bg, true_key) if true_key else None
            sys_obj.process_det_systematics(
                [reco_sel],
                [reco_bg],
                [gen_sel],
                [gen_bg],
                true_sig_vars=[true_sig] if true_sig is not None else None,
                true_sel_vars=[true_sel] if true_sel is not None else None,
                true_sel_background_vars=[true_bg] if true_bg is not None else None,
                sys_names=[var],
                accumulate=chunk_idx > 0,
            )
        return

    raise ValueError(f"unknown det chunk mode: {mode}")


def _process_det_chunks(
    systems,
    specs,
    file_map,
    norm,
    cut_chain,
    categories,
    cfg,
    aux_bases: AuxBases,
    pot_scaling: dict,
    verbose: bool,
    *,
    pot_debug: DetPotDebug | None = None,
):
    dbg = pot_debug or DetPotDebug.for_build(False)
    det_vars = file_map["DET_VARS"]
    det_fnames = file_map["DET_FNAMES"]

    for var, flist in tqdm(
        list(zip(det_vars, det_fnames)),
        desc="det vars",
        unit="var",
        disable=not verbose,
    ):
        if not flist:
            continue
        var_scale = pot_scaling["variations"][var]
        _warn_pot_scaling_mismatch(var, norm, var_scale, cfg=cfg)
        base_pot_var = float(norm["POT_DET"][var])
        ratio_var = float(var_scale["event_ratio_var"])
        events = load_variation_events(cfg, var)
        dbg.log_var_header(
            var, var_scale, norm, float(var_scale["event_ratio_nominal"]), ratio_var
        )
        rse_sample_path = dbg.start_rse_sample_file(cfg.save_dir, var)
        systems_var = _init_det_var_systems(specs, var)
        var_stats = DetPotChunkStats()

        for name in systems_var:
            if name in systems:
                _promote_nominal_cv_truth(systems_var[name], systems[name])

        file_base = 0
        var_flist_chunks = chunk_file_list(flist, cfg.chunk_nfiles)
        n_var_chunks = len(var_flist_chunks)
        for chunk_idx, chunk_files in enumerate(var_flist_chunks):
            slc = _load_slice_chunk(
                cfg,
                chunk_files,
                cut_chain,
                norm=norm,
                sample_scale=0.0,
                scale_mode="none",
                aux_bases=None,
                file_index_offset=file_base,
                verbose=verbose,
            )
            file_base += len(chunk_files)
            if chunk_idx == 0:
                dbg.append_rse_sample_section(
                    _slice_mc_only(slc),
                    "variation chunk0 (first 50 mc rows, sorted by RSE)",
                    rse_sample_path,
                )
            _accumulate_det_chunk(
                slc,
                events,
                categories,
                specs,
                systems_var,
                "variation",
                norm=norm,
                sample_pot=base_pot_var,
                cut_chain=cut_chain,
                aux_bases=aux_bases,
                chunk_idx=chunk_idx,
                var=var,
                event_ratio=ratio_var,
                verbose=verbose,
                pot_debug=dbg,
                chunk_stats=var_stats,
            )

        for name in systems_var:
            systems_var[name].finalize_chunked_build()

        costheta_sel = None
        if "costheta" in systems_var and var in systems_var["costheta"].systematics:
            costheta_sel = systems_var["costheta"].systematics[var]["sel"]
        dbg.log_aggregate(
            var,
            DetPotChunkStats(),
            var_stats,
            costheta_sel,
            n_cv_chunks=0,
            n_var_chunks=n_var_chunks,
        )

        for name in systems_var:
            if name in systems:
                _merge_det_var_systematics(systems[name], systems_var[name], var)


def _process_calo_systematics(
    systems,
    specs,
    slcs_calo: list[CAFSlice],
    categories: list[int],
):
    for calo_var, slc in zip(CALO_VARS, slcs_calo):
        slc_sig, slc_bg = _split_signal_background(slc, categories)
        for name, _bins, reco_key, true_key, _xsec in specs:
            if name not in systems:
                continue
            sys_obj = systems[name]
            reco_sel = _get_values(slc_sig, reco_key)
            reco_bg = _get_values(slc_bg, reco_key)
            gen_sel = _get_values(slc_sig, "genweight")
            gen_bg = _get_values(slc_bg, "genweight")
            true_sig = _get_values(slc_sig, true_key) if true_key else None
            sys_obj.process_det_systematics(
                [reco_sel],
                [reco_bg],
                [gen_sel],
                [gen_bg],
                true_sig_vars=[true_sig] if true_sig is not None else None,
                true_sel_vars=[true_sig] if true_sig is not None else None,
                true_sel_background_vars=[_get_values(slc_bg, true_key)] if true_key else None,
                sys_names=[calo_var],
                accumulate=False,
            )


def _process_cosmic(
    system,
    name,
    bins,
    reco_key,
    xsec_unit,
    slc_nominal_sig: CAFSlice,
    slc_offbeam_mc: CAFSlice,
    slc_offbeam_data: CAFSlice,
    cut,
    cfg,
):
    """CV: nominal signal + data offbeam; variation: nominal signal + MC offbeam."""
    reco_sel_nom = _get_values(slc_nominal_sig, reco_key)
    gen_sel_nom = _get_values(slc_nominal_sig, "genweight")
    reco_bg_cv = _get_values(slc_offbeam_data, reco_key)
    gen_bg_cv = _get_values(slc_offbeam_data, "genweight")
    reco_bg_var = _get_values(slc_offbeam_mc, reco_key)
    gen_bg_var = _get_values(slc_offbeam_mc, "genweight")

    cosmic = Systematics(
        name,
        bins,
        reco_sel_nom,
        reco_bg_cv,
        gen_sel_nom,
        gen_sel_nom,
        gen_bg_cv,
        xsec_unit=xsec_unit,
        true_sig=None,
        true_sel=None,
        true_sel_background=None,
        keys=["cosmic"],
        stype="Cosmic",
        pattern=None,
        data=system._data,
        genweights_data=system._genweights_data,
    )
    cosmic.process_det_systematics(
        [reco_sel_nom],
        [reco_bg_var],
        [gen_sel_nom],
        [gen_bg_var],
        true_sel_vars=None,
        sys_names=["cosmic"],
    )
    do_xsec_cov = (cut == cfg.cuts[-1]) and name in ("costheta", "momentum", "differential")
    cosmic.compute_covariances(keys=["cosmic"], compute_xsec_cov=do_xsec_cov)
    system.combine(cosmic, store_other=True, other_name="cosmic_data")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build chunked systematics universes (slim RW, det, CALO, cosmic)."
    )
    parser.add_argument("--small", action="store_true", help="Use SMALL file subset.")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Load one file per sample list (smoke test; overrides --small).",
    )
    parser.add_argument("--chunk-size", type=int, default=8, help="Files per chunk for slim/det/nominal/offbeam.")
    parser.add_argument("--day", default=datetime.now().strftime("%Y%m%d"), help="Output day/tag.")
    parser.add_argument("--ncpu", type=int, default=1, help="Workers for CAF/HDF loading.")
    parser.add_argument("--dry-run", action="store_true", help="Print chunk plan only (skips POT scan).")
    parser.add_argument(
        "--refresh-normalization",
        action="store_true",
        help="Deprecated: normalization.json is always recomputed at startup.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print cut and load progress.",
    )
    parser.add_argument(
        "--debug-pot",
        action="store_true",
        help="Print det POT scaling diagnostics to stdout (pipe to logs/).",
    )
    args = parser.parse_args()
    verbose = args.verbose
    pot_debug = DetPotDebug.for_build(args.debug_pot)  # DEBUG-POT

    cfg = build_config(
        small=args.small,
        tiny=args.tiny,
        day=args.day,
        chunk_nfiles=args.chunk_size,
        ncpu=args.ncpu,
    )
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.plot_dir).mkdir(parents=True, exist_ok=True)
    file_map = build_file_map(cfg)
    print(
        f"File map subsample: tiny={cfg.tiny} small={cfg.small} "
        f"lowe={len(file_map['MC_LOWE_FNAMES'])} "
        f"data_offbeam={len(file_map['DATA_OFFBEAM_FNAMES'])}"
    )
    slim_chunks = chunk_file_list(file_map["MC_SLIM_FNAMES"], cfg.chunk_nfiles)
    cuts = cfg.cuts

    det_chunk_counts = {
        var: len(chunk_file_list(flist, cfg.chunk_nfiles))
        for var, flist in zip(file_map["DET_VARS"], file_map["DET_FNAMES"])
    }

    if args.dry_run:
        from detsys_det_match import pot_scaling_path, runs_csv_path

        artifact_status = {
            var: {
                "runs_csv": str(runs_csv_path(cfg, var)),
                "runs_csv_exists": runs_csv_path(cfg, var).is_file(),
            }
            for var in file_map["DET_VARS"]
        }
        print(json.dumps(
            {
                "cuts": cuts,
                "tiny": cfg.tiny,
                "small": cfg.small,
                "ncpu": cfg.ncpu,
                "chunk_size": cfg.chunk_nfiles,
                "n_slim_files": len(file_map["MC_SLIM_FNAMES"]),
                "n_slim_chunks": len(slim_chunks),
                "det_vars": file_map["DET_VARS"],
                "det_chunks": det_chunk_counts,
                "calo_vars": CALO_VARS,
                "n_offbeam_mc_files": len(file_map["OFFBEAM_FNAMES"]),
                "n_nominal_files": len(file_map["MC_NOMINAL_FNAMES"]),
                "pot_scaling_json": str(pot_scaling_path(cfg)),
                "pot_scaling_exists": pot_scaling_path(cfg).is_file(),
                "det_artifacts": artifact_status,
            },
            indent=2,
        ))
        return 0

    pot_scaling = require_artifacts(cfg, file_map["DET_VARS"])
    norm_path = normalization_json_path(cfg.save_dir)
    written = write_normalization_json(
        cfg.save_dir, compute_normalization(file_map, ncpu=cfg.ncpu)
    )
    print(f"Wrote normalization to {written}")
    norm = load_normalization_json(cfg.save_dir)
    print(f"Using normalization from {norm_path}")

    categories = [0, 1] if cfg.contained else [0, 1]
    binning2d = Binning2D(diff_momentum_bins_2d=DIFF_MOMENTUM_BINS_2D)
    differential_edges = binning2d.differential_edges
    xsec_unit = _compute_xsec_unit(norm["POT_DATA"])
    specs = _variable_specs(differential_edges, xsec_unit)

    aux_bases = _prepare_aux(cfg, file_map, norm, verbose)

    slcs_calo: list[CAFSlice] | None = None
    muon_chain = None

    for cut in cuts:
        cut_chain = cfg.cuts[: cfg.cuts.index(cut) + 1] if cut in cfg.cuts else [cut]
        print(f"Processing cut {cut} with chain {cut_chain}")

        if cut == "muon":
            muon_chain = cut_chain
            nominal_at_muon = _accumulate_nominal_slice(
                cfg, file_map, norm, muon_chain, aux_bases, verbose
            )
            slcs_calo = _build_calo_slices_at_muon(nominal_at_muon)

        if slcs_calo is not None and cut in {"cont_full", "cont"}:
            for slc in slcs_calo:
                _preprocess(slc)
                slc.apply_cut(cut, verbose=verbose)

        slc_data = CAFSlice.load(
            file_map["DATA_FNAMES"],
            key=PAND_KEY,
            ncpu=cfg.ncpu,
            verbose=verbose,
        )
        _preprocess(slc_data)
        slc_data.apply_cut_chain(cut_chain, verbose=verbose)

        slc_lowe = aux_bases.lowe.copy()
        slc_offbeam_data = aux_bases.offbeam_data.copy()
        _preprocess(slc_lowe)
        _preprocess(slc_offbeam_data)
        slc_lowe.apply_cut_chain(cut_chain, verbose=verbose)
        slc_offbeam_data.apply_cut_chain(cut_chain, verbose=verbose)

        slc_offbeam_mc = _accumulate_offbeam_mc_at_cut(
            cfg, file_map, norm, cut_chain, verbose
        )
        slc_nominal_sig = _accumulate_nominal_cosmic_signal_at_cut(
            cfg, file_map, norm, cut_chain, categories, verbose
        )

        systems: dict[str, Systematics] = {}
        truth_keys = None

        slim_file_base = 0
        for chunk_idx, chunk_files in enumerate(
            tqdm(slim_chunks, desc=f"{cut} slim", unit="chunk", disable=not verbose)
        ):
            slc_chunk = _load_slice_chunk(
                cfg,
                chunk_files,
                None,
                norm=norm,
                sample_scale=norm["POT_MC_SLIM_FULL"],
                scale_mode="pot",
                rename_slim=True,
                file_index_offset=slim_file_base,
                verbose=verbose,
            )
            slim_file_base += len(chunk_files)
            if chunk_idx == 0:
                slc_chunk.combine(slc_offbeam_data, duplicate_ok=True, offset=OFFBEAM_COMBINE_OFFSET)
                slc_chunk.combine(slc_lowe, duplicate_ok=True, offset=LOWE_COMBINE_OFFSET)
            slc_chunk.pot = float(norm["POT_DATA"])

            slc_sig_full, _ = _split_signal_background(slc_chunk, categories)
            slc_chunk.apply_cut_chain(cut_chain, verbose=verbose)
            slc_sel_sig, slc_sel_bg = _split_signal_background(slc_chunk, categories)
            if len(slc_sel_sig.data) == 0 and len(slc_sel_bg.data) == 0:
                continue

            if not systems:
                truth_keys = slc_sig_full.data.truth.keys()
                systems = _init_systems(specs, slc_data, truth_keys)

            gen_sel = _get_values(slc_sel_sig, "genweight")
            gen_bg = _get_values(slc_sel_bg, "genweight")
            gen_sig_full = _get_values(slc_sig_full, "genweight")

            for name, _bins, reco_key, true_key, _xsec in specs:
                sys_obj = systems[name]
                sys_obj.accumulate_cv_chunk(
                    reco_sel=_get_values(slc_sel_sig, reco_key),
                    reco_sel_background=_get_values(slc_sel_bg, reco_key),
                    genweights_sel=gen_sel,
                    genweights_sel_background=gen_bg,
                    true_sig=_get_values(slc_sig_full, true_key) if true_key else None,
                    true_sel=_get_values(slc_sel_sig, true_key) if true_key else None,
                    true_sel_background=_get_values(slc_sel_bg, true_key) if true_key else None,
                    genweights_sig=gen_sig_full,
                )
                sys_obj.process_systematics_chunk(
                    slc_sig_full.data,
                    slc_sel_sig.data,
                    slc_sel_bg.data,
                    sys_names=SLIM_KEYS,
                )

        if not systems:
            print(f"No slim events after cuts for {cut}, skipping")
            continue

        pot_debug.log_cosmic_sample_lengths(
            cut, slc_nominal_sig, slc_offbeam_mc, slc_offbeam_data
        )

        for name, _bins, _reco_key, _true_key, _xsec in specs:
            if name in systems:
                systems[name].finalize_chunked_build()
                _pin_slim_cv_sel(systems[name])

        _accumulate_shared_nominal_cv(
            systems,
            specs,
            file_map,
            norm,
            cut_chain,
            categories,
            cfg,
            aux_bases,
            pot_scaling,
            verbose,
            pot_debug=pot_debug,
        )

        for name, _bins, _reco_key, _true_key, _xsec in specs:
            if name in systems:
                _sync_cv_acc_from_promoted(systems[name])
                systems[name].finalize_chunked_build()

        _process_det_chunks(
            systems,
            specs,
            file_map,
            norm,
            cut_chain,
            categories,
            cfg,
            aux_bases,
            pot_scaling,
            verbose,
            pot_debug=pot_debug,
        )

        if slcs_calo is not None and cut in CALO_CUTS:
            _ensure_calo_systematics(systems)
            _process_calo_systematics(systems, specs, slcs_calo, categories)

        cut_universe_dir = f"{cfg.save_dir}/{cut}"

        for name, bins, reco_key, _true_key, xsec in specs:
            if name not in systems:
                continue
            _process_cosmic(
                systems[name],
                name,
                bins,
                reco_key,
                xsec,
                slc_nominal_sig,
                slc_offbeam_mc,
                slc_offbeam_data,
                cut,
                cfg,
            )
            systems[name].save(cut_universe_dir, metadata_dir="metadata_detsys")
            print(f"Saved {name} universes to {cut_universe_dir}")

    print("Done building chunked universes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
