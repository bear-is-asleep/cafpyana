#!/usr/bin/env python3
"""Replay one det-var chunk through load/filter/scale for POT debugging."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CAFPYANA_WD = REPO_ROOT.parents[1]
for p in (str(REPO_ROOT), str(CAFPYANA_WD)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from detsys_config import build_config, build_file_map, chunk_file_list, compute_normalization
from detsys_det_match import load_variation_events, require_artifacts
from detsys_pot_debug import DetPotDebug

_build_path = Path(__file__).resolve().parent / "build_detsys_universes.py"
_spec = importlib.util.spec_from_file_location("build_detsys_universes", _build_path)
assert _spec and _spec.loader
bdu = importlib.util.module_from_spec(_spec)
sys.modules["build_detsys_universes"] = bdu
_spec.loader.exec_module(bdu)

DetPotChunkStats = bdu.DetPotChunkStats
_accumulate_det_chunk = bdu._accumulate_det_chunk
_apply_chunk_cv_truth = bdu._apply_chunk_cv_truth
_compute_xsec_unit = bdu._compute_xsec_unit
_init_det_var_systems = bdu._init_det_var_systems
_load_slice_chunk = bdu._load_slice_chunk
_prepare_aux = bdu._prepare_aux
_variable_specs = bdu._variable_specs


def _cut_chain(cfg, cut: str) -> list[str]:
    return cfg.cuts[: cfg.cuts.index(cut) + 1] if cut in cfg.cuts else [cut]


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug det POT scaling on one chunk per path.")
    parser.add_argument("--day", required=True, help="Output day/tag.")
    parser.add_argument("--var", default="pmtgain", help="Detector variation name.")
    parser.add_argument("--cut", default="precut", help="Cut stage to apply.")
    parser.add_argument("--small", action="store_true", help="Use SMALL file subset.")
    parser.add_argument("--tiny", action="store_true", help="One file per list.")
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()

    cfg = build_config(
        small=args.small,
        tiny=args.tiny,
        day=args.day,
        chunk_nfiles=args.chunk_size,
        ncpu=args.ncpu,
    )
    file_map = build_file_map(cfg)
    if args.var not in file_map["DET_VARS"]:
        raise SystemExit(f"Unknown var {args.var}; choices: {file_map['DET_VARS']}")

    var_idx = file_map["DET_VARS"].index(args.var)
    det_flist = file_map["DET_FNAMES"][var_idx]
    nom_flist = file_map["MC_NOMINAL_FNAMES"]
    if not det_flist:
        raise SystemExit(f"Empty file list for {args.var}")

    pot_scaling = require_artifacts(cfg, file_map["DET_VARS"])
    norm = compute_normalization(file_map, ncpu=cfg.ncpu)
    var_scale = pot_scaling["variations"][args.var]
    cut_chain = _cut_chain(cfg, args.cut)
    categories = [0, 1] if cfg.contained else [0, 1]
    events = load_variation_events(cfg, args.var)

    from sbnd.cafclasses.binning import Binning2D
    from sbnd.numu.numu_constants import DIFF_MOMENTUM_BINS_2D

    binning2d = Binning2D(diff_momentum_bins_2d=DIFF_MOMENTUM_BINS_2D)
    specs = _variable_specs(binning2d.differential_edges, _compute_xsec_unit(norm["POT_DATA"]))
    aux_bases = _prepare_aux(cfg, file_map, norm, verbose=False)
    systems_var = _init_det_var_systems(specs, args.var)
    pot_debug = DetPotDebug.for_build(True)

    sample_pot_nom = float(norm["POT_NOMINAL"])
    sample_pot_var = float(norm["POT_DET"][args.var])
    ratio_nom = float(var_scale["event_ratio_nominal"])
    ratio_var = float(var_scale["event_ratio_var"])

    print(f"[debug-pot-replay] day={args.day} var={args.var} cut={args.cut} chain={cut_chain}")
    pot_debug.log_var_header(args.var, var_scale, norm, ratio_nom, ratio_var)

    nom_chunk = chunk_file_list(nom_flist, cfg.chunk_nfiles)[0]
    var_chunk = chunk_file_list(det_flist, cfg.chunk_nfiles)[0]
    cv_stats = DetPotChunkStats()
    var_stats = DetPotChunkStats()

    slc_cv = _load_slice_chunk(
        cfg,
        nom_chunk,
        cut_chain,
        norm=norm,
        sample_scale=0.0,
        scale_mode="none",
        aux_bases=None,
        verbose=False,
    )
    _accumulate_det_chunk(
        slc_cv,
        events,
        categories,
        specs,
        systems_var,
        "cv",
        norm=norm,
        sample_pot=sample_pot_nom,
        cut_chain=cut_chain,
        aux_bases=aux_bases,
        chunk_idx=0,
        var=args.var,
        event_ratio=ratio_nom,
        pot_debug=pot_debug,
        chunk_stats=cv_stats,
    )
    _apply_chunk_cv_truth(systems_var["costheta"])

    slc_var = _load_slice_chunk(
        cfg,
        var_chunk,
        cut_chain,
        norm=norm,
        sample_scale=0.0,
        scale_mode="none",
        aux_bases=None,
        verbose=False,
    )
    _accumulate_det_chunk(
        slc_var,
        events,
        categories,
        specs,
        systems_var,
        "variation",
        norm=norm,
        sample_pot=sample_pot_var,
        cut_chain=cut_chain,
        aux_bases=aux_bases,
        chunk_idx=0,
        var=args.var,
        event_ratio=ratio_var,
        pot_debug=pot_debug,
        chunk_stats=var_stats,
    )
    systems_var["costheta"].finalize_chunked_build()

    costheta_sel = systems_var["costheta"].systematics[args.var]["sel"]
    pot_debug.log_aggregate(
        args.var,
        cv_stats,
        var_stats,
        costheta_sel,
        n_cv_chunks=1,
        n_var_chunks=1,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
