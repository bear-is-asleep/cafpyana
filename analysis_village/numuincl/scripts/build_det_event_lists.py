#!/usr/bin/env python3
"""Build per-variation common-event CSVs and pot_scaling.json for detector systematics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
CAFPYANA_WD = REPO_ROOT.parents[1]
for p in (str(REPO_ROOT), str(CAFPYANA_WD)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detsys_config import build_config, build_file_map, compute_normalization
from detsys_det_match import (
    build_variation_scaling_entry,
    pairwise_common_events,
    pot_scaling_path,
    runs_csv_path,
    write_pot_scaling_json,
    write_runs_csv,
)
from sbnd.cafclasses.nu import NU
from sbnd.cafclasses.parent import CAF

MCNU_KEY = "mcnu*"
HDR_KEY = "hdr*"


def _load_nominal_mcnu_hdr(file_map, ncpu: int):
    mcnu = NU.load(file_map["MC_NOMINAL_FNAMES"], key=MCNU_KEY, ncpu=ncpu, show_progress=False)
    hdr = CAF.load(file_map["MC_NOMINAL_FNAMES"], key=HDR_KEY, ncpu=ncpu, show_progress=False)
    return mcnu, hdr


def build_event_lists(
    cfg,
    file_map,
    norm,
    *,
    only_var: str | None = None,
    dry_run: bool = False,
) -> dict:
    nom_mcnu, nom_hdr = _load_nominal_mcnu_hdr(file_map, cfg.ncpu)
    pot_nominal_full = float(norm["POT_NOMINAL"])

    variations: dict = {}
    det_vars = file_map["DET_VARS"]
    det_fnames = file_map["DET_FNAMES"]

    for var, flist in tqdm(
        list(zip(det_vars, det_fnames)),
        desc="det variations",
        unit="var",
    ):
        if only_var is not None and var != only_var:
            continue
        if not flist:
            print(f"Skipping {var}: empty file list")
            continue

        csv_path = runs_csv_path(cfg, var)
        pot_det_full = float(norm["POT_DET"][var])

        if dry_run:
            print(f"{var}: {len(flist)} files -> {csv_path}")
            continue

        var_mcnu = NU.load(flist, key=MCNU_KEY, ncpu=cfg.ncpu, show_progress=False)
        var_hdr = CAF.load(flist, key=HDR_KEY, ncpu=cfg.ncpu, show_progress=False)
        events_df, n_nominal, n_variation, n_common = pairwise_common_events(
            nom_mcnu, nom_hdr, var_mcnu, var_hdr
        )
        write_runs_csv(events_df, csv_path)
        variations[var] = build_variation_scaling_entry(
            var=var,
            n_nominal=n_nominal,
            n_variation=n_variation,
            n_common=n_common,
            pot_nominal_full=pot_nominal_full,
            pot_det_full=pot_det_full,
            runs_csv=str(csv_path.relative_to(cfg.data_dir)),
        )
        print(
            f"{var}: n_common={n_common} / nom={n_nominal} var={n_variation} "
            f"ratio_nom={variations[var]['event_ratio_nominal']:.4f} "
            f"ratio_var={variations[var]['event_ratio_var']:.4f}"
        )

    if dry_run:
        return {}

    payload = {
        "version": cfg.version,
        "POT_NOMINAL_FULL": pot_nominal_full,
        "variations": variations,
    }
    out_path = pot_scaling_path(cfg)
    if only_var is not None and out_path.is_file():
        import json

        with out_path.open(encoding="utf-8") as f:
            merged = json.load(f)
        merged.setdefault("variations", {}).update(variations)
        merged["POT_NOMINAL_FULL"] = pot_nominal_full
        merged["version"] = cfg.version
        write_pot_scaling_json(cfg, merged)
    else:
        write_pot_scaling_json(cfg, payload)
    print(f"Wrote {out_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build det variation event lists and POT scaling.")
    parser.add_argument("--small", action="store_true", help="Use SMALL file subset.")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="One file per sample list (overrides --small).",
    )
    parser.add_argument("--ncpu", type=int, default=8, help="Workers for CAF/HDF loading.")
    parser.add_argument("--var", default=None, help="Process a single variation only.")
    parser.add_argument("--dry-run", action="store_true", help="Print paths only.")
    args = parser.parse_args()

    cfg = build_config(small=args.small, tiny=args.tiny, ncpu=args.ncpu)
    file_map = build_file_map(cfg)

    if args.dry_run:
        build_event_lists(cfg, file_map, {}, only_var=args.var, dry_run=True)
        print(f"pot_scaling -> {pot_scaling_path(cfg)}")
        return 0

    norm = compute_normalization(file_map, ncpu=cfg.ncpu)
    build_event_lists(cfg, file_map, norm, only_var=args.var, dry_run=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
