#!/usr/bin/env python3
"""
Move grid job dirs from scratch to pandora:
  {source}/{timestamp}__{suffix}/
    -> {dest_base}/{rel_dir}/{timestamp}__{suffix}/
"""

import argparse
import os
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from move_files_common import (
    count_files,
    dedupe_latest_by_suffix,
    print_dry_run_summary,
    run_moves,
)

SCRATCH_OUT = os.environ.get(
    "CAFPYANA_GRID_OUT_DIR", "/pnfs/sbnd/scratch/users/brindenc/cafpyana_out"
)
PANDORA = "/exp/sbnd/data/users/brindenc/analyze_sbnd/numu/v10_06_00_validation/pandora"
VERSION = "v9"

# job_suffix -> path under pandora
SUFFIX_TO_REL = {
    "slimsyst_nocuts_norecomb_mcbnb_largepand_slimsyst_nocuts_norecomb": f"mc_syst/{VERSION}",
    "nosyst_nocuts_recomb_mcbnb_largepand_nosyst_nocuts_recomb": f"mc_syst/{VERSION}",
    "fullsyst_fullcuts_norecomb_mcbnb_largepand_fullsyst_cut_norecomb": f"mc_syst/{VERSION}",
    "nosyst_nocuts_recomb_mcbnb_tinypand_nosyst_nocuts_recomb2": f"mc_syst/{VERSION}",
    "nosyst_nocuts_recomb_detvar_pmtqe_nosyst_nocuts_norecomb_large": f"det_var/pds/{VERSION}",
    "nosyst_nocuts_recomb_detvar_pmtgain_nosyst_nocuts_norecomb_large": f"det_var/pds/{VERSION}",
    "nosyst_nocuts_recomb_detvar_pmtspe_nosyst_nocuts_norecomb_large": f"det_var/pds/{VERSION}",
    "nosyst_nocuts_recomb_detvar_nosce_nosyst_nocuts_norecomb_large": f"det_var/sce/{VERSION}",
    "nosyst_nocuts_recomb_detvar_twicesce_nosyst_nocuts_norecomb_large": f"det_var/sce/{VERSION}",
    "nosyst_nocuts_recomb_detvar_wiremodxtheta_nosyst_nocuts_norecomb_large": f"det_var/wiremod/{VERSION}",
    "nosyst_nocuts_recomb_detvar_wiremodyz_nosyst_nocuts_norecomb_large": f"det_var/wiremod/{VERSION}",
    "nosyst_nocuts_recomb_detvar_nominal_nosyst_nocuts_recomb_large": f"det_var/nominal/{VERSION}",
    "dataonbeam_nocuts_dataonbeam_dev": f"data/{VERSION}",
    "nosyst_nocuts_recomb_data_large": f"offbeam/{VERSION}",
    "nosyst_nocuts_recomb_mcoffbeam_largepand_nosyst_nocuts_recomb": f"offbeam/{VERSION}",
    "nosyst_nocuts_recomb_mclowe_largepand_nosyst_nocuts_recomb": f"mc_lowe/{VERSION}",
}

JOB_DIR_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{6}__(.+)$")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=f"{SCRATCH_OUT}/dfs")
    p.add_argument("--dest-base", default=PANDORA)
    p.add_argument("--only", default="", help="Substring filter on job dir name")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ncpu", type=int, default=4)
    args = p.parse_args()

    source = Path(args.source)
    dest_base = Path(args.dest_base)
    candidates: list[tuple[str, tuple[Path, Path, int, int]]] = []
    skipped: list[str] = []

    for job_dir in sorted(source.iterdir()):
        if not job_dir.is_dir():
            continue
        m = JOB_DIR_RE.match(job_dir.name)
        if not m:
            continue
        if args.only and args.only not in job_dir.name:
            continue
        suffix = m.group(1)
        rel = SUFFIX_TO_REL.get(suffix)
        if rel is None:
            skipped.append(f"no map: {job_dir.name}")
            continue
        n_all, n_df = count_files(job_dir)
        if n_all == 0:
            skipped.append(f"empty: {job_dir.name}")
            continue
        candidates.append(
            (suffix, (job_dir, dest_base / rel / job_dir.name, n_all, n_df))
        )

    jobs, dup_skipped = dedupe_latest_by_suffix(candidates)
    skipped.extend(dup_skipped)

    if skipped:
        print("Skipped:", file=sys.stderr)
        for line in skipped:
            print(f"  {line}", file=sys.stderr)

    if args.dry_run:
        print_dry_run_summary(jobs)
        return 0

    return run_moves(jobs, args.overwrite, args.ncpu)


if __name__ == "__main__":
    sys.exit(main())
