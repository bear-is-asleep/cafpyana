#!/usr/bin/env python3
"""
Move grid job dirs from scratch to pandora:
  {source}/{timestamp}__{suffix}/
    -> {dest_base}/{rel_dir}/{timestamp}__{suffix}/
"""

import argparse
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRATCH_OUT = os.environ.get(
    "CAFPYANA_GRID_OUT_DIR", "/pnfs/sbnd/scratch/users/brindenc/cafpyana_out"
)
PANDORA = "/exp/sbnd/data/users/brindenc/analyze_sbnd/numu/v10_06_00_validation/pandora"

# job_suffix -> path under pandora
SUFFIX_TO_REL = {
    "slimsyst_nocuts_norecomb_mcbnb_largepand_slimsyst_nocuts_norecomb": "mc_syst/v8",
    "nosyst_nocuts_recomb_mcbnb_largepand_nosyst_nocuts_recomb": "mc_syst/v8",
    "fullsyst_fullcuts_norecomb_mcbnb_largepand_fullsyst_cut_norecomb": "mc_syst/v8",
    "nosyst_nocuts_recomb_mcbnb_tinypand_nosyst_nocuts_recomb2": "mc_syst/v8",
    "nosyst_nocuts_recomb_detvar_pmtqe_nosyst_nocuts_norecomb_large": "det_var/pds/v8",
    "nosyst_nocuts_recomb_detvar_pmtgain_nosyst_nocuts_norecomb_large": "det_var/pds/v8",
    "nosyst_nocuts_recomb_detvar_pmtspe_nosyst_nocuts_norecomb_large": "det_var/pds/v8",
    "nosyst_nocuts_recomb_detvar_nosce_nosyst_nocuts_norecomb_large": "det_var/sce/v8",
    "nosyst_nocuts_recomb_detvar_twicesce_nosyst_nocuts_norecomb_large": "det_var/sce/v8",
    "nosyst_nocuts_recomb_detvar_wiremodxtheta_nosyst_nocuts_norecomb_large": "det_var/wiremod/v8",
    "nosyst_nocuts_recomb_detvar_wiremodyz_nosyst_nocuts_norecomb_large": "det_var/wiremod/v8",
    "nosyst_nocuts_recomb_detvar_nominal_nosyst_nocuts_recomb_large": "det_var/nominal/v8",
    "dataonbeam_nocuts_dataonbeam_dev": "data/v8",
    "nosyst_nocuts_recomb_data_large": "offbeam/v8",
    "nosyst_nocuts_recomb_mcoffbeam_largepand_nosyst_nocuts_recomb": "offbeam/v8",
    "nosyst_nocuts_recomb_mclowe_largepand_nosyst_nocuts_recomb": "mc_lowe/v8",
}

JOB_DIR_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{6}__(.+)$")


def count_files(path: Path) -> tuple[int, int]:
    n_all = sum(1 for p in path.iterdir() if p.is_file())
    n_df = sum(1 for p in path.glob("*.df"))
    return n_all, n_df


def move_folder(src: str, dest: str, overwrite: bool) -> str:
    s, d = Path(src), Path(dest)
    if d.exists():
        if not overwrite:
            return "skipped"
        shutil.rmtree(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    return "moved"


def print_dry_run_summary(jobs: list[tuple[Path, Path, int, int]]) -> None:
    total_all = sum(n for _, _, n, _ in jobs)
    total_df = sum(n for _, _, _, n in jobs)
    print(f"Dry run: {len(jobs)} job dirs, {total_all} files ({total_df} .df)")
    for src, dest, n_all, n_df in jobs:
        print(f"\n  {dest}")
        print(f"    {n_all} files ({n_df} .df) <- {src}")


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
    jobs: list[tuple[Path, Path, int, int]] = []
    skipped: list[str] = []

    for job_dir in sorted(source.iterdir()):
        if not job_dir.is_dir():
            continue
        m = JOB_DIR_RE.match(job_dir.name)
        if not m:
            continue
        if args.only and args.only not in job_dir.name:
            continue
        rel = SUFFIX_TO_REL.get(m.group(1))
        if rel is None:
            skipped.append(f"no map: {job_dir.name}")
            continue
        n_all, n_df = count_files(job_dir)
        if n_all == 0:
            skipped.append(f"empty: {job_dir.name}")
            continue
        jobs.append((job_dir, dest_base / rel / job_dir.name, n_all, n_df))

    if skipped:
        print("Skipped:", file=sys.stderr)
        for line in skipped:
            print(f"  {line}", file=sys.stderr)

    if args.dry_run:
        print_dry_run_summary(jobs)
        return 0

    if not jobs:
        print("Nothing to move.")
        return 0

    work = [(str(s), str(d), args.overwrite) for s, d, _, _ in jobs]
    ncpu = max(1, min(args.ncpu, len(work)))
    moved = skipped_n = errors = 0
    with ProcessPoolExecutor(max_workers=ncpu) as ex:
        futs = {ex.submit(move_folder, *t): t for t in work}
        for fut in as_completed(futs):
            try:
                if fut.result() == "moved":
                    moved += 1
                else:
                    skipped_n += 1
            except Exception as exc:
                errors += 1
                print(f"ERROR {futs[fut][0]}: {exc}", file=sys.stderr)

    total_all = sum(n for _, _, n, _ in jobs)
    print(f"moved {moved} dirs ({total_all} files), skipped={skipped_n}, errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
