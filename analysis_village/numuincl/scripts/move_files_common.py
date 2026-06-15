"""Shared helpers for pandora job-dir move scripts."""

from __future__ import annotations

import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PANDORA_DATA = (
    "/exp/sbnd/data/users/brindenc/analyze_sbnd/numu/v10_06_00_validation/pandora"
)
PANDORA_PERSISTENT = (
    "/pnfs/sbnd/persistent/users/brindenc/analyze_sbnd/numu/v10_06_00_validation/pandora"
)
VERSION = "v8"

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

Job = tuple[Path, Path, int, int]


def job_timestamp(name: str) -> str:
    return name.split("__", 1)[0]


def dedupe_latest_by_suffix(
    candidates: list[tuple[str, Job]],
) -> tuple[list[Job], list[str]]:
    """When several scratch dirs share a suffix, keep only the newest."""
    best: dict[str, tuple[str, Job]] = {}
    skipped: list[str] = []
    for suffix, job in candidates:
        ts = job_timestamp(job[0].name)
        if suffix not in best:
            best[suffix] = (ts, job)
            continue
        old_ts, old_job = best[suffix]
        if ts > old_ts:
            skipped.append(f"older duplicate: {old_job[0].name}")
            best[suffix] = (ts, job)
        else:
            skipped.append(f"older duplicate: {job[0].name}")
    return [job for _, job in best.values()], skipped


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


def print_dry_run_summary(jobs: list[Job]) -> None:
    total_all = sum(n for _, _, n, _ in jobs)
    total_df = sum(n for _, _, _, n in jobs)
    print(f"Dry run: {len(jobs)} job dirs, {total_all} files ({total_df} .df)")
    for src, dest, n_all, n_df in jobs:
        print(f"\n  {dest}")
        print(f"    {n_all} files ({n_df} .df) <- {src}")


def run_moves(jobs: list[Job], overwrite: bool, ncpu: int) -> int:
    if not jobs:
        print("Nothing to move.")
        return 0

    work = [(str(s), str(d), overwrite) for s, d, _, _ in jobs]
    ncpu = max(1, min(ncpu, len(work)))
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
