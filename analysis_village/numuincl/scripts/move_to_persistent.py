#!/usr/bin/env python3
"""
Move job dirs from data pandora to persistent pandora:
  {source}/{rel}/{timestamp}__{suffix}/
    -> {dest_base}/{rel}/{timestamp}__{suffix}/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from move_files_common import (
    PANDORA_DATA,
    PANDORA_PERSISTENT,
    VERSION,
    Job,
    count_files,
    iter_job_dirs,
    print_dry_run_summary,
    print_makeup_dry_run_summary,
    run_moves,
)


def collect_jobs(
    source_base: Path,
    dest_base: Path,
    only: str,
    version: str,
) -> tuple[list[Job], list[str]]:
    jobs: list[Job] = []
    skipped: list[str] = []

    for job_dir in iter_job_dirs(source_base, version, only):
        n_all, n_df, nbytes = count_files(job_dir)
        if n_all == 0:
            skipped.append(f"empty: {job_dir}")
            continue
        rel = job_dir.parent.relative_to(source_base)
        jobs.append((job_dir, dest_base / rel / job_dir.name, n_all, n_df, nbytes))

    return jobs, skipped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=PANDORA_DATA)
    p.add_argument("--dest-base", default=PANDORA_PERSISTENT)
    p.add_argument(
        "--only",
        default="",
        help="Include job dirs whose name contains any substring (comma-separated)",
    )
    p.add_argument(
        "--version",
        default=VERSION,
        help=f"Version subdir under each category (default: {VERSION})",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--makeup",
        action="store_true",
        help="Resume interrupted move: copy missing files only, keep source",
    )
    p.add_argument("--ncpu", type=int, default=4)
    args = p.parse_args()

    if args.makeup and args.overwrite:
        p.error("--makeup and --overwrite are mutually exclusive")

    source_base = Path(args.source)
    dest_base = Path(args.dest_base)
    if not source_base.is_dir():
        print(f"Not a directory: {source_base}", file=sys.stderr)
        return 1

    jobs, skipped = collect_jobs(source_base, dest_base, args.only, args.version)

    if skipped:
        print("Skipped:", file=sys.stderr)
        for line in skipped:
            print(f"  {line}", file=sys.stderr)

    if args.dry_run:
        if args.makeup:
            print_makeup_dry_run_summary(jobs)
        else:
            print_dry_run_summary(jobs, copy=False)
        return 0

    return run_moves(jobs, args.overwrite, args.ncpu, makeup=args.makeup, copy=False)


if __name__ == "__main__":
    sys.exit(main())
