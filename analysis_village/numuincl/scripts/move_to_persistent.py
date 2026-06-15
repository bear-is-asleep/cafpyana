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
    JOB_DIR_RE,
    PANDORA_DATA,
    PANDORA_PERSISTENT,
    VERSION,
    Job,
    count_files,
    print_dry_run_summary,
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

    for job_dir in sorted(source_base.rglob("*")):
        if not job_dir.is_dir():
            continue
        if not JOB_DIR_RE.match(job_dir.name):
            continue
        if job_dir.parent.name != version:
            continue
        if only and only not in job_dir.name:
            continue
        n_all, n_df = count_files(job_dir)
        if n_all == 0:
            skipped.append(f"empty: {job_dir}")
            continue
        rel = job_dir.parent.relative_to(source_base)
        jobs.append((job_dir, dest_base / rel / job_dir.name, n_all, n_df))

    return jobs, skipped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=PANDORA_DATA)
    p.add_argument("--dest-base", default=PANDORA_PERSISTENT)
    p.add_argument("--only", default="", help="Substring filter on job dir name")
    p.add_argument(
        "--version",
        default=VERSION,
        help=f"Version subdir under each category (default: {VERSION})",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ncpu", type=int, default=4)
    args = p.parse_args()

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
        print_dry_run_summary(jobs)
        return 0

    return run_moves(jobs, args.overwrite, args.ncpu)


if __name__ == "__main__":
    sys.exit(main())
