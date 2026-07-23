"""Shared helpers for pandora job-dir move scripts."""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, RLock

from tqdm import tqdm

_TQDM_LOCK = RLock()

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
    "mc_datadriven_v4_mcbnb_largepand_datadriven_v4": f"dent/datadriven_v4/{VERSION}",
    "dent_reprocess_data_dent_reprocess_data": f"dent/dent_reprocess_data/{VERSION}",
}

JOB_DIR_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{6}__(.+)$")

Job = tuple[Path, Path, int, int, int]


def format_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    size = float(nbytes)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} PiB"


def parse_substrings(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def name_matches_only(name: str, only: str) -> bool:
    patterns = parse_substrings(only)
    if not patterns:
        return True
    return any(p in name for p in patterns)


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


def count_files(path: Path) -> tuple[int, int, int]:
    files = [p for p in path.iterdir() if p.is_file()]
    n_all = len(files)
    n_df = sum(1 for p in files if p.suffix == ".df")
    nbytes = sum(p.stat().st_size for p in files)
    return n_all, n_df, nbytes


def iter_job_dirs(
    source_base: Path,
    version: str,
    only: str = "",
) -> list[Path]:
    job_dirs: list[Path] = []
    for job_dir in sorted(source_base.rglob("*")):
        if not job_dir.is_dir():
            continue
        if not JOB_DIR_RE.match(job_dir.name):
            continue
        if job_dir.parent.name != version:
            continue
        if not name_matches_only(job_dir.name, only):
            continue
        job_dirs.append(job_dir)
    return job_dirs


def _clear_dest(d: Path) -> None:
    """Remove dest job dir; rename aside first if NFS rmtree fails."""
    if not d.exists():
        return
    try:
        shutil.rmtree(d)
        return
    except OSError:
        pass
    bak = d.with_name(f"{d.name}.remove.{os.getpid()}")
    if bak.exists():
        shutil.rmtree(bak, ignore_errors=True)
    d.rename(bak)
    shutil.rmtree(bak, ignore_errors=True)


def _list_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file())


def _copy_files(
    files: list[Path],
    dest: Path,
    desc: str,
    slot: int | None,
) -> None:
    if slot is None:
        for p in files:
            shutil.copy2(p, dest / p.name)
        return
    with tqdm(
        total=len(files),
        desc=desc[:60],
        position=slot,
        leave=False,
        mininterval=0.2,
        file=sys.stderr,
    ) as pbar:
        for p in files:
            shutil.copy2(p, dest / p.name)
            pbar.update(1)


def copy_folder(
    src: str,
    dest: str,
    overwrite: bool,
    slot: int | None = None,
    show_progress: bool = False,
) -> str:
    s, d = Path(src), Path(dest)
    if d.exists():
        if not overwrite:
            return "skipped"
        _clear_dest(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.mkdir(parents=True, exist_ok=True)
    files = _list_files(s)
    _copy_files(files, d, s.name, slot if show_progress else None)
    return "copied"


def move_folder(
    src: str,
    dest: str,
    overwrite: bool,
    slot: int | None = None,
    show_progress: bool = False,
) -> str:
    s, d = Path(src), Path(dest)
    if d.exists():
        if not overwrite:
            return "skipped"
        _clear_dest(d)
    if show_progress and slot is not None:
        d.parent.mkdir(parents=True, exist_ok=True)
        d.mkdir(parents=True, exist_ok=True)
        files = _list_files(s)
        _copy_files(files, d, s.name, slot)
        shutil.rmtree(s)
        return "moved"
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    return "moved"


def makeup_plan(src: Path, dest: Path) -> tuple[str, int]:
    """Dry-run action and file count: skip, copy-all, fill, warn-extra."""
    n_src, _, _ = count_files(src)
    if n_src == 0:
        return "skip", 0
    if not dest.exists():
        return "copy-all", n_src
    n_dest, _, _ = count_files(dest)
    if n_dest == n_src:
        return "skip", 0
    if n_dest > n_src:
        return "warn-extra", 0
    dest_names = {p.name for p in dest.iterdir() if p.is_file()}
    missing = sum(
        1 for p in src.iterdir() if p.is_file() and p.name not in dest_names
    )
    return "fill", missing


def makeup_folder(
    src: str,
    dest: str,
    slot: int | None = None,
    show_progress: bool = False,
) -> str:
    s, d = Path(src), Path(dest)
    n_src, _, _ = count_files(s)
    if n_src == 0:
        return "skipped"
    d.parent.mkdir(parents=True, exist_ok=True)
    if not d.exists():
        d.mkdir()
    else:
        n_dest, _, _ = count_files(d)
        if n_dest == n_src:
            return "skipped"
        if n_dest > n_src:
            print(f"WARN dest has more files than src: {d}", file=sys.stderr)
            return "skipped"
    dest_names = {p.name for p in d.iterdir() if p.is_file()}
    missing = [p for p in _list_files(s) if p.name not in dest_names]
    if not missing:
        return "skipped"
    _copy_files(missing, d, s.name, slot if show_progress else None)
    return "madeup"


def print_dry_run_summary(jobs: list[Job], *, copy: bool = True) -> None:
    total_all = sum(n for _, _, n, _, _ in jobs)
    total_df = sum(n for _, _, _, n, _ in jobs)
    total_bytes = sum(n for _, _, _, _, n in jobs)
    verb = "copy" if copy else "move"
    print(
        f"Dry run: {len(jobs)} job dirs, {total_all} files "
        f"({total_df} .df), {format_size(total_bytes)} ({verb})"
    )
    for src, dest, n_all, n_df, nbytes in jobs:
        print(f"\n  {dest}")
        print(
            f"    {n_all} files ({n_df} .df), {format_size(nbytes)} <- {src}"
        )


def missing_file_bytes(src: Path, dest: Path) -> int:
    dest_names = {p.name for p in dest.iterdir() if p.is_file()} if dest.exists() else set()
    return sum(
        p.stat().st_size
        for p in src.iterdir()
        if p.is_file() and p.name not in dest_names
    )


def print_makeup_dry_run_summary(jobs: list[Job]) -> None:
    actions: dict[str, int] = {}
    total_bytes = 0
    for src, dest, _, _, nbytes in jobs:
        action, n = makeup_plan(src, dest)
        actions[action] = actions.get(action, 0) + 1
        if action == "copy-all":
            xfer_bytes = nbytes
            total_bytes += xfer_bytes
            detail = f"copy all {n} files, {format_size(xfer_bytes)}"
        elif action == "fill":
            xfer_bytes = missing_file_bytes(src, dest)
            total_bytes += xfer_bytes
            detail = f"fill {n} missing files (~{format_size(xfer_bytes)})"
        elif action == "skip":
            detail = "complete"
        else:
            detail = f"dest has extra files (src={src})"
        print(f"\n  {dest}")
        print(f"    {action}: {detail} <- {src}")
    print(
        f"\nMakeup dry run: {len(jobs)} job dirs "
        f"(skip={actions.get('skip', 0)}, "
        f"copy-all={actions.get('copy-all', 0)}, "
        f"fill={actions.get('fill', 0)}, "
        f"warn-extra={actions.get('warn-extra', 0)}), "
        f"~{format_size(total_bytes)} to transfer"
    )


def run_moves(
    jobs: list[Job],
    overwrite: bool,
    ncpu: int,
    makeup: bool = False,
    copy: bool = True,
    show_progress: bool = True,
) -> int:
    if not jobs:
        print("Nothing to move.")
        return 0

    if makeup:
        mode = "makeup"
    elif copy:
        mode = "copy"
    else:
        mode = "move"

    ncpu = max(1, min(ncpu, len(jobs)))
    done = skipped_n = errors = 0
    slots = list(range(1, ncpu + 1))
    slot_lock = Lock()

    def borrow_slot() -> int | None:
        if not show_progress:
            return None
        while True:
            with slot_lock:
                if slots:
                    return slots.pop(0)
            time.sleep(0.01)

    def release_slot(slot: int | None) -> None:
        if slot is None:
            return
        with slot_lock:
            slots.append(slot)

    def run_one(s: Path, d: Path) -> str:
        slot = borrow_slot()
        try:
            if mode == "makeup":
                return makeup_folder(str(s), str(d), slot, show_progress)
            if mode == "copy":
                return copy_folder(str(s), str(d), overwrite, slot, show_progress)
            return move_folder(str(s), str(d), overwrite, slot, show_progress)
        finally:
            release_slot(slot)

    global_bar = None
    if show_progress:
        tqdm.set_lock(_TQDM_LOCK)
        global_bar = tqdm(
            total=len(jobs),
            desc=f"{mode} dirs",
            position=0,
            unit="dir",
            file=sys.stderr,
            leave=True,
        )

    with ThreadPoolExecutor(max_workers=ncpu) as ex:
        futs = {
            ex.submit(run_one, s, d): (s, d, n_all, nbytes)
            for s, d, n_all, _, nbytes in jobs
        }
        for fut in as_completed(futs):
            src, dest, n_all, nbytes = futs[fut]
            try:
                result = fut.result()
                if result in ("moved", "copied", "madeup"):
                    done += 1
                else:
                    skipped_n += 1
                if show_progress:
                    if global_bar is not None:
                        global_bar.update(1)
                    tqdm.write(
                        f"done {result}: {dest.name} "
                        f"({n_all} files, {format_size(nbytes)}) -> {dest.parent}"
                    )
            except Exception as exc:
                errors += 1
                if global_bar is not None:
                    global_bar.update(1)
                msg = f"ERROR {src}: {exc}"
                if show_progress:
                    tqdm.write(msg)
                else:
                    print(msg, file=sys.stderr)

    if global_bar is not None:
        global_bar.close()

    if makeup:
        verb = "madeup"
    elif copy:
        verb = "copied"
    else:
        verb = "moved"
    total_all = sum(n for _, _, n, _, _ in jobs)
    total_bytes = sum(n for _, _, _, _, n in jobs)
    print(
        f"{verb} {done} dirs ({total_all} files, {format_size(total_bytes)}), "
        f"skipped={skipped_n}, errors={errors}"
    )
    return 1 if errors else 0
