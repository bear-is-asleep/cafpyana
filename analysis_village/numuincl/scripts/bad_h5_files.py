#!/usr/bin/env python3
"""
Find corrupt or unreadable HDF5 files under a directory.

Uses sbnd.general.utils.is_hdf5_file (magic-byte check) and h5py open/read.
Optionally writes bad_h5.list and/or moves bad files to trash.

With --version, scans files inside timestamp__suffix job dirs under pandora
(same layout as move_files / move_to_persistent).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py

NUMU_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (NUMU_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from move_files_common import PANDORA_DATA, iter_job_dirs
from sbnd.general.utils import is_hdf5_file, parallel_map_ordered

DEFAULT_SUFFIXES = ('.h5', '.hdf5', '.hdf', '.df')
TRASH_DIR = Path(PANDORA_DATA) / 'trash'


def collect_files(
    folder: Path,
    recursive: bool,
    suffixes: tuple[str, ...],
) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f'Not a directory: {folder}')
    suffixes = tuple(s.lower() for s in suffixes)
    iterator = folder.rglob('*') if recursive else folder.iterdir()
    paths = [
        p for p in iterator
        if p.is_file() and p.suffix.lower() in suffixes
    ]
    return sorted(paths)


def collect_job_dir_files(
    job_dirs: list[Path],
    suffixes: tuple[str, ...],
) -> list[Path]:
    suffixes = tuple(s.lower() for s in suffixes)
    paths: list[Path] = []
    for job_dir in job_dirs:
        paths.extend(
            p for p in job_dir.iterdir()
            if p.is_file() and p.suffix.lower() in suffixes
        )
    return sorted(paths)


def check_hdf5(path: str) -> tuple[str, str] | None:
    if not is_hdf5_file(path):
        return (path, 'missing or invalid HDF5 signature')
    try:
        with h5py.File(path, 'r') as f:
            list(f.keys())
    except Exception as exc:
        return (path, str(exc))
    return None


def _trash_destination(trash_dir: Path, src: Path) -> Path:
    dst = trash_dir / src.name
    if not dst.exists():
        return dst
    stem, suffix = src.stem, src.suffix
    n = 1
    while True:
        candidate = trash_dir / f'{stem}_{n}{suffix}'
        if not candidate.exists():
            return candidate
        n += 1


def move_bad_files(
    bad: list[tuple[str, str]], trash_dir: Path, *, quiet: bool,
) -> int:
    trash_dir.mkdir(parents=True, exist_ok=True)
    errors = 0
    for path, _ in bad:
        src = Path(path)
        dst = _trash_destination(trash_dir, src)
        try:
            shutil.move(str(src), str(dst))
            if not quiet:
                print(f'moved {src} -> {dst}')
        except OSError as exc:
            errors += 1
            print(f'ERROR moving {src}: {exc}', file=sys.stderr)
    return errors


def print_dry_run_summary(
    files: list[Path],
    folder: Path,
    *,
    version: str | None,
    job_dirs: list[Path] | None,
    suffixes: tuple[str, ...],
    list_out: Path | None,
    no_list: bool,
    remove: bool,
) -> None:
    n_df = sum(1 for f in files if f.suffix.lower() == '.df')
    print(f'Dry run: would check {len(files)} files ({n_df} .df)')
    print(f'  suffixes: {suffixes}')
    if version:
        print(f'  base: {folder}')
        print(f'  version: {version}, job dirs: {len(job_dirs or [])}')
        counts: dict[Path, int] = {}
        for f in files:
            counts[f.parent] = counts.get(f.parent, 0) + 1
        for job_dir in job_dirs or []:
            n = counts.get(job_dir, 0)
            print(f'    {job_dir.name}: {n} files')
    else:
        print(f'  folder: {folder}')
    if not no_list:
        list_path = (list_out or folder / 'bad_h5.list').resolve()
        print(f'  would write bad paths to {list_path}')
    if remove:
        print(f'  would move bad files to {TRASH_DIR}')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        'folder',
        type=Path,
        nargs='?',
        default=None,
        help=f'Directory to scan (default: {PANDORA_DATA} when --version is set)',
    )
    p.add_argument(
        '--version',
        default=None,
        metavar='VER',
        help='Scan job dirs under this version subdir (e.g. v10)',
    )
    p.add_argument(
        '--only',
        default='',
        help='Include job dirs whose name contains any substring (comma-separated, with --version)',
    )
    p.add_argument(
        '--suffix',
        action='append',
        default=[],
        help=f'File suffix to scan (repeatable). Default: {", ".join(DEFAULT_SUFFIXES)}',
    )
    p.add_argument('--no-recursive', action='store_true', help='Only scan top level')
    p.add_argument('--dry-run', action='store_true', help='Print scan plan only; no checks or writes')
    p.add_argument('--ncpu', type=int, default=8, help='Parallel workers (default: 1)')
    p.add_argument('--quiet', action='store_true', help='Only print bad file paths')
    p.add_argument(
        '--list-out',
        type=Path,
        default=None,
        help='Write bad paths here (default: <folder>/bad_h5.list)',
    )
    p.add_argument('--no-list', action='store_true', help='Do not write a .list file')
    p.add_argument(
        '--remove',
        action='store_true',
        help=f'Move bad files to {TRASH_DIR} (still writes bad_h5.list unless --no-list)',
    )
    args = p.parse_args()

    if args.folder is None:
        if args.version is None:
            p.error('folder is required unless --version is set')
        args.folder = Path(PANDORA_DATA)

    suffixes = tuple(args.suffix) if args.suffix else DEFAULT_SUFFIXES
    folder = args.folder.resolve()
    job_dirs: list[Path] | None = None
    if args.version:
        job_dirs = iter_job_dirs(folder, args.version, args.only)
        files = collect_job_dir_files(job_dirs, suffixes)
        scope = f' version={args.version}, {len(job_dirs)} job dirs'
    else:
        files = collect_files(folder, not args.no_recursive, suffixes)
        scope = ''
    if not files:
        print(f'No files with suffixes {suffixes} under {folder}{scope}')
        return 0

    if args.dry_run:
        if not args.quiet:
            print_dry_run_summary(
                files,
                folder,
                version=args.version,
                job_dirs=job_dirs,
                suffixes=suffixes,
                list_out=args.list_out,
                no_list=args.no_list,
                remove=args.remove,
            )
        else:
            print(len(files))
        return 0

    tasks = [str(f) for f in files]
    results = parallel_map_ordered(
        check_hdf5,
        tasks,
        ncpu=args.ncpu,
        show_progress=not args.quiet and len(tasks) > 1,
        desc='check_hdf5',
    )
    bad = [r for r in results if r is not None]

    if not args.no_list:
        list_path = (args.list_out or folder / 'bad_h5.list').resolve()
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text(
            '\n'.join(path for path, _ in bad) + ('\n' if bad else ''),
            encoding='utf-8',
        )
        if not args.quiet:
            print(f'Wrote {len(bad)} paths to {list_path}')

    for path, reason in bad:
        if args.quiet:
            print(path)
        else:
            print(f'{path}\n  {reason}')

    move_errors = 0
    if args.remove and bad:
        move_errors = move_bad_files(bad, TRASH_DIR, quiet=args.quiet)

    n_ok = len(files) - len(bad)
    if not args.quiet:
        print(f'\n{len(bad)} bad / {n_ok} ok / {len(files)} total')
        if args.remove and bad:
            moved = len(bad) - move_errors
            print(f'moved {moved} to {TRASH_DIR}, errors {move_errors}')
    return 1 if bad or move_errors else 0


if __name__ == '__main__':
    sys.exit(main())
