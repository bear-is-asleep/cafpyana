#!/usr/bin/env python3
"""
Load a subset of flatcaf ROOT files (XRootD), concat slice vertices, check duplicates.

Mirrors what run_df_maker does: many files → one table → duplicate vertices mean overlap.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import uproot

SCRIPT_DIR = Path(__file__).resolve().parent
NUMU_DIR = SCRIPT_DIR.parent
CAFPYANA_WD = NUMU_DIR.parents[1]

DEFAULT_LIST = (
    "/exp/sbnd/data/users/brindenc/analyze_sbnd/numu/v10_06_00_validation"
    "/pandora/mc_syst/mc_fullpand.list"
)

SLC_VERTEX_BRANCHES = [
    "rec.slc.vertex.x",
    "rec.slc.vertex.y",
    "rec.slc.vertex.z",
]

HOSTNAME = os.environ.get("HOSTNAME", "")
XROOTD_PREFIX = "root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr"


def _bootstrap_imports() -> None:
    for p in (str(CAFPYANA_WD), str(CAFPYANA_WD / "pyanalib")):
        if p not in sys.path:
            sys.path.insert(0, p)


def to_xrootd(path: str) -> str:
    if path.startswith("root://"):
        return path
    if path.startswith("/pnfs") and "gpvm" not in (HOSTNAME or ""):
        return path.replace("/pnfs", XROOTD_PREFIX, 1)
    if path.startswith("xroot"):
        return path[1:]
    return path


def open_flatcaf(path: str, *, attempts: int = 3, timeout: int = 300):
    url = to_xrootd(path)
    last_exc = None
    for k in range(attempts):
        try:
            return uproot.open(url, timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if k + 1 < attempts:
                time.sleep(2.0 * (k + 1))
    raise last_exc


def _col_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    for col in df.columns:
        leaf = col[-1] if isinstance(col, tuple) else col
        if leaf == name:
            return df[col]
    raise KeyError(f"column {name!r} not in {list(df.columns)[:8]}...")


def _slc_index_name(index: pd.Index) -> str:
    for name in index.names:
        if name and "slc" in name and "index" in name:
            return name
    if index.nlevels >= 2:
        return index.names[-1]
    raise ValueError(f"Cannot find slice index level in {index.names}")


def load_slice_vertices(f, file_idx: int, path: str) -> pd.DataFrame:
    """Slice index + vertex only (no hdr / RSE)."""
    from pyanalib.pandas_helpers import loadbranches

    if "recTree" not in f:
        raise ValueError(f"File missing recTree (keys: {list(f.keys())})")

    slc = loadbranches(f["recTree"], SLC_VERTEX_BRANCHES).rec.slc
    slc_idx_name = _slc_index_name(slc.index)

    return pd.DataFrame(
        {
            "file_idx": file_idx,
            "path": path,
            "entry": slc.index.get_level_values("entry"),
            "slc_index": slc.index.get_level_values(slc_idx_name),
            "vtx_x": _col_series(slc, "x").astype(float).round(6),
            "vtx_y": _col_series(slc, "y").astype(float).round(6),
            "vtx_z": _col_series(slc, "z").astype(float).round(6),
        }
    )


def load_slice_vertices_from_path(file_idx: int, path: str) -> pd.DataFrame:
    with open_flatcaf(path) as f:
        return load_slice_vertices(f, file_idx, path)


def read_input_list(list_path: str) -> list[str]:
    paths = []
    with open(list_path) as handle:
        for line in handle:
            if "#" in line:
                continue
            line = line.strip()
            if line:
                paths.append(line)
    return paths


def paths_for_grid_job(
    list_paths: list[str], job_id: int, ngrid: int
) -> list[tuple[int, str]]:
    """file_idx = __ntuple within job; list line = job_id + file_idx * ngrid."""
    out = []
    file_idx = 0
    line_idx = job_id
    while line_idx < len(list_paths):
        out.append((file_idx, list_paths[line_idx]))
        file_idx += 1
        line_idx += ngrid
    return out


def resolve_labeled_paths(args: argparse.Namespace) -> list[tuple[int, str]]:
    if args.file:
        return [(0, args.file)]

    if args.files:
        return list(enumerate(p.strip() for p in args.files.split(",") if p.strip()))

    list_paths = read_input_list(args.list)

    if args.job is not None:
        return paths_for_grid_job(list_paths, args.job, args.ngrid)

    start = args.start
    end = start + args.count if args.count is not None else len(list_paths)
    if args.max_files is not None:
        end = min(end, start + args.max_files)
    end = min(end, len(list_paths))
    return [(i - start, list_paths[i]) for i in range(start, end)]


def duplicate_vertex_report(df: pd.DataFrame) -> bool:
    """Print duplicate-vertex stats on combined dataframe. Returns True if any dupes."""
    n_raw = len(df)
    df = df.dropna(subset=["vtx_x", "vtx_y", "vtx_z"]).copy()
    n_valid = len(df)
    print(f"\n=== Combined slice vertices ===")
    print(f"  files loaded: {df['file_idx'].nunique():,}")
    print(f"  rows total: {n_raw:,}  after dropping NaN vtx: {n_valid:,}")

    if n_valid == 0:
        print("  No valid vertices to check.")
        return False

    df["vertex"] = list(zip(df["vtx_x"], df["vtx_y"], df["vtx_z"]))
    n_unique = df["vertex"].nunique()
    n_dup_rows = n_valid - n_unique
    print(f"  unique vertices: {n_unique:,}  duplicate rows: {n_dup_rows:,}")

    cross = (
        df.groupby("vertex", sort=False)
        .filter(lambda g: len(g) > 1)
        .copy()
    )
    cross_file = (
        cross.groupby("vertex", sort=False)
        .filter(lambda g: g["file_idx"].nunique() > 1)
    )
    n_cross_rows = len(cross_file)
    n_cross_vertices = cross_file["vertex"].nunique() if n_cross_rows else 0
    print(
        f"  rows with duplicate vertex (any file): {len(cross):,}  "
        f"({cross['vertex'].nunique():,} distinct vertices)"
    )
    print(
        f"  cross-file duplicate rows: {n_cross_rows:,}  "
        f"({n_cross_vertices:,} vertices in >1 file)"
    )

    if n_dup_rows == 0:
        print("  No duplicate vertices.")
        return False

    counts = df["vertex"].value_counts()
    print("  top duplicated vertices (count, n_files):")
    for vtx, cnt in counts[counts > 1].head(15).items():
        n_files = df.loc[df["vertex"] == vtx, "file_idx"].nunique()
        print(f"    {vtx}  x{cnt}  files={n_files}")
        if n_files > 1:
            sub = df.loc[df["vertex"] == vtx, ["file_idx", "entry", "slc_index"]]
            print(sub.drop_duplicates().head(6).to_string(index=False, header=False))

    return n_dup_rows > 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Load flatcaf subset via XRootD, concat slice vertices, report duplicates."
        ),
    )
    src = p.add_argument_group("input (pick one)")
    src.add_argument("--file", default="", help="Single flatcaf path")
    src.add_argument("--files", default="", help="Comma-separated flatcaf paths")
    src.add_argument("--list", default=DEFAULT_LIST, help="Input file list")
    src.add_argument(
        "--job",
        type=int,
        default=None,
        help="Grid job id: load all files job, job+ngrid, ... from --list",
    )
    src.add_argument("--ngrid", type=int, default=1000, help="ngrid from run_df_maker")
    src.add_argument("--start", type=int, default=0, help="First list line (0-based)")
    src.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of list lines from --start (default: rest of list)",
    )
    src.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap files taken from list slice (with --start/--count)",
    )
    return p


def main() -> int:
    _bootstrap_imports()
    args = build_parser().parse_args()

    labeled = resolve_labeled_paths(args)
    if not labeled:
        print("ERROR: no input files resolved", file=sys.stderr)
        return 2

    print(f"Loading {len(labeled)} file(s)...")
    t0 = time.monotonic()
    frames = []
    for file_idx, path in labeled:
        print(f"  [{file_idx}] {path}")
        frames.append(load_slice_vertices_from_path(file_idx, path))
    combined = pd.concat(frames, ignore_index=True)
    elapsed = time.monotonic() - t0
    print(f"Loaded in {elapsed:.1f}s ({elapsed / len(labeled):.1f}s/file)")

    duplicate_vertex_report(combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
