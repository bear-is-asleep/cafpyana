#!/usr/bin/env python3
"""Compute chi2 summaries from a saved Systematics directory (fast path: saved total cov)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
CAFPYANA_WD = REPO_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(CAFPYANA_WD)):
    if p not in sys.path:
        sys.path.insert(0, p)

from chi2_from_caf import (
    DEFAULT_DATA_DIR,
    CafDataContext,
    cut_chain_for_syst_dir,
    compute_chi2_from_caf,
    load_caf_data,
)
from chi2_io import save_chi2_json_at
from detsys_config import DetsysConfig
from sbnd.stats.systematics import Systematics

CANONICAL_VARIABLES = [
    "costheta",
    "momentum",
    "differential",
    "bcfm",
    "flashpe",
    "startx",
    "starty",
    "startz",
    "endx",
    "endy",
    "endz",
    "phi",
    "theta",
    "length",
    "vtxx",
    "vtyy",
    "vtzx",
]

CHI2_IGNORE_KEYS = [
    "metadata_detsys",
    "metadata",
    "detsys",
    "geant4_syst",
    "geant4",
    "xsec",
    "flux",
    "g4",
    "pds",
    "sce",
    "tpc",
    "calo",
    "stat",
    "stat_flat",
    "stat_rw",
    "stat_lowe",
    "reinteractions",
    "GENIE",
    "Flux",
    "cosmic_data",
    "nt",
    "pot",
]


def _detect_metadata_dir(syst_path: Path) -> str:
    if (syst_path / "metadata_detsys").is_dir():
        return "metadata_detsys"
    if (syst_path / "metadata").is_dir():
        return "metadata"
    raise FileNotFoundError(
        f"No metadata_detsys/ or metadata/ under {syst_path}; "
        "expected a saved Systematics directory."
    )


def _discover_variables(syst_path: Path, metadata_dir: str) -> list[str]:
    meta_root = syst_path / metadata_dir
    suffix = "_sel_data.csv"
    found: list[str] = []
    for path in meta_root.glob(f"*{suffix}"):
        var_name = path.name[: -len(suffix)]
        if var_name:
            found.append(var_name)
    if not found:
        raise FileNotFoundError(f"No *{suffix} files under {meta_root}")
    ordered = [v for v in CANONICAL_VARIABLES if v in found]
    extras = sorted(v for v in found if v not in ordered)
    return ordered + extras


def _order_variables(variables: list[str]) -> list[str]:
    ordered = [v for v in CANONICAL_VARIABLES if v in variables]
    extras = sorted(v for v in variables if v not in ordered)
    return ordered + extras


def _compute_chi2_for_var_metadata(
    syst_path: Path,
    var_name: str,
    *,
    metadata_dir: str,
    ncpu: int,
) -> dict:
    use_legacy_names = metadata_dir == "metadata_detsys"
    syst = Systematics.from_saved(
        str(syst_path),
        var_name=var_name,
        metadata_dir=metadata_dir,
        ignore_keys=CHI2_IGNORE_KEYS,
        select_keys=["total"],
        lite=False,
        use_legacy_names=use_legacy_names,
        ncpus=ncpu,
    )
    if syst.sel_data is None:
        raise ValueError(f"{var_name}: missing sel_data; cannot compute chi2 without data histogram")
    if "total" not in syst.systematics:
        raise ValueError(
            f"{var_name}: missing 'total' summary; run analyze_detsys_universes or notebook merge first"
        )
    total = syst.systematics["total"]
    if total.get("event_cov") is None:
        raise ValueError(
            f"{var_name}: 'total' has no event_cov; run analyze_detsys_universes or notebook merge first"
        )
    return syst._calc_chi2(keys=["total"], include_summary=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute chi2.json from a saved Systematics directory using saved total covariance."
    )
    parser.add_argument(
        "--syst-path",
        required=True,
        help="Saved systematics directory (e.g. .../syst/cont or .../syst/full).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: {syst_path}/chi2.json).",
    )
    parser.add_argument(
        "--variables",
        default=None,
        help="Comma-separated variable names (default: auto-discover from metadata).",
    )
    parser.add_argument("--ncpu", type=int, default=1, help="Workers for loading saved systematics.")
    parser.add_argument(
        "--from-metadata",
        action="store_true",
        default=False,
        help="Use sel_data from saved metadata too (default: reload data from CAF).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="CAF data root when reloading data (default: inferred from syst path).",
    )
    parser.add_argument("--version", default=None, help="CAF file version tag (default: v8).")
    args = parser.parse_args()

    syst_path = Path(args.syst_path).resolve()
    if not syst_path.is_dir():
        print(f"Missing systematics directory: {syst_path}")
        return 1

    try:
        metadata_dir = _detect_metadata_dir(syst_path)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    if args.variables:
        variables = _order_variables([v.strip() for v in args.variables.split(",") if v.strip()])
    else:
        try:
            variables = _discover_variables(syst_path, metadata_dir)
        except FileNotFoundError as exc:
            print(exc)
            return 1

    data_ctx: CafDataContext | None = None
    if not args.from_metadata:
        if args.data_dir:
            data_dir = args.data_dir
        elif len(syst_path.parents) >= 4:
            data_dir = str(syst_path.parents[3])
        else:
            data_dir = DEFAULT_DATA_DIR
        version = args.version or DetsysConfig().version
        cut_chain = cut_chain_for_syst_dir(syst_path)
        data_ctx = load_caf_data(
            data_dir=data_dir,
            version=version,
            cut_chain=cut_chain,
            ncpu=args.ncpu,
        )

    chi2_map: dict[str, dict] = {}
    skipped: list[str] = []
    for var_name in tqdm(variables, desc="chi2", unit="var"):
        try:
            if args.from_metadata:
                chi2_map[var_name] = _compute_chi2_for_var_metadata(
                    syst_path,
                    var_name,
                    metadata_dir=metadata_dir,
                    ncpu=args.ncpu,
                )
            else:
                assert data_ctx is not None
                chi2_map[var_name] = compute_chi2_from_caf(
                    syst_path,
                    var_name,
                    data_ctx,
                    metadata_dir=metadata_dir,
                )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"Skip {var_name}: {exc}")
            skipped.append(var_name)

    if not chi2_map:
        print("No chi2 values computed.")
        if skipped:
            print(f"Skipped variables: {', '.join(skipped)}")
        return 1

    out_path = Path(args.output) if args.output else syst_path / "chi2.json"
    written = save_chi2_json_at(out_path, chi2_map)
    print(f"Chi2 summary for {syst_path} (metadata_dir={metadata_dir}):")
    for var_name in variables:
        if var_name not in chi2_map:
            continue
        entry = chi2_map[var_name]
        print(f" - {var_name}: chi2={entry['chi2']:.4f}, dof={entry['dof']}, p={entry['pvalue']:.4g}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    if written:
        print(f"Wrote chi2 to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
