#!/usr/bin/env python3
"""Plot chi2 summaries written by analyze_detsys_universes.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
CAFPYANA_WD = REPO_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(CAFPYANA_WD)):
    if p not in sys.path:
        sys.path.insert(0, p)

SBNDANA_DIR = "/exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/sbnd"
sys.path.insert(0, SBNDANA_DIR)
sys.path.insert(0, f"{SBNDANA_DIR.replace('/numuincl/sbnd', '/numuincl')}")
plt.style.use(f"{SBNDANA_DIR}/plotlibrary/numu2025.mplstyle")

from chi2_from_caf import (
    cut_chain_for_syst_dir,
    compute_chi2_from_caf,
    load_caf_data,
)
from detsys_config import DETSYS_CUTS_ALL, DetsysConfig
from naming import INTERNAL_LABEL
from sbnd.general import plotters
from sbnd.general.utils import get_scientific_str

DEFAULT_DATA_DIR = DetsysConfig().data_dir

VARIABLES = [
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


def _save_root(data_dir: str, checkpoint: str) -> Path:
    return Path(data_dir) / "data" / checkpoint / "syst"


def _plot_root(data_dir: str, checkpoint_out: str) -> Path:
    return Path(data_dir) / "plots" / checkpoint_out / "syst"


def _ordered_variables(chi2_map: dict[str, dict]) -> list[str]:
    ordered = [v for v in VARIABLES if v in chi2_map]
    extras = sorted(k for k in chi2_map if k not in ordered)
    return ordered + extras


def _discover_cuts(save_root: Path) -> list[str]:
    cuts = [p.name for p in save_root.iterdir() if p.is_dir() and (p / "chi2.json").is_file()]
    cut_order = {cut: idx for idx, cut in enumerate(DETSYS_CUTS_ALL)}
    return sorted(cuts, key=lambda c: (cut_order.get(c, len(cut_order)), c))


def _format_pval_label(pval: float) -> str:
    if pval < 0.01:
        inner = get_scientific_str(pval).strip("$")
        return rf"$p={inner}$"
    return f"p={pval:.3f}"


def _detect_metadata_dir(syst_path: Path) -> str:
    if (syst_path / "metadata_detsys").is_dir():
        return "metadata_detsys"
    if (syst_path / "metadata").is_dir():
        return "metadata"
    raise FileNotFoundError(f"No metadata under {syst_path}")


def _compute_chi2_map_from_caf(
    syst_path: Path,
    *,
    data_dir: str,
    version: str | None,
    ncpu: int,
) -> dict[str, dict]:
    metadata_dir = _detect_metadata_dir(syst_path)
    cut_chain = cut_chain_for_syst_dir(syst_path)
    ctx = load_caf_data(
        data_dir=data_dir,
        version=version or DetsysConfig().version,
        cut_chain=cut_chain,
        ncpu=ncpu,
    )
    chi2_map: dict[str, dict] = {}
    for var_name in VARIABLES:
        try:
            chi2_map[var_name] = compute_chi2_from_caf(
                syst_path,
                var_name,
                ctx,
                metadata_dir=metadata_dir,
            )
        except (FileNotFoundError, KeyError) as exc:
            print(f"Skip {var_name}: {exc}")
    return chi2_map


def _load_chi2_json(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}, got {type(payload).__name__}")
    return payload


def _plot_chi2_summary(
    chi2_map: dict[str, dict],
    *,
    cut: str,
    checkpoint: str,
    plot_dir: Path,
) -> str:
    variables = _ordered_variables(chi2_map)
    chi2_vals = np.array([float(chi2_map[v]["chi2"]) for v in variables], dtype=float)
    dof_vals = np.array([int(chi2_map[v]["dof"]) for v in variables], dtype=float)
    pvals = np.array([float(chi2_map[v]["pvalue"]) for v in variables], dtype=float)
    chi2_per_dof = np.divide(chi2_vals, dof_vals, out=np.zeros_like(chi2_vals), where=dof_vals > 0)

    x = np.arange(len(variables))
    colors = np.where(pvals < 0.05, "#c0392b", "#2980b9")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    bars = ax.bar(x, chi2_per_dof, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", ls="--", lw=1.0, label=r"$\chi^2/\nu = 1$")
    ax.set_ylabel(r"$\chi^2/\nu$")
    ax.set_xlabel("Variable")
    ax.set_title(f"Data vs MC systematic covariance ({cut} cut, {checkpoint})")
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    y_top = max(float(np.max(chi2_per_dof)), 1.0)
    label_pad = max(0.2 * y_top, 0.35)
    ax.set_ylim(0.0, y_top + 2.2 * label_pad)

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.04 * label_pad,
            _format_pval_label(float(pvals[idx])),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=30,
        )

    plotters.add_label(ax, INTERNAL_LABEL, where=(0.01, 1.01), fontsize=8, color="black")
    fig.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    savename = f"chi2_summary_{cut}cut"
    plotters.save_plot(savename, fig=fig, folder_name=str(plot_dir))
    plt.close(fig)
    return str(plot_dir / f"{savename}.png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot chi2.json summaries from analyze_detsys_universes.py."
    )
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint tag (same as --day).")
    parser.add_argument(
        "--checkpoint-out",
        default=None,
        help="Output checkpoint tag for plots (default: same as --checkpoint).",
    )
    parser.add_argument("--cut", default=None, help="Single cut to plot (default: all cuts with chi2.json).")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Base data directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--syst-path",
        default=None,
        help="Saved syst directory (e.g. .../syst/full). Overrides --checkpoint/--cut.",
    )
    parser.add_argument(
        "--from-json",
        action="store_true",
        default=False,
        help="Read chi2.json instead of recomputing from CAF + saved cov.",
    )
    parser.add_argument("--version", default=None, help="CAF version tag when recomputing (default: v8).")
    parser.add_argument("--ncpu", type=int, default=1, help="Workers for data CAF load.")
    args = parser.parse_args()

    checkpoint_out = args.checkpoint_out or args.checkpoint

    if args.syst_path:
        syst_jobs = [(Path(args.syst_path).resolve(), args.syst_path.rstrip("/").split("/")[-1])]
    else:
        save_root = _save_root(args.data_dir, args.checkpoint)
        if not save_root.is_dir():
            print(f"Missing save directory: {save_root}")
            return 1
        cuts = [args.cut] if args.cut else _discover_cuts(save_root)
        if not cuts:
            print(f"No chi2.json files found under {save_root}")
            return 1
        syst_jobs = [(save_root / cut, cut) for cut in cuts]

    wrote: list[str] = []
    for syst_path, cut in syst_jobs:
        if not syst_path.is_dir():
            print(f"Skip cut {cut}: missing {syst_path}")
            continue

        if args.from_json:
            chi2_path = syst_path / "chi2.json"
            if not chi2_path.is_file():
                print(f"Skip cut {cut}: missing {chi2_path}")
                continue
            chi2_map = _load_chi2_json(chi2_path)
        else:
            chi2_map = _compute_chi2_map_from_caf(
                syst_path,
                data_dir=args.data_dir,
                version=args.version,
                ncpu=args.ncpu,
            )
            if not chi2_map:
                print(f"Skip cut {cut}: no chi2 values computed")
                continue

        plot_dir = _plot_root(args.data_dir, checkpoint_out) / cut / "chi2"
        out_path = _plot_chi2_summary(
            chi2_map,
            cut=cut,
            checkpoint=args.checkpoint,
            plot_dir=plot_dir,
        )
        wrote.append(out_path)
        print(f"Cut {cut}:")
        for var_name in _ordered_variables(chi2_map):
            entry = chi2_map[var_name]
            print(
                f" - {var_name}: chi2={entry['chi2']:.4f}, "
                f"dof={entry['dof']}, p={entry['pvalue']:.4g}"
            )
        print(f"Wrote plot to {out_path}")

    if not wrote:
        print("No plots written.")
        return 1

    print(f"Done. Wrote {len(wrote)} plot(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
