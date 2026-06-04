#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
CAFPYANA_WD = REPO_ROOT.parents[1]
for p in (str(REPO_ROOT), str(CAFPYANA_WD)):
    if p not in sys.path:
        sys.path.insert(0, p)

#My imports 
SBNDANA_DIR = '/exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/sbnd'
sys.path.insert(0,SBNDANA_DIR)
sys.path.insert(0,f'{SBNDANA_DIR.replace("/numuincl/sbnd","/numuincl")}')
plt.style.use(f'{SBNDANA_DIR}/plotlibrary/numu2025.mplstyle')

from detsys_config import DET_VARS_ALL, DetsysConfig, build_config
from naming import INTERNAL_LABEL
from sbnd.general import plotters
from sbnd.stats.systematics import Systematics

# Stale combined buckets written by old analyze/notebook saves; must not reload from disk or sys_dict.
SUMMARY_BUCKET_KEYS = frozenset({"total", "pds", "sce", "tpc", "calo"})
IGNORE_KEYS = frozenset({"metadata_detsys", "geant4_syst", "detsys"})
LOAD_IGNORE_KEYS = IGNORE_KEYS | SUMMARY_BUCKET_KEYS
AGGREGATE_SUMMARY_KEYS = frozenset({"total", "pds", "sce", "tpc", "calo"})
SLIM_KEYS = ("xsec", "flux", "g4")
RESTORE_SUMMARY_KEYS = frozenset({*SLIM_KEYS, "cosmic"})


def _drop_stale_summaries(s: Systematics) -> None:
    """Drop combined summary buckets and CV stash keys, not slim/cosmic universes."""
    for k in list(s.systematics.keys()):
        v = s.systematics[k]
        if k in IGNORE_KEYS:
            s.systematics.pop(k, None)
            continue
        if v.get("variation") == "self" and k != "cosmic_data":
            s.systematics.pop(k, None)
            continue
        if v.get("variation") == "summary" and k in AGGREGATE_SUMMARY_KEYS:
            s.systematics.pop(k, None)


def _restore_slim_cosmic_keys(s: Systematics) -> None:
    """Re-enable keys corrupted by an earlier combine_summaries save."""
    for k in RESTORE_SUMMARY_KEYS:
        if k not in s.systematics:
            continue
        d = s.systematics[k]
        if d.get("variation") == "summary":
            d["variation"] = d.get("type") or "RW"


def _has_det_systematics(s: Systematics) -> bool:
    return any(k in s.systematics for k in DET_VARS_ALL)


def _flat_keys(cut: str, cfg: DetsysConfig) -> list[str]:
    keys = ["nt", "stat_flat"]
    if cut == cfg.cuts[-1]:
        return ["pot", *keys]
    return keys


def _add_flat_systematics(s: Systematics, flat_keys: list[str]) -> None:
    if "nt" in flat_keys:
        s.process_flat_systematic("nt", 0.01)
    if "stat_flat" in flat_keys:
        s.process_stat_systematics("stat_flat")
    if "pot" in flat_keys:
        s.process_flat_systematic("pot", 0.02)


def _summary_groups(s: Systematics, flat_keys: list[str]) -> tuple[list[str], list[str]]:
    """
    Match detsys.ipynb step 7: combine_keys only has total + det type groups;
    summary_keys adds flat, slim, cosmic for plots.
    """
    keys_present = set(s.systematics.keys())
    types_present = {sdict.get("type") for sdict in s.systematics.values()}
    types_present.discard(None)

    combine_keys: list[str] = ["total"]
    summary_keys: list[str] = ["total"]

    if _has_det_systematics(s):
        summary_keys.extend(k for k in flat_keys if k not in summary_keys)

    for k in SLIM_KEYS:
        if k in keys_present:
            summary_keys.append(k)

    if "cosmic" in keys_present:
        summary_keys.append("cosmic")

    if "calo" in types_present or "calo" in keys_present:
        summary_keys.append("calo")
        combine_keys.append("calo")
    if "pds" in types_present:
        summary_keys.append("pds")
        combine_keys.append("pds")
    if "tpc" in types_present:
        summary_keys.append("tpc")
        combine_keys.append("tpc")
    if "sce" in types_present:
        summary_keys.append("sce")
        combine_keys.append("sce")

    return combine_keys, summary_keys


def _proc_keys(s: Systematics) -> list[str]:
    keys: list[str] = []
    for k, sys_dict in s.systematics.items():
        if k in IGNORE_KEYS:
            continue
        if sys_dict.get("variation") == "self":
            continue
        if sys_dict.get("variation") == "summary":
            continue
        keys.append(k)
    return keys


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute covariances and plots from saved chunked universes.")
    parser.add_argument("--cut", default=None, help="Single cut to process (default: all cuts).")
    parser.add_argument("--day", required=True, help="Output day/tag.")
    parser.add_argument("--skip-plots", action="store_true", default=False, help="Skip plotting stage.")
    parser.add_argument(
        "--plot-universes",
        action="store_true",
        default=True,
        help="Plot individual universe distributions (sel and sigma_tilde where available).",
    )
    parser.add_argument(
        "--plot-covariance-matrices",
        action="store_true",
        default=False,
        help="Plot covariance/correlation matrices under twod/.",
    )
    parser.add_argument("--ncpu", type=int, default=1, help="Workers for loading saved systematics (use 1 on small nodes).")
    args = parser.parse_args()

    cfg = build_config(day=args.day, ncpu=args.ncpu)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.plot_dir).mkdir(parents=True, exist_ok=True)
    cuts = [args.cut] if args.cut else cfg.cuts

    for cut in cuts:
        universe_dir = f"{cfg.save_dir}/{cut}"
        save_dir = universe_dir
        plot_dir = f"{cfg.plot_dir}/{cut}/fracunc"
        universe_plot_dir = f"{cfg.plot_dir}/{cut}/dist"
        cov_plot_dir = f"{cfg.plot_dir}/{cut}/twod"
        if not Path(universe_dir).is_dir():
            print(f"Skip cut {cut}: missing directory {universe_dir}")
            continue
        systems = []
        print(f"Loading universes from {universe_dir}")
        for var_name in tqdm(VARIABLES, desc=f"{cut} load", unit="var"):
            sys_obj = Systematics.from_saved(
                universe_dir,
                var_name=var_name,
                metadata_dir="metadata_detsys",
                ignore_keys=list(LOAD_IGNORE_KEYS),
                ncpus=cfg.ncpu,
                lite=False,
            )
            _drop_stale_summaries(sys_obj)
            _restore_slim_cosmic_keys(sys_obj)
            systems.append(sys_obj)

        flat_keys = _flat_keys(cut, cfg)
        chi2_map = {}
        for s in tqdm(systems, desc=f"{cut} cov", unit="var"):
            if _has_det_systematics(s):
                _add_flat_systematics(s, flat_keys)
            proc_keys = _proc_keys(s)
            do_xsec_cov = (cut == cfg.cuts[-1]) and (s.variable_name in {"costheta", "momentum", "differential"})
            s.compute_covariances(keys=proc_keys, compute_xsec_cov=do_xsec_cov)

            combine_keys, summary_keys = _summary_groups(s, flat_keys)
            s.combine_summaries(summary_keys=combine_keys)
            s.compute_inverse_covariances()
            # Keep the build universe tree clean: summaries live in memory for plots only.
            s.save(save_dir=save_dir, metadata_dir="metadata_detsys", save_summaries=False)
            if s.sel_data is not None:
                chi2_map[s.variable_name] = s._calc_chi2(keys=["total"], include_summary=True)

            if args.skip_plots:
                continue
            s.xlabel = s._get_default_xlabel()
            suffix = f"_{cut}cut"
            s.set_colors()
            for unc_type in ("event", "xsec"):
                if unc_type == "xsec" and s.variable_name not in {"costheta", "momentum", "differential"}:
                    continue
                for use_fracunc in (True, False):
                    label = "errs" if use_fracunc else "unc"
                    fig, ax, _ = s.plot_event_rate_errs(
                        unc_type,
                        exclude_keys=combine_keys,
                        use_fracunc=use_fracunc,
                        sort=True,
                    )
                    plotters.add_label(
                        ax,
                        INTERNAL_LABEL,
                        where=(0.01, 1.01),
                        fontsize=8,
                        color="black",
                    )
                    plotters.save_plot(
                        f"{s.variable_name}_{unc_type}_{label}_full{suffix}",
                        fig=fig,
                        folder_name=plot_dir,
                    )
                    plt.close(fig)

                    fig, ax, _ = s.plot_event_rate_errs(
                        unc_type,
                        include_keys=summary_keys,
                        use_fracunc=use_fracunc,
                        sort=True,
                    )
                    plotters.add_label(
                        ax,
                        INTERNAL_LABEL,
                        where=(0.01, 1.01),
                        fontsize=8,
                        color="black",
                    )
                    plotters.save_plot(
                        f"{s.variable_name}_{unc_type}_{label}_summary{suffix}",
                        fig=fig,
                        folder_name=plot_dir,
                    )
                    plt.close(fig)

            if args.plot_universes:
                s.plot_all_distributions(
                    plot_key="sel",
                    plot_dir=universe_plot_dir,
                    save_plots=True,
                    suffix=suffix,
                )
                if s.variable_name in {"costheta", "momentum", "differential"}:
                    s.plot_all_distributions(
                        plot_key="sigma_tilde",
                        plot_dir=f"{universe_plot_dir}/xsec_scaled",
                        save_plots=True,
                        scale_by_xsec_unit=True,
                        suffix=suffix,
                    )
                    s.plot_all_distributions(
                        plot_key="sigma_tilde",
                        plot_dir=f"{universe_plot_dir}/xsec_unscaled",
                        save_plots=True,
                        scale_by_xsec_unit=False,
                        suffix=suffix,
                    )

            if args.plot_covariance_matrices:
                s.plot_all_covariance_matrices(
                    plot_dir=cov_plot_dir,
                    save_plots=True,
                    progress_bar=True,
                    suffix=suffix,
                )
            # Defensive close in case any plotting path leaves figures open.
            plt.close("all")

        if chi2_map:
            print(f"Chi2 summary for cut {cut}:")
            for key, val in chi2_map.items():
                print(f" - {key}: chi2={val['chi2']:.4f}, dof={val['dof']}")

    print("Done analyzing universes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
