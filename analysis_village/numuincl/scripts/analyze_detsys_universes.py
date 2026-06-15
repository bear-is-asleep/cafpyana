#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
CAFPYANA_WD = REPO_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(CAFPYANA_WD)):
    if p not in sys.path:
        sys.path.insert(0, p)

#My imports 
SBNDANA_DIR = '/exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/sbnd'
sys.path.insert(0,SBNDANA_DIR)
sys.path.insert(0,f'{SBNDANA_DIR.replace("/numuincl/sbnd","/numuincl")}')
plt.style.use(f'{SBNDANA_DIR}/plotlibrary/numu2025.mplstyle')

from chi2_io import save_chi2_json
from detsys_config import CUTS_FROM_MUON, CALO_VARS, DET_VARS_ALL, DetsysConfig, build_config
from naming import INTERNAL_LABEL
from sbnd.general import plotters
from sbnd.stats.systematics import Systematics

# Stale combined buckets written by old analyze/notebook saves; must not reload from disk or sys_dict.
SLIM_KEYS = ("xsec", "flux", "g4")
SUMMARY_BUCKET_KEYS = frozenset({"total", "pds", "sce", "tpc", "calo", *SLIM_KEYS})
IGNORE_KEYS = frozenset({"metadata_detsys", "geant4_syst", "detsys"})
LOAD_IGNORE_KEYS = IGNORE_KEYS | SUMMARY_BUCKET_KEYS
AGGREGATE_SUMMARY_KEYS = frozenset({"total", "pds", "sce", "tpc", "calo"})
RESTORE_SUMMARY_KEYS = frozenset({*SLIM_KEYS, "cosmic"})
# CV-only keys: provide reference histograms for a syst, not an uncertainty themselves.
CV_ONLY_KEYS = frozenset({"cosmic_data"})
_COV_FIELDS = (
    "event_cov",
    "event_cov_unaltered",
    "event_fraccov",
    "event_fraccov_unaltered",
    "event_corr",
    "event_fracunc",
    "event_totalunc",
    "event_inv_cov",
    "xsec_cov",
    "xsec_cov_unaltered",
    "xsec_fraccov",
    "xsec_fraccov_unaltered",
    "xsec_corr",
    "xsec_fracunc",
    "xsec_totalunc",
    "xsec_inv_cov",
)


def _drop_stale_summaries(s: Systematics) -> None:
    """Drop combined summary buckets and CV stash keys, not slim/cosmic universes."""
    stale_buckets = AGGREGATE_SUMMARY_KEYS | frozenset(SLIM_KEYS)
    for k in list(s.systematics.keys()):
        v = s.systematics[k]
        if k in IGNORE_KEYS or k in stale_buckets:
            s.systematics.pop(k, None)
            continue
        if v.get("variation") == "self" and k not in CV_ONLY_KEYS:
            s.systematics.pop(k, None)
            continue


def _ensure_slim_universe_types(s: Systematics) -> None:
    """Bundled slim keys must carry type=xsec|flux|g4 for combine_summaries roll-up."""
    type_by_key = {
        "xsec_slim": "xsec",
        "xsec_multisigma": "xsec",
        "flux": "flux",
        "g4": "g4",
    }
    for k, sd in s.systematics.items():
        if sd.get("variation") in ("summary", "self"):
            continue
        if k in type_by_key:
            sd["type"] = type_by_key[k]
        elif sd.get("type") in (None, "", "unknown") and k.startswith("ZExp"):
            sd["type"] = "xsec"


def _prepare_cosmic_key(s: Systematics) -> None:
    """
    cosmic uses cosmic_data for CV (nom signal + data offbeam).
    The cosmic key only holds the MC-offbeam unisim universe histogram.
    """
    if "cosmic" not in s.systematics:
        return
    d = s.systematics["cosmic"]
    d["type"] = "cosmic"
    d["variation"] = d.get("variation") or "unisim"
    sel = d.get("sel")
    if isinstance(sel, np.ndarray):
        if sel.ndim == 2:
            d["sel"] = [row.copy() for row in sel]
        else:
            d["sel"] = [sel.copy()]
    d["sel_background"] = []


def _prepare_cv_only_keys(s: Systematics) -> None:
    """cosmic_data is the cosmic CV stash only, never a covariance contributor."""
    for k in CV_ONLY_KEYS:
        if k not in s.systematics:
            continue
        d = s.systematics[k]
        d["variation"] = "self"
        if not d.get("type"):
            d["type"] = "cosmic"
        for field in _COV_FIELDS:
            d[field] = None


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


def _has_calo_systematics(s: Systematics) -> bool:
    return any(k in s.systematics for k in CALO_VARS)


def _flat_keys(cut: str, cfg: DetsysConfig, *, analyze_cut: str | None) -> list[str]:
    keys = ["nt", "stat_flat"]
    is_final_stage = analyze_cut is not None or cut == cfg.cuts[-1]
    if is_final_stage:
        return ["pot", *keys]
    return keys


def _clear_cov_fields(sys_dict: dict) -> None:
    """Drop stale covariance products so flat keys are always recomputed."""
    for field in _COV_FIELDS:
        sys_dict[field] = None


def _add_flat_systematics(s: Systematics, flat_keys: list[str]) -> None:
    """Register nt / stat_flat / pot variations for every variable at this cut."""
    if "nt" in flat_keys:
        if "nt" in s.systematics:
            _clear_cov_fields(s.systematics["nt"])
        s.process_flat_systematic("nt", 0.01)
    if "stat_flat" in flat_keys:
        if "stat_flat" in s.systematics:
            _clear_cov_fields(s.systematics["stat_flat"])
        s.process_stat_systematics("stat_flat")
    if "pot" in flat_keys:
        if "pot" in s.systematics:
            _clear_cov_fields(s.systematics["pot"])
        s.process_flat_systematic("pot", 0.02)


def _cov_proc_keys(s: Systematics, flat_keys: list[str]) -> list[str]:
    """Universe keys whose covariances feed the total summary (incl. flat systs)."""
    _add_flat_systematics(s, flat_keys)
    return list(dict.fromkeys(_proc_keys(s) + flat_keys))


def _slim_summary_keys(s: Systematics) -> list[str]:
    """Slim summary keys; xsec rolls up xsec_slim, xsec_multisigma, ZExp via type=xsec."""
    keys_present = set(s.systematics.keys())
    types_present = {
        sd.get("type")
        for sd in s.systematics.values()
        if sd.get("variation") not in ("summary", "self")
    }
    types_present.discard(None)
    out: list[str] = []
    for sk in SLIM_KEYS:
        if sk in keys_present or sk in types_present:
            out.append(sk)
            continue
        if sk == "xsec" and any(k in keys_present for k in ("xsec_slim", "xsec_multisigma")):
            out.append(sk)
        elif sk == "flux" and "flux" in keys_present:
            out.append(sk)
        elif sk == "g4" and "g4" in keys_present:
            out.append(sk)
    return out


def _summary_plot_keys(s: Systematics, flat_keys: list[str]) -> list[str]:
    """Summary buckets with computed event_totalunc (post combine_summaries)."""
    _, candidates = _summary_groups(s, flat_keys)
    return [
        k
        for k in candidates
        if s.systematics.get(k, {}).get("event_totalunc") is not None
    ]


def _summary_groups(s: Systematics, flat_keys: list[str]) -> tuple[list[str], list[str]]:
    """
    Match detsys.ipynb step 7: combine_keys only has total + det type groups;
    summary_keys adds flat, slim, cosmic for plots.
    """
    keys_present = set(s.systematics.keys())
    types_present = {sdict.get("type") for sdict in s.systematics.values()}
    types_present.discard(None)

    combine_keys: list[str] = ["total"]
    summary_keys: list[str] = ["total", *flat_keys]

    summary_keys.extend(_slim_summary_keys(s))

    if "cosmic" in keys_present:
        summary_keys.append("cosmic")

    if _has_calo_systematics(s) or "calo" in types_present or "calo" in keys_present:
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
        if k in IGNORE_KEYS or k in CV_ONLY_KEYS:
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
    parser.add_argument("--ncpu", type=int, default=16, help="Workers for loading saved systematics (use 1 on small nodes).")
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
            _ensure_slim_universe_types(sys_obj)
            _restore_slim_cosmic_keys(sys_obj)
            _prepare_cosmic_key(sys_obj)
            _prepare_cv_only_keys(sys_obj)
            systems.append(sys_obj)

        flat_keys = _flat_keys(cut, cfg, analyze_cut=args.cut)
        chi2_map = {}
        is_final_stage = args.cut is not None or cut == cfg.cuts[-1]
        for s in tqdm(systems, desc=f"{cut} cov", unit="var"):
            proc_keys = _cov_proc_keys(s, flat_keys)
            do_xsec_cov = is_final_stage and (
                s.variable_name in {"costheta", "momentum", "differential"}
            )
            s.compute_covariances(keys=proc_keys, compute_xsec_cov=do_xsec_cov)

            combine_keys, summary_keys = _summary_groups(s, flat_keys)
            if (
                cut in CUTS_FROM_MUON
                and _has_det_systematics(s)
                and not _has_calo_systematics(s)
                and "calo" not in combine_keys
            ):
                warnings.warn(
                    f"CALO systematics missing for {s.variable_name} at cut {cut}; "
                    f"calo summary will not be built. Re-run --full-det build "
                    f"(CALO snapshots must persist from muon through cont).",
                    stacklevel=2,
                )
            slim_summaries = _slim_summary_keys(s)
            combine_all = list(dict.fromkeys(combine_keys + slim_summaries))
            s.combine_summaries(summary_keys=combine_all)
            summary_plot_keys = _summary_plot_keys(s, flat_keys)
            inv_keys = list(dict.fromkeys(proc_keys + flat_keys + combine_all))
            s.compute_inverse_covariances(keys=inv_keys)
            # Keep the build universe tree clean: summaries live in memory for plots only.
            s.save(save_dir=save_dir, metadata_dir="metadata_detsys", save_summaries=True)
            if s.sel_data is not None:
                chi2_map[s.variable_name] = s._calc_chi2(keys=["total"], include_summary=True)

            if args.skip_plots:
                continue
            s.xlabel = s._get_default_xlabel()
            suffix = f"_{cut}cut"
            s.set_colors()
            for unc_type in ("event",):
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
                    if ax is None:
                        continue
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
                        include_keys=summary_plot_keys,
                        use_fracunc=use_fracunc,
                        sort=True,
                    )
                    if ax is None:
                        continue
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
            chi2_path = save_chi2_json(save_dir, chi2_map)
            print(f"Chi2 summary for cut {cut}:")
            for key, val in chi2_map.items():
                print(f" - {key}: chi2={val['chi2']:.4f}, dof={val['dof']}")
            if chi2_path:
                print(f"Wrote chi2 to {chi2_path}")

    print("Done analyzing universes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
