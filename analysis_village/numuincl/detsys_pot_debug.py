"""
Det POT stage diagnostics for --debug-pot.

Standalone module: delete this file and remove DEBUG-POT hooks in
scripts/build_detsys_universes.py to drop all debug output.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sbnd.cafclasses.slice import CAFSlice

OFFBEAM_COMBINE_OFFSET = int(1e7)
_GENWEIGHT_RTOL = 1e-9


@dataclass(frozen=True)
class RowBreakdown:
    total: int
    mc: int
    aux: int
    sig_mc: int
    genweight_mc_sig: float


@dataclass(frozen=True)
class ChunkStageContext:
    var: str | None
    mode: str
    chunk_idx: int
    event_ratio: float | None = None


class DetPotDebug:
    """No-op when disabled; all build-script hooks may call methods unconditionally."""

    _disabled: DetPotDebug | None = None

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    @classmethod
    def for_build(cls, enabled: bool) -> DetPotDebug:
        if enabled:
            return cls(enabled=True)
        if cls._disabled is None:
            cls._disabled = cls(enabled=False)
        return cls._disabled

    def _print(self, msg: str) -> None:
        if self.enabled:
            print(msg, flush=True)

    @staticmethod
    def _ntuple_level(index) -> np.ndarray:
        if isinstance(index, np.ndarray):
            return index
        import pandas as pd

        if isinstance(index, pd.MultiIndex):
            if "__ntuple" in index.names:
                return index.get_level_values("__ntuple").to_numpy()
            return index.get_level_values(-1).to_numpy()
        return np.asarray(index)

    @classmethod
    def mc_row_mask(cls, index) -> np.ndarray:
        return cls._ntuple_level(index) < OFFBEAM_COMBINE_OFFSET

    @classmethod
    def snapshot(cls, slc: CAFSlice, categories: list[int]) -> RowBreakdown:
        n_total = len(slc.data)
        if n_total == 0:
            return RowBreakdown(0, 0, 0, 0, 0.0)
        mc_mask = cls.mc_row_mask(slc.data.index)
        n_mc = int(mc_mask.sum())
        n_aux = n_total - n_mc
        genweight_mc_sig = 0.0
        n_sig_mc = 0
        if n_mc > 0:
            truth_event_type_col = slc.get_key("truth.event_type")[0]
            gen_col = slc.get_key("genweight")[0]
            types = slc.data.loc[mc_mask, truth_event_type_col].values
            sig_mask = np.isin(types, categories)
            n_sig_mc = int(sig_mask.sum())
            weights = slc.data.loc[mc_mask, gen_col].astype(float).values
            genweight_mc_sig = float(weights[sig_mask].sum())
        return RowBreakdown(n_total, n_mc, n_aux, n_sig_mc, genweight_mc_sig)

    @staticmethod
    def costheta_sel_totals(sel: Any) -> tuple[float | None, float | None]:
        arr = np.asarray(sel, dtype=float)
        if arr.size == 0:
            return None, None
        flat = arr.ravel()
        return float(flat[0]), float(flat.sum())

    def start_rse_sample_file(self, save_dir: str, var: str) -> Path | None:
        """Create {save_dir}/debug/rse_sample_{var}.txt for chunk-0 RSE dumps."""
        if not self.enabled:
            return None
        path = Path(save_dir) / "debug" / f"rse_sample_{var}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self._print(f"[debug-pot] rse_sample var={var} path={path}")
        return path

    @staticmethod
    def mc_row_count(slc: CAFSlice) -> int:
        if len(slc.data) == 0:
            return 0
        return int(DetPotDebug.mc_row_mask(slc.data.index).sum())

    @staticmethod
    def format_rse_mc_sample(slc: CAFSlice, *, n: int = 50) -> str:
        """First n MC rows sorted by (run, subrun, evt): run, subrun, evt, truth.E."""
        from detsys_det_match import _slice_event_columns

        run_col, subrun_col, evt_col, e_col = _slice_event_columns(slc)
        df = slc.data
        if len(df) == 0:
            return "(no rows)\n"
        mc_mask = DetPotDebug.mc_row_mask(df.index)
        if not mc_mask.any():
            return "(no mc rows)\n"
        sub = df.loc[mc_mask, [run_col, subrun_col, evt_col, e_col]].copy()
        sub = sub.sort_values([run_col, subrun_col, evt_col], kind="mergesort")
        sub = sub.head(n)
        e_vals = pd.to_numeric(sub[e_col], errors="coerce")
        out = pd.DataFrame(
            {
                "run": sub[run_col].to_numpy(),
                "subrun": sub[subrun_col].to_numpy(),
                "evt": sub[evt_col].to_numpy(),
                "E": e_vals.to_numpy(),
            }
        )
        return out.to_string(index=False, float_format="%.6f") + "\n"

    def append_rse_sample_section(
        self,
        slc: CAFSlice,
        label: str,
        path: Path | None,
        *,
        n: int = 50,
    ) -> None:
        if not self.enabled or path is None:
            return
        n_total = len(slc.data)
        n_mc = self.mc_row_count(slc)
        self._print(
            f"[debug-pot] rse_sample section={label!r} "
            f"slc.data rows={n_total} mc_rows={n_mc} (showing first {n} sorted by RSE)"
        )
        body = self.format_rse_mc_sample(slc, n=n)
        block = (
            f"=== {label} (slc.data rows={n_total}, mc_rows={n_mc}) ===\n"
            "run subrun evt E\n"
            f"{body}\n"
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(block)

    def log_cosmic_sample_lengths(
        self,
        cut: str,
        slc_nominal_sig: CAFSlice,
        slc_offbeam_mc: CAFSlice,
        slc_offbeam_data: CAFSlice,
    ) -> None:
        if not self.enabled:
            return
        n_nom = len(slc_nominal_sig.data)
        n_mc = len(slc_offbeam_mc.data)
        n_data = len(slc_offbeam_data.data)
        ratio = n_mc / n_data if n_data else float("inf")
        self._print(
            f"[debug-pot] cut={cut} cosmic samples "
            f"nominal_sig={n_nom} offbeam_mc={n_mc} offbeam_data={n_data} "
            f"ratio_mc_over_data={ratio:.6f} "
            f"(mc mc_rows={self.mc_row_count(slc_offbeam_mc)} "
            f"data mc_rows={self.mc_row_count(slc_offbeam_data)})"
        )

    def log_var_header(
        self,
        var: str,
        var_scale: dict,
        norm: dict,
        ratio_nom: float,
        ratio_var: float,
    ) -> None:
        if not self.enabled:
            return
        self._print(
            f"[debug-pot] var={var} n_common={var_scale.get('n_common')} "
            f"event_ratio_nominal={ratio_nom:.6f} event_ratio_var={ratio_var:.6f} "
            f"POT_NOMINAL={float(norm['POT_NOMINAL']):.6e} "
            f"POT_DET={float(norm['POT_DET'][var]):.6e} "
            f"POT_DATA={float(norm['POT_DATA']):.6e} "
            f"json_POT_NOMINAL_FILTERED={var_scale.get('POT_NOMINAL_FILTERED')} "
            f"json_POT_DET_SCALED={var_scale.get('POT_DET_SCALED')}"
        )

    def log_scale_apply(
        self,
        *,
        n_rows_scaled: int,
        n_aux_rows: int,
        sample_pot: float,
        n_before: int,
        n_after: int,
        pot_eff: float,
    ) -> None:
        if not self.enabled:
            return
        frac = n_after / n_before if n_before else 0.0
        self._print(
            f"[debug-pot] stage=scale_apply rows={n_rows_scaled} aux_rows={n_aux_rows} "
            f"n_mc_before={n_before} n_mc_after={n_after} filter_frac={frac:.6f} "
            f"sample_pot={sample_pot:.6e} pot_eff={pot_eff:.6e} "
            f"(pot_eff=sample_pot*filter_frac)"
        )

    def trace_det_chunk(
        self,
        *,
        ctx: ChunkStageContext,
        entry: RowBreakdown,
        post_filter: RowBreakdown,
        post_scale: RowBreakdown,
        post_aux: RowBreakdown,
        sample_pot: float,
        pot_eff: float,
        n_mc_before_filter: int,
        pot_data: float,
    ) -> None:
        if not self.enabled:
            return
        er_str = f"{ctx.event_ratio:.6f}" if ctx.event_ratio is not None else "n/a"
        scale_factor = pot_data / pot_eff if pot_eff else 0.0
        filter_frac = (
            post_filter.mc / n_mc_before_filter if n_mc_before_filter else 0.0
        )
        head = (
            f"var={ctx.var} mode={ctx.mode} chunk={ctx.chunk_idx} "
            f"event_ratio_json={er_str}"
        )
        self._print(
            f"[debug-pot] stage=entry {head} "
            f"total={entry.total} mc={entry.mc} aux={entry.aux} sig_mc={entry.sig_mc}"
        )
        self._print(
            f"[debug-pot] stage=post_filter {head} "
            f"mc={post_filter.mc} sig_mc={post_filter.sig_mc} aux={post_filter.aux} "
            f"genweight_mc_sig={post_filter.genweight_mc_sig:.6e} filter_frac={filter_frac:.6f}"
        )
        self._print(
            f"[debug-pot] stage=post_scale {head} "
            f"mc={post_scale.mc} sig_mc={post_scale.sig_mc} aux={post_scale.aux} "
            f"genweight_mc_sig={post_scale.genweight_mc_sig:.6e} "
            f"sample_pot={sample_pot:.6e} pot_eff={pot_eff:.6e} scale_factor={scale_factor:.6e}"
        )
        self._print(
            f"[debug-pot] stage=post_aux {head} "
            f"mc={post_aux.mc} sig_mc={post_aux.sig_mc} aux={post_aux.aux} "
            f"genweight_mc_sig={post_aux.genweight_mc_sig:.6e} "
            f"aux_attach_mode=first_chunk_only aux_rows={post_aux.aux}"
        )
        self._check_chunk_invariants(
            head=head,
            entry=entry,
            post_filter=post_filter,
            post_scale=post_scale,
            post_aux=post_aux,
        )

    def trace_det_chunk_empty_after_filter(
        self,
        *,
        ctx: ChunkStageContext,
        entry: RowBreakdown,
        n_mc_before_filter: int,
    ) -> None:
        if not self.enabled:
            return
        head = f"var={ctx.var} mode={ctx.mode} chunk={ctx.chunk_idx}"
        self._print(
            f"[debug-pot] stage=entry {head} "
            f"total={entry.total} mc={entry.mc} aux={entry.aux} sig_mc={entry.sig_mc}"
        )
        self._print(
            f"[debug-pot] stage=post_filter {head} mc=0 (empty after filter, "
            f"n_mc_before={n_mc_before_filter})"
        )
        if entry.aux > 0:
            self._print(
                f"[debug-pot] VIOLATION: {head} aux={entry.aux} on det chunk entry "
                "(expected 0 before mc-only filter)"
            )

    def _check_chunk_invariants(
        self,
        *,
        head: str,
        entry: RowBreakdown,
        post_filter: RowBreakdown,
        post_scale: RowBreakdown,
        post_aux: RowBreakdown,
    ) -> None:
        violations: list[str] = []
        if entry.aux > 0:
            violations.append(f"entry.aux={entry.aux} (expected 0 on det load path)")
        if post_filter.mc > entry.mc:
            violations.append(
                f"post_filter.mc={post_filter.mc} > entry.mc={entry.mc} (filter increased mc rows)"
            )
        if post_scale.mc != post_filter.mc:
            violations.append(
                f"post_scale.mc={post_scale.mc} != post_filter.mc={post_filter.mc} (scale changed mc rows)"
            )
        if post_scale.aux > 0:
            violations.append(f"post_scale.aux={post_scale.aux} (scale must be mc-only)")
        if post_aux.mc != post_scale.mc:
            violations.append(
                f"post_aux.mc={post_aux.mc} != post_scale.mc={post_scale.mc}"
            )
        if post_aux.sig_mc != post_scale.sig_mc:
            violations.append(
                f"post_aux.sig_mc={post_aux.sig_mc} != post_scale.sig_mc={post_scale.sig_mc}"
            )
        gw_scale = post_scale.genweight_mc_sig
        gw_aux = post_aux.genweight_mc_sig
        if gw_scale != 0.0 or gw_aux != 0.0:
            denom = max(abs(gw_scale), abs(gw_aux), 1e-30)
            if abs(gw_aux - gw_scale) / denom > _GENWEIGHT_RTOL:
                violations.append(
                    f"genweight_mc_sig changed after aux combine "
                    f"({gw_scale:.6e} -> {gw_aux:.6e})"
                )
        if violations:
            for v in violations:
                self._print(f"[debug-pot] VIOLATION: {head} {v}")
        else:
            self._print(f"[debug-pot] stage=invariants {head} ok")

    def log_aggregate(
        self,
        var: str,
        cv_stats: Any,
        var_stats: Any,
        costheta_sel: Any | None,
        *,
        n_cv_chunks: int,
        n_var_chunks: int,
    ) -> None:
        if not self.enabled:
            return
        cv_frac = cv_stats.n_after / cv_stats.n_before if cv_stats.n_before else 0.0
        var_frac = var_stats.n_after / var_stats.n_before if var_stats.n_before else 0.0
        self._print(
            f"[debug-pot] var={var} aggregate cv n_before={cv_stats.n_before} "
            f"n_after={cv_stats.n_after} filter_frac={cv_frac:.6f} "
            f"n_aux_before={cv_stats.n_aux_before} n_aux_after={cv_stats.n_aux_after} "
            f"genweight_mc_sig={cv_stats.genweight_sum_after:.6e}"
        )
        self._print(
            f"[debug-pot] var={var} aggregate variation n_before={var_stats.n_before} "
            f"n_after={var_stats.n_after} filter_frac={var_frac:.6f} "
            f"n_aux_before={var_stats.n_aux_before} n_aux_after={var_stats.n_aux_after} "
            f"genweight_mc_sig={var_stats.genweight_sum_after:.6e}"
        )
        aux_per_cv = cv_stats.n_aux_after if n_cv_chunks else 0
        aux_per_var = var_stats.n_aux_after if n_var_chunks else 0
        self._print(
            f"[debug-pot] var={var} aux_attach cv_chunks={n_cv_chunks} "
            f"aux_on_first_cv_chunk={aux_per_cv} var_chunks={n_var_chunks} "
            f"aux_on_first_var_chunk={aux_per_var}"
        )
        if costheta_sel is not None:
            col0, allcols = self.costheta_sel_totals(costheta_sel)
            if col0 is not None and allcols is not None:
                self._print(
                    f"[debug-pot] var={var} costheta_sel_col0={col0:.6e} "
                    f"costheta_sel_allcols={allcols:.6e}"
                )
