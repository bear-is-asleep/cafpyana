"""Per-variation detector systematic event matching and POT scaling artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from detsys_config import PDS_VARS, SCE_VARS, WIREMOD_VARS

if TYPE_CHECKING:
    from detsys_config import DetsysConfig
    from sbnd.cafclasses.nu import NU
    from sbnd.cafclasses.parent import CAF
    from sbnd.cafclasses.slice import CAFSlice

# Same step as sbnd.general.utils.offset_ntuple_index / CAF._load_combined file_idx.
NTUPLE_FILE_STEP = 1000
OFFBEAM_COMBINE_OFFSET = int(1e7)

EVENT_CSV_COLUMNS = ["file_index", "__ntuple", "entry"]


def det_var_subdir(var: str) -> str:
    if var in PDS_VARS:
        return "pds"
    if var in SCE_VARS:
        return "sce"
    if var in WIREMOD_VARS:
        return "wiremod"
    raise ValueError(f"Unknown detector variation: {var}")


def runs_csv_path(cfg: DetsysConfig, var: str) -> Path:
    subdir = det_var_subdir(var)
    return Path(cfg.data_dir) / "det_var" / subdir / cfg.version / f"{var}_runs.csv"


def pot_scaling_path(cfg: DetsysConfig) -> Path:
    return Path(cfg.data_dir) / "det_var" / cfg.version / "pot_scaling.json"


def _index_depth(index: pd.Index) -> int:
    if isinstance(index, pd.MultiIndex):
        return index.nlevels
    return 1


def _index_level_values(index: pd.Index, name: str) -> np.ndarray:
    if not isinstance(index, pd.MultiIndex):
        if name == "__ntuple":
            return np.asarray(index, dtype=np.int64)
        raise ValueError(
            f"Index level {name!r} required but index is not a MultiIndex "
            f"(names={getattr(index, 'names', None)})"
        )
    if name not in index.names:
        raise ValueError(
            f"Index level {name!r} not found in {index.names}; "
            "expected CAF index with __ntuple and entry"
        )
    return index.get_level_values(name).to_numpy(dtype=np.int64)


def _triple_frame_from_index(index: pd.Index) -> pd.DataFrame:
    ntuple = _index_level_values(index, "__ntuple")
    entry = _index_level_values(index, "entry")
    file_index = ntuple // NTUPLE_FILE_STEP
    return pd.DataFrame(
        {
            "file_index": file_index.astype(np.int64),
            "__ntuple": ntuple.astype(np.int64),
            "entry": entry.astype(np.int64),
        }
    )


def _dedupe_triple_frame(triple: pd.DataFrame) -> pd.DataFrame:
    len_before = len(triple)
    out = triple.drop_duplicates(subset=EVENT_CSV_COLUMNS, keep="first")
    len_after = len(out)
    if len_before != len_after:
        print(f"Dropped {len_before - len_after} duplicate triple-key rows")
    return out.reset_index(drop=True)


def _mcnu_triple_frame(mcnu: NU) -> pd.DataFrame:
    return _dedupe_triple_frame(_triple_frame_from_index(mcnu.data.index))


def _mc_row_mask(index: pd.Index) -> np.ndarray:
    ntuple = _index_level_values(index, "__ntuple")
    return ntuple < OFFBEAM_COMBINE_OFFSET


def _slice_triple_frame(slc: CAFSlice) -> pd.DataFrame:
    index = slc.data.index
    mc_mask = _mc_row_mask(index)
    if not mc_mask.any():
        return pd.DataFrame(columns=EVENT_CSV_COLUMNS)
    triple = _triple_frame_from_index(index[mc_mask])
    triple["_row"] = np.flatnonzero(mc_mask)
    return triple


def _hdr_event_column_names(hdr: CAF) -> tuple[str, str, str]:
    names = []
    for col in ("run", "subrun", "evt"):
        if col in hdr.data.columns:
            names.append(col)
        else:
            names.append(hdr.get_key(col)[0])
    return tuple(names)


def _attach_hdr_columns(mcnu: NU, hdr: CAF) -> None:
    """
    Attach hdr run/subrun/evt to each mcnu row (optional logging / legacy checks).
    """
    run_col, subrun_col, evt_col = _hdr_event_column_names(hdr)
    hdr_event = hdr.data[[run_col, subrun_col, evt_col]]

    if len(mcnu.data) == len(hdr.data) and mcnu.data.index.equals(hdr.data.index):
        mcnu.data["run"] = hdr_event[run_col].values
        mcnu.data["subrun"] = hdr_event[subrun_col].values
        mcnu.data["evt"] = hdr_event[evt_col].values
        return

    mcnu_depth = _index_depth(mcnu.data.index)
    hdr_depth = _index_depth(hdr.data.index)

    if mcnu_depth > hdr_depth:
        mcnu_keys = pd.MultiIndex.from_tuples(
            [idx[:hdr_depth] if isinstance(idx, tuple) else (idx,) for idx in mcnu.data.index]
        )
        aligned = hdr_event.reindex(mcnu_keys)
        if aligned.isna().any().any():
            n_miss = int(aligned.isna().any(axis=1).sum())
            raise ValueError(
                f"{n_miss} mcnu rows could not be matched to hdr "
                f"(mcnu={len(mcnu.data)}, hdr={len(hdr.data)})"
            )
        mcnu.data["run"] = aligned[run_col].values
        mcnu.data["subrun"] = aligned[subrun_col].values
        mcnu.data["evt"] = aligned[evt_col].values
        return

    if mcnu_depth == hdr_depth:
        aligned = hdr_event.reindex(mcnu.data.index)
        if aligned.isna().any().any():
            n_miss = int(aligned.isna().any(axis=1).sum())
            raise ValueError(
                f"{n_miss} mcnu rows could not be matched to hdr "
                f"(mcnu={len(mcnu.data)}, hdr={len(hdr.data)})"
            )
        mcnu.data["run"] = aligned[run_col].values
        mcnu.data["subrun"] = aligned[subrun_col].values
        mcnu.data["evt"] = aligned[evt_col].values
        return

    raise ValueError(
        f"Cannot align mcnu (index depth {mcnu_depth}, len {len(mcnu.data)}) "
        f"to hdr (index depth {hdr_depth}, len {len(hdr.data)})"
    )


def prepare_mcnu_event_index(mcnu: NU, hdr: CAF) -> NU:
    """
    Build unique (file_index, __ntuple, entry) keys from mcnu index.

    hdr is accepted for API compatibility; matching uses index triples only.
    Returns a shallow NU copy whose data holds one row per unique triple.
    """
    m = mcnu.copy(deep=True)
    if len(hdr.data) > 0:
        _attach_hdr_columns(m, hdr)
    triple = _mcnu_triple_frame(m)
    out = mcnu.copy(deep=False)
    out.data = triple
    return out


def mcnu_indexed_event_counts(mcnu: NU, hdr: CAF) -> int:
    return len(prepare_mcnu_event_index(mcnu, hdr).data)


def pairwise_common_events(
    nom_mcnu: NU,
    nom_hdr: CAF,
    var_mcnu: NU,
    var_hdr: CAF,
) -> tuple[pd.DataFrame, int, int, int]:
    """
    Pairwise intersection of nominal and variation mcnu on (file_index, __ntuple, entry).

    Returns (events_df, n_nominal, n_variation, n_common).
    """
    nom_triple = _mcnu_triple_frame(nom_mcnu)
    var_triple = _mcnu_triple_frame(var_mcnu)
    n_nominal = len(nom_triple)
    n_variation = len(var_triple)
    common = nom_triple.merge(var_triple, on=EVENT_CSV_COLUMNS, how="inner")
    n_common = len(common)
    if n_common == 0:
        raise ValueError(
            f"No common mcnu events (nominal={n_nominal}, variation={n_variation})"
        )
    events_df = common[EVENT_CSV_COLUMNS].copy()
    return events_df, n_nominal, n_variation, n_common


def write_runs_csv(events_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(path, index=False)


def build_variation_scaling_entry(
    *,
    var: str,
    n_nominal: int,
    n_variation: int,
    n_common: int,
    pot_nominal_full: float,
    pot_det_full: float,
    runs_csv: str,
) -> dict[str, Any]:
    event_ratio_nominal = n_common / n_nominal if n_nominal else 0.0
    event_ratio_var = n_common / n_variation if n_variation else 0.0
    return {
        "subdir": det_var_subdir(var),
        "n_nominal": n_nominal,
        "n_variation": n_variation,
        "n_common": n_common,
        "event_ratio_nominal": event_ratio_nominal,
        "event_ratio_var": event_ratio_var,
        "POT_NOMINAL_FULL": pot_nominal_full,
        "POT_NOMINAL_FILTERED": pot_nominal_full * event_ratio_nominal,
        "POT_DET_FULL": pot_det_full,
        "POT_DET_SCALED": pot_det_full * event_ratio_var,
        "runs_csv": runs_csv,
    }


def write_pot_scaling_json(cfg: DetsysConfig, payload: dict[str, Any]) -> Path:
    path = pot_scaling_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def load_pot_scaling(cfg: DetsysConfig) -> dict[str, Any]:
    path = pot_scaling_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run scripts/build_det_event_lists.py first"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_variation_events(cfg: DetsysConfig, var: str) -> pd.DataFrame:
    path = runs_csv_path(cfg, var)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run scripts/build_det_event_lists.py first"
        )
    df = pd.read_csv(path)
    missing = set(EVENT_CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for col in EVENT_CSV_COLUMNS:
        df[col] = df[col].astype(np.int64)
    return df


def _slice_col(slc: CAFSlice, key: str) -> Any:
    """Resolve a dotted CAF column key to the dataframe column tuple."""
    return slc.get_key(key)[0]


def _slice_event_columns(slc: CAFSlice) -> tuple[Any, Any, Any, Any]:
    """Run/subrun/evt/truth.E columns for debug logging only."""
    return (
        _slice_col(slc, "run"),
        _slice_col(slc, "subrun"),
        _slice_col(slc, "evt"),
        _slice_col(slc, "truth.E"),
    )


def filter_slice_to_events(slc: CAFSlice, events_df: pd.DataFrame) -> CAFSlice:
    """
    Keep slice rows whose (file_index, __ntuple, entry) is in the common-event CSV.
    """
    from sbnd.cafclasses.slice import CAFSlice as _CAFSlice

    if events_df.empty:
        return _CAFSlice(slc.data.iloc[0:0].copy(), pot=slc.pot)

    events = events_df[EVENT_CSV_COLUMNS].drop_duplicates()
    work = _slice_triple_frame(slc)
    if work.empty:
        return _CAFSlice(slc.data.iloc[0:0].copy(), pot=slc.pot)

    matched = work.merge(events, on=EVENT_CSV_COLUMNS, how="inner")
    keep_rows = np.sort(matched["_row"].unique().astype(int))
    return _CAFSlice(slc.data.iloc[keep_rows], pot=slc.pot)


def require_artifacts(cfg: DetsysConfig, det_vars: list[str]) -> dict[str, Any]:
    scaling = load_pot_scaling(cfg)
    variations = scaling.get("variations", {})
    missing_vars = [v for v in det_vars if v not in variations]
    if missing_vars:
        raise FileNotFoundError(
            f"pot_scaling.json missing variations: {missing_vars}; "
            "run scripts/build_det_event_lists.py"
        )
    for var in det_vars:
        csv_path = runs_csv_path(cfg, var)
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing {csv_path}; run scripts/build_det_event_lists.py"
            )
    return scaling
