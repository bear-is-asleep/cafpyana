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

OFFBEAM_COMBINE_OFFSET = int(1e7)

# Match mcnu on physics ID (detsys.ipynb); filter slices on index keys.
PHYSICS_MATCH_COLUMNS = ["run", "subrun", "evt", "E"]
EVENT_CSV_COLUMNS = ["__ntuple", "entry"]
NOMINAL_ALL_ENTRIES_NAME = "nominal_ntuple_entries.csv"


def det_var_subdir(var: str) -> str:
    if var in PDS_VARS:
        return "pds"
    if var in SCE_VARS:
        return "sce"
    if var in WIREMOD_VARS:
        return "wiremod"
    raise ValueError(f"Unknown detector variation: {var}")


def det_var_root(cfg: DetsysConfig) -> Path:
    """Write/read root for det event-list artifacts (under out_dir)."""
    return Path(cfg.out_dir) / "det_var"


def runs_csv_path(cfg: DetsysConfig, var: str) -> Path:
    return det_var_root(cfg) / det_var_subdir(var) / cfg.version / f"{var}_runs.csv"


def nominal_runs_csv_path(cfg: DetsysConfig, var: str) -> Path:
    return (
        det_var_root(cfg)
        / det_var_subdir(var)
        / cfg.version
        / f"{var}_nominal_runs.csv"
    )


def nominal_all_entries_path(cfg: DetsysConfig) -> Path:
    return det_var_root(cfg) / cfg.version / NOMINAL_ALL_ENTRIES_NAME


def pot_scaling_path(cfg: DetsysConfig) -> Path:
    return det_var_root(cfg) / cfg.version / "pot_scaling.json"


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


def _index_frame_from_mcnu_index(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ntuple": _index_level_values(index, "__ntuple").astype(np.int64),
            "entry": _index_level_values(index, "entry").astype(np.int64),
        }
    )


def _mc_row_mask(index: pd.Index) -> np.ndarray:
    ntuple = _index_level_values(index, "__ntuple")
    return ntuple < OFFBEAM_COMBINE_OFFSET


def _hdr_event_column_names(hdr: CAF) -> tuple[str, str, str]:
    names = []
    for col in ("run", "subrun", "evt"):
        if col in hdr.data.columns:
            names.append(col)
        else:
            names.append(hdr.get_key(col)[0])
    return tuple(names)


def _attach_hdr_columns(mcnu: NU, hdr: CAF) -> None:
    """Attach hdr run/subrun/evt to each mcnu row."""
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


def _physics_column_names(mcnu: NU) -> tuple[Any, Any, Any, Any]:
    for col in ("run", "subrun", "evt"):
        if col not in mcnu.data.columns:
            raise ValueError(
                f"mcnu missing hdr column {col!r}; attach hdr before physics match"
            )
    e_col = mcnu.get_key("E")[0]
    return "run", "subrun", "evt", e_col


def _dedupe_physics_frame(frame: pd.DataFrame) -> pd.DataFrame:
    len_before = len(frame)
    out = frame.drop_duplicates(subset=PHYSICS_MATCH_COLUMNS, keep="first")
    len_after = len(out)
    if len_before != len_after:
        print(f"Dropped {len_before - len_after} duplicate physics-key rows")
    return out.reset_index(drop=True)


def _mcnu_physics_frame(mcnu: NU, hdr: CAF) -> pd.DataFrame:
    m = mcnu.copy(deep=False)
    if len(hdr.data) > 0:
        _attach_hdr_columns(m, hdr)
    run_col, subrun_col, evt_col, e_col = _physics_column_names(m)
    frame = pd.DataFrame(
        {
            "run": m.data[run_col].to_numpy(dtype=np.int64),
            "subrun": m.data[subrun_col].to_numpy(dtype=np.int64),
            "evt": m.data[evt_col].to_numpy(dtype=np.int64),
            "E": np.round(m.data[e_col].to_numpy(dtype=float), 6),
            "__ntuple": _index_level_values(m.data.index, "__ntuple").astype(np.int64),
            "entry": _index_level_values(m.data.index, "entry").astype(np.int64),
        }
    )
    return _dedupe_physics_frame(frame)


def nominal_ntuple_entries_frame(mcnu: NU) -> pd.DataFrame:
    """All unique nominal (__ntuple, entry) keys (no physics dedupe)."""
    frame = _index_frame_from_mcnu_index(mcnu.data.index)
    len_before = len(frame)
    out = frame.drop_duplicates(subset=EVENT_CSV_COLUMNS, keep="first")
    len_after = len(out)
    if len_before != len_after:
        print(f"Dropped {len_before - len_after} duplicate nominal index-key rows")
    return out.reset_index(drop=True)


def prepare_mcnu_event_index(mcnu: NU, hdr: CAF) -> NU:
    """One row per unique physics key with (__ntuple, entry) attached."""
    out = mcnu.copy(deep=False)
    out.data = _mcnu_physics_frame(mcnu, hdr)
    return out


def mcnu_indexed_event_counts(mcnu: NU, hdr: CAF) -> int:
    return len(_mcnu_physics_frame(mcnu, hdr))


def _index_events_from_common(common: pd.DataFrame, suffix: str) -> pd.DataFrame:
    ntuple_col = f"__ntuple{suffix}"
    entry_col = f"entry{suffix}"
    out = common[[ntuple_col, entry_col]].rename(
        columns={ntuple_col: "__ntuple", entry_col: "entry"}
    )
    return out.drop_duplicates(subset=EVENT_CSV_COLUMNS).reset_index(drop=True)


def pairwise_common_events(
    nom_mcnu: NU,
    nom_hdr: CAF,
    var_mcnu: NU,
    var_hdr: CAF,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int]:
    """
    Intersect nominal and variation mcnu on (run, subrun, evt, E).

    Returns (var_events_df, nom_events_df, n_nominal, n_variation, n_common).
    Each events_df holds (__ntuple, entry) in that sample's load index.
    """
    nom_events = _mcnu_physics_frame(nom_mcnu, nom_hdr)
    var_events = _mcnu_physics_frame(var_mcnu, var_hdr)
    n_nominal = len(nom_events)
    n_variation = len(var_events)
    common = nom_events.merge(
        var_events,
        on=PHYSICS_MATCH_COLUMNS,
        how="inner",
        suffixes=("_nom", "_var"),
    )
    n_common = len(common)
    if n_common == 0:
        raise ValueError(
            f"No common mcnu events (nominal={n_nominal}, variation={n_variation})"
        )
    var_index_df = _index_events_from_common(common, "_var")
    nom_index_df = _index_events_from_common(common, "_nom")
    return var_index_df, nom_index_df, n_nominal, n_variation, n_common


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
    nominal_runs_csv: str,
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
        "nominal_runs_csv": nominal_runs_csv,
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


def _load_index_events_csv(path: Path) -> pd.DataFrame:
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


def load_variation_events(cfg: DetsysConfig, var: str) -> pd.DataFrame:
    return _load_index_events_csv(runs_csv_path(cfg, var))


def load_nominal_common_events(cfg: DetsysConfig, var: str) -> pd.DataFrame:
    return _load_index_events_csv(nominal_runs_csv_path(cfg, var))


def _slice_index_frame(slc: CAFSlice) -> pd.DataFrame:
    index = slc.data.index
    mc_mask = _mc_row_mask(index)
    if not mc_mask.any():
        return pd.DataFrame(columns=[*EVENT_CSV_COLUMNS, "_row"])
    sub_index = index[mc_mask]
    frame = _index_frame_from_mcnu_index(sub_index)
    frame["_row"] = np.flatnonzero(mc_mask)
    return frame


def filter_slice_to_events(slc: CAFSlice, events_df: pd.DataFrame) -> CAFSlice:
    """Keep slice rows whose (__ntuple, entry) is in the common-event CSV."""
    from sbnd.cafclasses.slice import CAFSlice as _CAFSlice

    if events_df.empty:
        return _CAFSlice(slc.data.iloc[0:0].copy(), pot=slc.pot)

    events = events_df[EVENT_CSV_COLUMNS].drop_duplicates()
    work = _slice_index_frame(slc)
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
        for path in (runs_csv_path(cfg, var), nominal_runs_csv_path(cfg, var)):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path}; run scripts/build_det_event_lists.py"
                )
    return scaling
