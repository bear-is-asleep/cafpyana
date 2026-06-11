"""Shared chi2 JSON helpers for analyze/compute/plot scripts."""
from __future__ import annotations

import json
from pathlib import Path


def chi2_to_jsonable(entry: dict) -> dict:
    """Strip numpy arrays from _calc_chi2 output; keep JSON-safe scalars."""
    return {
        "keys": list(entry["keys"]),
        "chi2": float(entry["chi2"]),
        "dof": int(entry["dof"]),
        "pvalue": float(entry["pvalue"]),
    }


def save_chi2_json(save_dir: str | Path, chi2_map: dict[str, dict]) -> str | None:
    """Write per-variable chi2 summary to {save_dir}/chi2.json."""
    if not chi2_map:
        return None
    payload = {var_name: chi2_to_jsonable(entry) for var_name, entry in chi2_map.items()}
    out_path = Path(save_dir) / "chi2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(out_path)


def save_chi2_json_at(path: str | Path, chi2_map: dict[str, dict]) -> str | None:
    """Write chi2_map to an explicit JSON file path."""
    if not chi2_map:
        return None
    out_path = Path(path)
    payload = {var_name: chi2_to_jsonable(entry) for var_name, entry in chi2_map.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(out_path)
