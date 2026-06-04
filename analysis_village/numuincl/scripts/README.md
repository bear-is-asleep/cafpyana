# Detector systematics pipeline

Chunked scripts for building and evaluating Pandora numu detector systematics (slim RW, detector variations, CALO, cosmic). Configuration lives in [`detsys_config.py`](../detsys_config.py) at the repo root.

## Pipeline overview

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `build_det_event_lists.py` | Pairwise nominal ∩ variation event lists and POT scaling |
| 2 | `build_detsys_universes.py` | Chunked universe build (requires step 1 artifacts) |
| 3 | `analyze_detsys_universes.py` | Covariances, summaries, fractional uncertainty plots |

Reference notebook (not run by these scripts): [`notebooks/detsys.ipynb`](../notebooks/detsys.ipynb).

## Step 1: Event matching and POT scaling

For each detector variation (`pmtgain`, `nosce`, `wiremodxtheta`, …), find mcnu events in common with nominal on **`(file_index, __ntuple, entry)`** (same convention as chunked CAF loads: `file_index = __ntuple // 1000`) and write:

- Per-var CSV: `{data_dir}/det_var/{pds|sce|wiremod}/{version}/{var}_runs.csv` with columns `file_index`, `__ntuple`, `entry`
- Global JSON: `{data_dir}/det_var/{version}/pot_scaling.json`

**After updating matching code**, regenerate step 1 before building universes:

```bash
python scripts/build_det_event_lists.py --small   # or full file lists
```

## Step 2: Build universes

Loads slim MC in chunks, applies per-variation event filters and scaled POT, builds RW / det / CALO / cosmic universes, and saves under `{data_dir}/data/{day}/syst/universes/{cut}/`.

```bash
# Build universes
python scripts/build_detsys_universes.py ...
```

POT and livetime for scaling come from `{data_dir}/data/{day}/syst/metadata/normalization.json`, which is **recomputed on every build** from the current `file_map`. File subsampling in [`detsys_config.py`](../detsys_config.py): `--tiny` / `--small` use `sample_div` on nominal/det; aux keys (`OFFBEAM_FNAMES`, `MC_LOWE_FNAMES`, `DATA_OFFBEAM_FNAMES`) use `aux_div` only when both are set (no double slice). Full builds use `aux_div` on that aux group only. Slim/cosmic/aux paths use the matching normalization keys from the same lists.

Detector variations use `pot_scaling.json` for **common-event lists** (`*_runs.csv`) only. Sample POT for `scale_to_pot` is on-the-fly `POT_NOMINAL` / `POT_DET[var]` from `normalization.json`, times MC-only length ratio after the common-event filter (same as [`notebooks/detsys.ipynb`](../notebooks/detsys.ipynb)). Offbeam data and MC lowe are combined **after** that filter/scale on the **first chunk only** of each CV and variation loop (same pattern as slim RW). `event_ratio_*` in the json is not multiplied into sample POT.

Activate the cafpyana env first (`source setup.sh` from `numuincl/`, or `source .../envs/venv_py310_cafpyana/bin/activate`).

Debug det POT scaling (use `--small` for a fast replay; pipe stdout to `logs/` if you want):

```bash
python scripts/debug_det_pot_chunk.py --day checkpoint7_test8 --var pmtgain --cut precut --small
python scripts/build_detsys_universes.py --day checkpoint7_test8 --debug-pot --small \
  > logs/det_pot_debug_checkpoint7_test8.log 2>&1
```

`--debug-pot` prints staged diagnostics from [`detsys_pot_debug.py`](../detsys_pot_debug.py) (safe to delete that module later): `entry`, `post_filter`, `post_scale`, `post_aux`, `invariants`, plus aggregate `costheta_sel_col0` vs `costheta_sel_allcols`. Healthy chunks show `entry aux=0`, `post_scale aux=0`, unchanged `mc`/`genweight_mc_sig` from `post_scale` to `post_aux`. `VIOLATION` means aux on entry, scale on non-MC rows, or MC mutation during aux combine. To remove all debug code: delete `detsys_pot_debug.py` and strip `DEBUG-POT` hooks in `build_detsys_universes.py` (import, `--debug-pot`, `pot_debug=` args).

Chunk concatenation does not sum POT or livetime from loaded files; `CAFSlice.combine` metadata is reset from the JSON values after each combine.

## Step 3: Analyze universes

Loads saved universes, computes covariances and inverse covariances, writes summaries, and plots fractional uncertainties.

```bash
python scripts/analyze_detsys_universes.py --day checkpoint7 --ncpu 1

# One cut, skip plots
python scripts/analyze_detsys_universes.py --cut fv --skip-plots
```

Outputs:

- Summaries: `{data_dir}/data/{day}/syst/{cut}/`
- Plots: `{data_dir}/plots/{day}/syst/{cut}/fracunc/`

## Tests

```bash
python sbnd/tests/test_det_event_lists.py
python sbnd/tests/test_chunked_systematics.py
```

Optional integration test against real HDF (slow):

```bash
DET_EVENT_LISTS_INTEGRATION=1 python sbnd/tests/test_det_event_lists.py
```

## Related scripts

- `bad_h5_files.py` / `remove_all_bad_h5_files.sh`: identify or remove corrupt input files before a long build.
