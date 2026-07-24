# Detector systematics pipeline

Chunked scripts for building and evaluating Pandora numu detector systematics (slim RW, detector variations, CALO, cosmic). Configuration lives in `[detsys_config.py](../detsys_config.py)` at the repo root.

## Pipeline overview


| Step | Script                        | Purpose                                                  |
| ---- | ----------------------------- | -------------------------------------------------------- |
| 1    | `build_det_event_lists.py`    | Pairwise nominal ∩ variation event lists and POT scaling |
| 2    | `build_detsys_universes.py`   | Chunked universe build (requires step 1 for det passes)  |
| 3    | `analyze_detsys_universes.py` | Covariances, summaries, fractional uncertainty plots     |


Reference notebook (not run by these scripts): `[notebooks/detsys.ipynb](../notebooks/detsys.ipynb)`.

## Build modes


| Flag                 | Mode               | Builds                             | Saves                             |
| -------------------- | ------------------ | ---------------------------------- | --------------------------------- |
| *(none)*             | `default`          | slim + det + CALO + cosmic         | all keys                          |
| `--small`            | `small`            | monolithic subsample               | all keys                          |
| `--tiny`             | `tiny`             | smoke test (1 file/group)          | all keys                          |
| `--full-slim`        | `full_slim`        | slim RW only                       | `xsec`, `flux`, `g4` (merge)      |
| `--full-det`         | `full_det`         | det + CALO only                    | det vars + CALO (writes metadata) |
| `--full-cosmic`      | `full_cosmic`      | cosmic only                        | `cosmic`, `cosmic_data` (merge)   |
| `--full-slim-test`   | `full_slim_test`   | same as full-slim at 1/100 files   | same as full-slim                 |
| `--full-det-test`    | `full_det_test`    | same as full-det at 1/100 files    | same as full-det                  |
| `--full-cosmic-test` | `full_cosmic_test` | same as full-cosmic at 1/100 files | same as full-cosmic               |


Only one mode flag at a time. Tunable divisors live as globals in `[detsys_config.py](../detsys_config.py)` (`_FULL_TEST_DIV = 100` for test modes).


Divisor semantics: `1` = full list, `0` = single file, `N≥2` = `len // N` head slice.

### Three-pass full production (same `--day`)

Recommended order: **det → cosmic → slim** (det writes CV metadata first).

```bash
python scripts/build_det_event_lists.py

python scripts/build_detsys_universes.py --full-det --day YYYYMMDD --recompute-norm
python scripts/build_detsys_universes.py --full-cosmic --day YYYYMMDD --recompute-norm
python scripts/build_detsys_universes.py --full-slim --day YYYYMMDD --recompute-norm

python scripts/analyze_detsys_universes.py --day YYYYMMDD
```

Smoke the three passes at 1/100 file count:

```bash
python scripts/build_detsys_universes.py --full-det-test --day test_YYYYMMDD --recompute-norm
python scripts/build_detsys_universes.py --full-cosmic-test --day test_YYYYMMDD --recompute-norm
python scripts/build_detsys_universes.py --full-slim-test --day test_YYYYMMDD --recompute-norm
```



## Step 1: Event matching and POT scaling

For each detector variation (`pmtgain`, `nosce`, `wiremodxtheta`, …), find mcnu events in common with nominal on `(run, subrun, evt, E)`, then write per-variation CSVs of matched `(__ntuple, entry)` keys (separate files for nominal vs variation index space) plus global `pot_scaling.json`. All nominal `(__ntuple, entry)` keys are also written to `det_var/{version}/nominal_ntuple_entries.csv`.

```bash
python scripts/build_det_event_lists.py
```

Artifacts land under `{out_dir}/det_var/` (default `out_dir` = `data_dir`), keyed by `version` (default `v9`):

- `det_var/{version}/pot_scaling.json`
- `det_var/{version}/nominal_ntuple_entries.csv`
- `det_var/{pds,sce,wiremod}/{version}/{var}_runs.csv`
- `det_var/{pds,sce,wiremod}/{version}/{var}_nominal_runs.csv`

### Reusing an existing event list

`build_detsys_universes.py` does not rebuild lists. Det modes (`--full-det`, `--full-det-test`, …) only **read** whatever already sits at `{out_dir}/det_var/.../{version}/`. Match `--version` (default `v9`) and `--out-dir` to what you used in step 1. No need to re-run step 1.

```bash
# lists already under default data_dir/det_var/{version}/
python scripts/build_detsys_universes.py --full-det-test --day test_YYYYMMDD --recompute-norm

# non-default version and/or lists elsewhere
python scripts/build_detsys_universes.py --full-det-test --day test_YYYYMMDD --recompute-norm \
  --version v9 --out-dir /path/to/parent_of_det_var
```

`--data-dir` still selects HDF inputs. `--out-dir` selects both event-list reads **and** universe writes (`{out_dir}/data/{day}/syst`). `--version` selects the `det_var/.../{version}/` path segment (same flag on `build_det_event_lists.py`).

## Step 2: Build universes

POT and livetime come from `{out_dir}/data/{day}/syst/metadata/normalization.json`. Pass `--recompute-norm` to rescan `file_map`.

`--full-det` / `--full-det-test` require step 1 det artifacts for the configured `version`. `--full-slim` and `--full-cosmic` do not.

Activate the cafpyana env first (`source setup.sh` from `numuincl/`).

## Step 3: Analyze universes

```bash
python scripts/analyze_detsys_universes.py --day checkpoint7 --ncpu 1
python scripts/analyze_detsys_universes.py --cut fv --skip-plots
```



## Tests

```bash
python sbnd/tests/test_build_splits.py
python sbnd/tests/test_det_event_lists.py
python sbnd/tests/test_chunked_systematics.py
```



## Related scripts

- `bad_h5_files.py` / `remove_all_bad_h5_files.sh`: identify or remove corrupt input files before a long build.

