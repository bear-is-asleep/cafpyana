# $\nu_\mu$ Inclusive Analysis

## Introduction

- Extract inclusive $\nu_\mu$ events with the Pandora reconstruction chain.
- Shared helpers and classes live in `sbnd/` (mysbndana / sbnd_helper).

## Getting started (setup)

- Pull the `sbnd` submodule (mysbndana / sbnd_helper):

```bash
# from cafpyana root, after cloning the fork
git submodule update --init analysis_village/numuincl/sbnd
```

## Making Files

- Selection and dataframe jobs via `runit.py` or `runit.sh`. Runs grid or pool mode.
- Jobs described by YAMLs under `yamls/`; maker logic in `makedf1muX.py`.
- Example: `python runit.py -y yamls/<sample>.yaml` (optional `--dry-run`, `--only`).



## Postprocessing of files

- Move job outputs with `scripts/move_files.py` as needed.
- Detector systematics are built in chunks (RAM limits). Use the chunked pipeline under `scripts/`; instructions in `scripts/README.md`.



## Analysis - core notebooks

- `systematics.ipynb`: evaluates reweightable systematics (including statistical uncertainties), combines them with detector systematics, and stores results with `systematic.save(...)`.
- `unfolding.ipynb`: runs unfolding / cross section extraction and stores results with `xsec.save(...)`.
- `interaction_plots.ipynb`: stacked histograms (loads systematics from `systematic.save(...)`), purity/efficiency tables, and resolution plots.

Other notebooks are deprecated, study-specific, or incomplete.

## Data

Analysis uses SBND Gen I production. Analysis files, plots, and saved data can be found here - `/exp/sbnd/data/users/brindenc/analyze_sbnd/numu/v10_06_00_validation/pandora`