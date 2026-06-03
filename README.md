[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17819391.svg)](https://doi.org/10.5281/zenodo.17819391)

# PReSto LMR Template

By [David Edge](https://orcid.org/0000-0001-6938-2850), [Deborah Khider](https://orcid.org/0000-0001-7501-8430), [Nick McKay](https://orcid.org/0000-0003-3598-5113), [Tanaya Gondhalekar](https://orcid.org/0009-0004-2440-3266), & [Julien Emile-Geay](https://orcid.org/0000-0001-5920-4751).

[PReSto](https://paleopresto.com) (Paleoclimate Reconstruction Storehouse) lowers the barriers to utilizing, reproducing, and customizing paleoclimate reconstructions. This repository is a template used by PReSto to run the Last Millennium Reanalysis (LMR) via GitHub Actions.

## LMR Method

This template reproduces and customizes the Last Millennium Reanalysis, version 2.1 ([Tardif et al., 2019](https://doi.org/10.5194/cp-15-1251-2019)), which uses the offline data assimilation method of [Hakim et al. (2016)](https://doi.org/10.1002/2016JD024751). The reconstruction is implemented using the [cfr](https://fzhu2e.github.io/cfr/) Python package ([Zhu et al., 2024](https://doi.org/10.5194/gmd-17-3409-2024)).

Proxy observations are drawn from either:
- **Archived compilations** (e.g., PAGES 2k v2) downloaded directly from [LiPDverse](https://lipdverse.org)
- **Filtered selections** queried from LiPDverse via PReSto's interactive map interface

The prior is CCSM4 Last Millennium simulation (850–1850 CE) for surface temperature (`tas`) and precipitation (`pr`).

## File Structure

| Path | Purpose |
|------|---------|
| `scripts/cfr_main_code.py` | Main reconstruction driver |
| `scripts/lipd_to_pdb.py` | Converts LiPD `.lpd` files to cfr ProxyDatabase |
| `scripts/convert_lipd_to_cfr_dataframe.py` | Converts legacy LiPD pickle to CFR DataFrame |
| `scripts/combine_seeds.py` | Merges multi-seed reconstruction outputs into `combined_recon.nc` |
| `lmr_configs.yml` | Reconstruction parameters (overwritten per run by PReSto) |
| `query_params.json` | Data query filters (committed by PReSto to trigger the workflow) |
| `Dockerfile` | Container definition for the cfr environment |
| `environment.yml` | Conda environment specification |
| `CITATION.cff` | Citation metadata |

## Workflows

### `cfr-custom.yml` — LMR CFR Reconstruction

Two-job pipeline triggered by a push to `query_params.json` or manual dispatch:

1. **prepare-data** — Acquires proxy data via one of three pathways:
   - *Archived*: downloads a pre-built compilation pickle from LiPDverse
   - *Filtered*: runs the `lipdGenerator` Docker container to query LiPDverse and package selected `.lpd` files, then converts them to a cfr ProxyDatabase
   - *Traditional*: downloads a pre-generated pickle from a provided URL
2. **reconstruct** — Runs the CFR reconstruction inside the `davidedge/lmr2` Docker container, combines seed runs, uploads results as artifacts, and commits them to the repository

### `visualize.yml` — Visualization

Triggered automatically after a successful `cfr-custom.yml` run (or manually). Calls the [presto-viz](https://github.com/DaveEdge1/presto-viz) reusable workflow to generate an interactive visualization and deploys it to GitHub Pages.

## How to Use

1. **Fork or clone** this repository
2. Edit `lmr_configs.yml` to customize reconstruction parameters — see the [cfr LMR guide](https://fzhu2e.github.io/cfr/ug-lmr.html) for configuration options
3. Push your changes; the workflow triggers automatically when `query_params.json` is updated, or run it manually from the **Actions** tab
4. Reconstruction results are saved as artifacts (90-day retention) and committed to the `recons/` directory
5. Visualizations are deployed to the repository's GitHub Pages site
