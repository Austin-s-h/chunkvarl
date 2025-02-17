# chunkvarl

YouTube -

## Getting Started

This project is managed using uv. To create a ready-to-run virtual environment containing all dependencies for simulation run

```bash
uv venv
source .venv/bin/activate

usage: calcdry [-h] [--simulations SIMULATIONS] [--output-dir OUTPUT_DIR] [--bootstrap-samples BOOTSTRAP_SAMPLES] data_file
calcdry --simulations 100 --output-dir varl_thiev --bootstrap-samples 10000 tests/data/varl_thieving.csv
```

## Running your own simulations

TODO: Update `calc_dry.py` to operate on a simple data frame or even better, some sort of wiki lookup.

## Developer Setup

For documenting and testing purposes, or for local development

```bash
uv sync --all-extras
invoke lint
invoke test
```
