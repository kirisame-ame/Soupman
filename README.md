# Primordial Soup Replicator Simulation

A Python simulation inspired by the replicator concept from The Selfish Gene. The model focuses on emergent selection in a well-mixed "primordial soup" with replication, mutation, decay, and finite resources. A Dear PyGui interface visualizes the population and metrics in real time.

## Features

- Discrete-time simulation with replication, mutation, and decay
- Resource-limited environment with emergent selection
- Live UI with plots, stats, and lineage tracking
- Optional batch analysis script for averaging outcomes

## Requirements

- Python 3.10+ (recommended)
- Dependencies in requirements.txt

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run the UI

```bash
python src/main.py
```

## Run the lineage analysis script

```bash
python src/scripts/run_lineage_analysis.py
```

This script runs multiple simulations, prints aggregate tables, and writes a diversity chart to docs/diversity_avg.png.

## Project structure

```text
src/
  analysis/        Metrics and statistics
  simulation/      Core engine, environment, and rules
  ui/              Dear PyGui interface
  main.py          Application entry point
  scripts/         Standalone analysis utilities

docs/
```

## Notes

- The visualization is illustrative, not a chemistry model.
- There is no explicit fitness function; selection emerges from the rules.
