# F1 WAR Metric Pipeline

**Formula 1 Wins Above Replacement (WAR-F1)** — A reproducible pipeline for computing driver value isolated from car performance.

## Overview

This project implements a comprehensive metric to quantify **driver value** in Formula 1, similar to WAR (Wins Above Replacement) in baseball. The metric isolates driver skill from car performance using Bayesian hierarchical modeling and race simulations.

### What is WAR-F1?

**WAR-F1** measures the expected **championship points** a driver adds **above a replacement-level driver** if both occupy the **same seat** (same constructor, same weekend conditions/strategy) across a season.

The metric has two components:
- **pWAR (pace-based)**: Derived from lap-time analysis using a Bayesian mixed-effects model
- **rWAR (results-based)**: *[Future enhancement]* Derived from finishing positions using rank-ordered models

## Features

✅ **Fully reproducible** pipeline with cached data and seeded random numbers
✅ **Bayesian hierarchical model** to separate driver ability from car performance
✅ **Race simulations** with overtaking logic to compute expected points
✅ **Uncertainty quantification** with credible intervals
✅ **Professional outputs**: CSV tables, charts, and HTML reports
✅ **Configurable parameters** via command-line interface

## Installation

### Requirements

- Python 3.10 or higher
- ~2GB disk space for FastF1 cache
- 8GB+ RAM recommended for Bayesian modeling

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd potential-journey

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or using pyproject.toml
pip install -e .
```

## Quick Start

Run the complete pipeline for the 2025 season with default parameters:

```bash
python3 scripts/run_2025.py
# Or make it executable and run directly:
# ./scripts/run_2025.py
```

This will:
1. Download and cache F1 data from FastF1
2. Filter and prepare lap data
3. Fit Bayesian hierarchical model (~10-20 minutes)
4. Simulate races and compute pWAR
5. Generate CSV outputs and HTML report in `out/`

### Custom Configuration

```bash
python3 scripts/run_2025.py \
  --draws 2000 \
  --tune 1500 \
  --sims 20000 \
  --rep_quantile 0.25 \
  --out out/ \
  --html out/report_2025.html
```

### All Options

```bash
python3 scripts/run_2025.py --help
```

Key parameters:
- `--draws`: Posterior draws per chain (default: 1000)
- `--tune`: Tuning steps for NUTS sampler (default: 1000)
- `--sims`: Race simulations per configuration (default: 10000)
- `--rep_quantile`: Replacement level quantile (default: 0.25)
- `--overtake_difficulty`: Track overtaking difficulty, 0-1 (default: 0.3)
- `--seed`: Random seed for reproducibility (default: 42)

## Methodology

### 1. Data Collection

- **Source**: FastF1 API (official F1 timing data)
- **Scope**: 2025 season race sessions only
- **Caching**: All downloads cached locally for reproducibility

### 2. Lap Filtering

Valid race laps must satisfy:
- ✅ Race session (not practice/qualifying)
- ✅ Not a pit in/out lap
- ✅ Green flag conditions only (`TrackStatus == '1'`)
- ✅ Accurate timing (`IsAccurate == True`)
- ✅ Within [P1, P99] quantile range per race (outlier removal)

### 3. Feature Engineering

**Outcome**: Demeaned lap time (LapTime - race median)

**Fixed effects** (context controls):
- Tyre compound (categorical)
- Tyre life + tyre life²
- Weather: air temp, track temp, humidity
- Stint number
- Grid position bucket (P1-4, P5-8, P9-20, Back)

**Random effects** (hierarchical structure):
- **Driver ability** ← *key metric of interest*
- **Constructor × Race** ← absorbs car/setup/strategy variance
- Optionally: Driver × Track, Driver × Constructor

### 4. Bayesian Hierarchical Model

```
LapTime_demean ~ Normal(μ, σ)

μ = intercept + β·X_fixed + driver_ability[i] + team_race_effect[j]

Priors:
  driver_ability[i] ~ Normal(0, σ_driver)
  team_race_effect[j] ~ Normal(0, σ_team_race)
  β ~ Normal(0, 0.5)
  σ_driver, σ_team_race ~ HalfNormal(...)
```

**Interpretation**:
- Driver ability in **seconds per lap** (negative = faster)
- Constructor × Race effects absorb all car-related variance per weekend
- Fixed effects control for tyre strategy, weather, etc.

### 5. Replacement Level

**Definition**: The 25th percentile of driver abilities per posterior draw.

This represents the performance of a "replacement-level" driver — someone who could be readily signed from the available driver pool.

### 6. Race Simulation

For each race:
1. Extract driver abilities from posterior
2. Simulate 10,000 race outcomes using:
   - Driver pace deltas (from model)
   - Grid positions
   - Simple overtaking logic (time-gap and difficulty-based)
3. Compute expected points for actual driver
4. **Swap** driver with replacement-level ability
5. Recompute expected points
6. **pWAR = actual points - replacement points**

Repeat across 500+ posterior draws to get credible intervals.

### 7. Aggregation

- **Per-race pWAR**: Points above replacement for each race
- **Season pWAR**: Sum across all races
- **pWAR in wins**: Divide by 25 points/win

## Outputs

All outputs saved to `out/` directory:

### CSV Files

1. **`pwar_by_race_2025.csv`**
   - Columns: `Season, RoundNumber, EventName, Driver, Team, GridPosition, pWAR_points_mean, pWAR_points_p5, pWAR_points_p95, pWAR_wins_mean, pWAR_wins_p5, pWAR_wins_p95`

2. **`pwar_by_season_2025.csv`**
   - Columns: `Season, Driver, Teams, RacesCount, pWAR_points_mean, pWAR_points_p5, pWAR_points_p95, pWAR_wins_mean, pWAR_wins_p5, pWAR_wins_p95`

### Posterior Trace

- **`pace_trace_2025.nc`**: ArviZ InferenceData in NetCDF format
  - Contains full posterior samples
  - Can be loaded for further analysis: `az.from_netcdf('pace_trace_2025.nc')`

### Diagnostics

- **`diagnostics.csv`**: Model convergence diagnostics
  - R-hat (should be < 1.05)
  - Effective sample size (ESS)
  - Summary statistics for all parameters

### HTML Report

- **`report_2025.html`**: Comprehensive report including:
  - Executive summary with top-10 leaderboard
  - Visualizations (charts embedded as PNGs)
  - Methodology section
  - Model diagnostics
  - Run metadata (parameters, versions, timestamp)

### Charts

Generated in `out/charts/`:
- `top_drivers.png`: Top 10 drivers by pWAR (bar chart with error bars)
- `driver_abilities.png`: Driver pace abilities in sec/lap
- `per_race_contributions.png`: Stacked bars showing per-race pWAR

### Metadata

- **`run_metadata.json`**: Complete run configuration
  - Timestamp
  - All parameters used
  - Software versions
  - Ensures full reproducibility

## Project Structure

```
.
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Package configuration
├── scripts/
│   └── run_2025.py           # Main CLI orchestrator
├── src/
│   └── war_f1/
│       ├── __init__.py
│       ├── data_fetch.py     # FastF1 data retrieval
│       ├── prep.py           # Data cleaning & feature engineering
│       ├── models_pace.py    # Bayesian hierarchical model (pWAR)
│       ├── simulate.py       # Race simulation & WAR calculation
│       ├── report.py         # Output generation & visualization
│       └── models_results.py # rWAR (placeholder, future)
├── cache/                    # FastF1 cache (gitignored)
└── out/                      # Pipeline outputs (gitignored)
```

## Validation

### Within-Team Comparison

The pipeline includes validation by comparing teammate performance:
- Driver abilities should correlate with qualifying/race head-to-head records
- Teammate deltas should be stable across races

### Convergence Diagnostics

- All parameters monitored for R-hat < 1.05
- Effective sample size (ESS) checked
- Warnings logged for convergence issues

### Sensitivity Analysis

Vary `--rep_quantile` (0.20–0.30) to check rank stability:
```bash
python3 scripts/run_2025.py --rep_quantile 0.20 --out out/rep20/
python3 scripts/run_2025.py --rep_quantile 0.30 --out out/rep30/
```

Top drivers should show minimal rank changes.

## Future Enhancements

### Planned Features

- [ ] **rWAR**: Results-based model using rank-ordered logit (Plackett-Luce)
- [ ] **Blended WAR**: Combine pWAR and rWAR with configurable weights
- [ ] **Track-specific overtaking difficulty**: Different parameters per circuit
- [ ] **Safety car modeling**: Probabilistic SC/VSC events and pit strategy
- [ ] **DRS/traffic effects**: Incorporate DRS zones and gap-based traffic modeling
- [ ] **Historical comparison**: Extend to multiple seasons (2018–2025)
- [ ] **Interactive dashboard**: Web-based visualization with Plotly/Dash

### Contributing

Contributions are welcome! Areas for improvement:
- Model enhancements (better priors, additional random effects)
- More sophisticated race simulation
- Additional validation metrics
- Performance optimization (parallel chains, GPU acceleration)

## Troubleshooting

### FastF1 Errors

If data fetching fails:
1. Check internet connection
2. Clear cache: `rm -rf cache/`
3. Try individual race: `fastf1.get_session(2025, 1, 'R').load()`

### Memory Issues

If model fitting fails due to memory:
- Reduce `--draws` and `--tune`
- Disable optional random effects: don't use `--include_driver_track`
- Filter to fewer races for testing

### Slow Performance

- Use fewer posterior draws for uncertainty: `--n_posterior_draws 200`
- Reduce simulations: `--sims 5000`
- Use faster model: avoid `--include_driver_track` and `--include_driver_team`

### Convergence Warnings

If R-hat > 1.05:
- Increase tuning: `--tune 2000`
- Increase target acceptance: `--target_accept 0.95`
- Check diagnostics in `out/diagnostics.csv`

## Citation

If you use this pipeline in research or analysis, please cite:

```
F1 WAR Metric Pipeline (2025)
https://github.com/your-repo/war-f1
```

## License

[Specify your license here]

## Acknowledgments

- **FastF1**: https://github.com/theOehrly/Fast-F1
- **PyMC**: https://www.pymc.io/
- **ArviZ**: https://python.arviz.org/

## Contact

For questions, issues, or suggestions, please open a GitHub issue or contact [your email].

---

**Built with ❤️ for F1 analytics**
