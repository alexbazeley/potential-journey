# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Verify installation
python3 -c "import fastf1, pymc, arviz; print('All dependencies installed!')"
```

## Fast Test Run (2024 data, reduced parameters)

Since 2025 data may not be fully available yet, test with 2024:

```bash
# Quick test with reduced parameters (~5-10 minutes total)
python3 scripts/run_2025.py \
  --year 2024 \
  --draws 500 \
  --tune 500 \
  --sims 1000 \
  --n_posterior_draws 100 \
  --out out/test/
```

This will:
- Fetch 2024 season data (may take 2-3 minutes to download/cache)
- Fit model with fewer samples (faster but less precise)
- Run fewer simulations
- Generate all outputs in `out/test/`

## Check Outputs

```bash
# View CSV results
head out/test/pwar_by_season_2024.csv

# Open HTML report (macOS)
open out/test/report_2024.html

# Open HTML report (Linux)
xdg-open out/test/report_2024.html

# Check diagnostics
cat out/test/diagnostics.csv | head -20
```

## Full Production Run (2025 season)

Once you've verified the pipeline works:

```bash
# Production run with full parameters (~20-30 minutes)
python3 scripts/run_2025.py \
  --year 2025 \
  --draws 2000 \
  --tune 1500 \
  --sims 20000 \
  --n_posterior_draws 1000 \
  --out out/2025_final/ \
  --html out/2025_final/report.html
```

## Common Issues

### Issue: "No data for 2025 season"

**Solution**: The 2025 season may not have started or have limited data. Use 2024 for testing:
```bash
python3 scripts/run_2025.py --year 2024
```

### Issue: "ImportError: No module named 'pymc'"

**Solution**: Install dependencies:
```bash
pip3 install -r requirements.txt
```

### Issue: Model fitting is very slow

**Solution**: Reduce draws and tune:
```bash
python3 scripts/run_2025.py --draws 500 --tune 500
```

### Issue: Memory error during model fitting

**Solution**:
1. Reduce draws: `--draws 500 --tune 500`
2. Don't include optional random effects (default behavior)
3. Close other applications to free RAM

### Issue: FastF1 cache errors

**Solution**: Clear cache and retry:
```bash
rm -rf cache/
python3 scripts/run_2025.py
```

## Understanding the Results

After the pipeline completes, check these files:

1. **`pwar_by_season_<year>.csv`** - Season totals, sorted by pWAR
   - Look for `pWAR_wins_mean` column - this is the easiest to interpret
   - Top drivers should have 4-8 wins above replacement
   - Mid-tier drivers: 1-3 wins
   - Bottom drivers: 0-1 wins or negative

2. **`report_<year>.html`** - Full report with visualizations
   - Open in any web browser
   - Includes methodology, charts, and diagnostics

3. **`diagnostics.csv`** - Check convergence
   - Look at `r_hat` column: all values should be < 1.05
   - If r_hat > 1.05, increase `--tune` and rerun

## Next Steps

### Experiment with Parameters

```bash
# Vary replacement level
python3 scripts/run_2025.py --rep_quantile 0.30 --out out/rep30/

# Compare results
diff <(tail -n+2 out/pwar_by_season_2025.csv | sort) \
     <(tail -n+2 out/rep30/pwar_by_season_2025.csv | sort)
```

### Analyze Multiple Seasons

```bash
# Run for 2023, 2024, 2025
for year in 2023 2024 2025; do
  python3 scripts/run_2025.py --year $year --out out/$year/
done

# Compare top drivers across years
for year in 2023 2024 2025; do
  echo "=== $year ==="
  head -5 out/$year/pwar_by_season_$year.csv
done
```

### Load Posterior for Analysis

```python
import arviz as az
import pandas as pd

# Load trace
trace = az.from_netcdf('out/pace_trace_2025.nc')

# View driver abilities
driver_abilities = trace.posterior['driver_ability']
print(driver_abilities.mean(dim=['chain', 'draw']))

# Plot posterior distributions
az.plot_forest(trace, var_names=['sigma_driver', 'sigma_team_race'])

# Analyze specific drivers
# (Need to match driver IDs to names using encodings)
```

## Performance Tips

### Speed Up Data Fetching

FastF1 caches downloaded data. After the first run, subsequent runs for the same year are much faster.

### Speed Up Model Fitting

- Use fewer chains: PyMC defaults to 4 chains; reducing doesn't help much due to parallel sampling
- Use fewer draws: `--draws 1000` is usually sufficient for stable estimates
- Skip optional random effects (they're disabled by default)

### Speed Up Simulations

- Reduce `--sims` to 5000 (still gives good estimates)
- Reduce `--n_posterior_draws` to 200-300

### Minimal Run (for testing code changes)

```bash
# Ultra-fast test run
python3 scripts/run_2025.py \
  --year 2024 \
  --draws 100 \
  --tune 100 \
  --sims 100 \
  --n_posterior_draws 10 \
  --out out/quick_test/
```

⚠️ Results will be very imprecise but good for testing the pipeline.

## Getting Help

1. Check `war_f1_pipeline.log` for detailed logs
2. Review the full README.md for methodology details
3. See SAMPLE_OUTPUTS.md for expected output formats
4. Check diagnostics.csv for convergence issues

## Example: Interpreting Results

Suppose `pwar_by_season_2024.csv` shows:

```
Driver,pWAR_wins_mean
VER,7.2
HAM,4.8
LEC,4.1
```

**Interpretation**:
- Verstappen was worth **7.2 race wins** more than a replacement-level driver would have been in the Red Bull
- Hamilton: **4.8 wins** above replacement
- Leclerc: **4.1 wins** above replacement

The gaps between teammates (e.g., VER vs PER) are especially informative since they had the same car.
