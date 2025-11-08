# Sample Output Schemas

This document shows the structure and sample data for pipeline outputs.

## CSV Outputs

### 1. pwar_by_race_2025.csv

Per-race pWAR results for each driver.

```csv
Season,RoundNumber,EventName,Driver,Team,GridPosition,pWAR_points_mean,pWAR_points_p5,pWAR_points_p95,pWAR_wins_mean,pWAR_wins_p5,pWAR_wins_p95
2025,1,Bahrain Grand Prix,VER,Red Bull Racing,1,8.45,6.12,10.78,0.34,0.24,0.43
2025,1,Bahrain Grand Prix,HAM,Mercedes,3,5.23,3.45,7.01,0.21,0.14,0.28
2025,1,Bahrain Grand Prix,LEC,Ferrari,2,6.78,4.89,8.67,0.27,0.20,0.35
2025,1,Bahrain Grand Prix,NOR,McLaren,4,4.12,2.34,5.90,0.16,0.09,0.24
2025,1,Bahrain Grand Prix,SAI,Ferrari,5,3.45,1.78,5.12,0.14,0.07,0.20
2025,2,Saudi Arabian Grand Prix,VER,Red Bull Racing,1,9.12,7.23,11.01,0.36,0.29,0.44
2025,2,Saudi Arabian Grand Prix,LEC,Ferrari,2,7.34,5.45,9.23,0.29,0.22,0.37
```

**Columns**:
- `Season`: Championship year
- `RoundNumber`: Race number in season
- `EventName`: Grand Prix name
- `Driver`: Driver 3-letter code
- `Team`: Constructor name
- `GridPosition`: Starting grid position
- `pWAR_points_mean`: Mean pWAR in championship points
- `pWAR_points_p5`: 5th percentile (lower credible interval)
- `pWAR_points_p95`: 95th percentile (upper credible interval)
- `pWAR_wins_mean`: Mean pWAR in win equivalents (points/25)
- `pWAR_wins_p5`: 5th percentile for wins
- `pWAR_wins_p95`: 95th percentile for wins

### 2. pwar_by_season_2025.csv

Season totals aggregated per driver.

```csv
Season,Driver,Teams,RacesCount,pWAR_points_mean,pWAR_points_p5,pWAR_points_p95,pWAR_wins_mean,pWAR_wins_p5,pWAR_wins_p95
2025,VER,Red Bull Racing,22,186.4,165.3,207.5,7.46,6.61,8.30
2025,HAM,Mercedes,22,142.8,125.6,160.0,5.71,5.02,6.40
2025,LEC,Ferrari,22,128.5,110.2,146.8,5.14,4.41,5.87
2025,NOR,McLaren,22,98.3,82.1,114.5,3.93,3.28,4.58
2025,SAI,Ferrari,22,87.2,70.8,103.6,3.49,2.83,4.14
2025,RUS,Mercedes,22,76.4,61.2,91.6,3.06,2.45,3.66
2025,PER,Red Bull Racing,22,45.3,32.1,58.5,1.81,1.28,2.34
2025,ALO,Aston Martin,22,38.7,24.5,52.9,1.55,0.98,2.12
```

**Columns**:
- `Season`: Championship year
- `Driver`: Driver 3-letter code
- `Teams`: Constructor(s) driver raced for (comma-separated if changed teams)
- `RacesCount`: Number of races included
- `pWAR_points_mean`: Total season pWAR in championship points
- `pWAR_points_p5/p95`: Credible interval bounds
- `pWAR_wins_mean`: Total season pWAR in win equivalents
- `pWAR_wins_p5/p95`: Credible interval bounds

## Interpretation Guide

### What do the numbers mean?

**Positive pWAR**: Driver added value above replacement
- `pWAR_points_mean = 10.5` → Driver earned ~10.5 more championship points than a replacement-level driver would have in the same car

**pWAR in wins**: Easier interpretation
- `pWAR_wins_mean = 2.0` → Driver was worth approximately 2 race wins above replacement

**Credible intervals**: Uncertainty bounds
- `[p5, p95]` represents 90% credible interval
- Wider intervals = more uncertainty (e.g., fewer laps, inconsistent performance)

### Example Interpretations

1. **VER: 186.4 points (7.46 wins) above replacement**
   - Verstappen added the equivalent of 7.5 race wins compared to a replacement-level driver
   - With 90% probability, his true WAR is between 6.6 and 8.3 wins

2. **HAM: 142.8 points (5.71 wins)**
   - Hamilton was worth 5.7 wins above replacement
   - Strong performer but not quite at Verstappen's level

3. **PER: 45.3 points (1.81 wins)**
   - Perez added value but significantly less than his teammate
   - Suggests the Red Bull car was strong, but Perez's individual contribution was moderate

### Comparing Teammates

Teammate deltas reveal driver quality holding car constant:
- **VER vs PER**: 141.1 point gap → Verstappen hugely outperformed his teammate
- **HAM vs RUS**: 66.4 point gap → Hamilton outperformed, but gap is smaller
- **LEC vs SAI**: 41.3 point gap → Leclerc ahead but both competitive

### Negative pWAR

If a driver has **negative pWAR**, they performed *below* replacement level:
- This suggests the team would have scored more points with an average available driver
- Rare for full-season drivers, more common for rookies or struggling drivers

## Posterior Trace File

### pace_trace_2025.nc

NetCDF file containing full posterior samples. Can be loaded in Python:

```python
import arviz as az

# Load trace
trace = az.from_netcdf('out/pace_trace_2025.nc')

# View summary
print(az.summary(trace, var_names=['driver_ability']))

# Extract driver abilities
driver_abilities = trace.posterior['driver_ability']  # shape: (chains, draws, n_drivers)

# Plot posteriors
az.plot_posterior(trace, var_names=['sigma_driver', 'sigma_team_race'])
```

## Diagnostics File

### diagnostics.csv

Model convergence metrics:

```csv
parameter,mean,sd,hdi_5%,hdi_95%,r_hat,ess_bulk,ess_tail
intercept,0.012,0.045,-0.058,0.082,1.001,2456,2789
sigma_driver,0.234,0.021,0.201,0.268,1.003,1834,2012
sigma_team_race,0.876,0.043,0.809,0.943,1.002,2123,2345
sigma,1.234,0.012,1.215,1.253,1.000,3456,3234
```

**Key metrics**:
- `r_hat`: Should be < 1.05 (ideally < 1.01). Measures chain convergence.
- `ess_bulk`: Effective sample size for bulk of distribution. Higher is better (aim for >1000).
- `ess_tail`: Effective sample size for tails. Important for credible intervals.

## HTML Report Preview

The HTML report (`report_2025.html`) includes:

1. **Executive Summary**
   - Top 10 leaderboard table
   - Key findings and season highlights

2. **Visualizations**
   - Top drivers bar chart with error bars
   - Driver pace abilities (sec/lap)
   - Per-race contributions stacked bars

3. **Methodology**
   - Data filtering criteria
   - Model specification
   - Replacement level definition
   - Simulation approach

4. **Diagnostics**
   - Convergence summary
   - Parameter counts
   - R-hat and ESS statistics

5. **Appendix**
   - Full run metadata (JSON)
   - All parameters used
   - Software versions

## Run Metadata

### run_metadata.json

```json
{
  "pipeline_version": "0.1.0",
  "run_timestamp": "2025-03-15T14:32:18.123456",
  "year": 2025,
  "parameters": {
    "min_lap_quantile": 0.01,
    "max_lap_quantile": 0.99,
    "n_draws": 1000,
    "n_tune": 1000,
    "target_accept": 0.9,
    "n_sims": 10000,
    "n_laps": 60,
    "overtake_difficulty": 0.3,
    "replacement_quantile": 0.25,
    "n_posterior_draws": 500,
    "random_seed": 42,
    "include_driver_track": false,
    "include_driver_team": false
  }
}
```

This ensures complete reproducibility — anyone can rerun with identical settings.
