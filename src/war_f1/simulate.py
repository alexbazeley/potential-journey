"""
Race simulation and WAR calculation module.

Simulates race outcomes based on driver pace abilities, computes expected points,
and calculates WAR by comparing actual driver to replacement-level driver.
"""

import logging
from typing import Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# F1 points system (top 10)
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}


def compute_replacement_level(
    driver_ability_samples: np.ndarray,
    quantile: float = 0.25
) -> np.ndarray:
    """
    Compute replacement-level ability per posterior draw.

    Replacement level = the quantile-th percentile of driver abilities
    in each posterior draw.

    Parameters
    ----------
    driver_ability_samples : ndarray
        Shape (n_samples, n_drivers)
    quantile : float
        Quantile for replacement level (default 0.25 = 25th percentile)

    Returns
    -------
    ndarray
        Replacement level per sample, shape (n_samples,)
    """
    replacement = np.percentile(driver_ability_samples, quantile * 100, axis=1)
    logger.info(f"Replacement level (mean): {replacement.mean():.4f} sec/lap")
    return replacement


def simulate_race_finish(
    driver_pace: np.ndarray,
    grid_positions: np.ndarray,
    n_laps: int = 60,
    overtake_difficulty: float = 0.3,
    random_state: np.random.RandomState = None
) -> np.ndarray:
    """
    Simulate a single race finishing order based on pace and grid.

    Simple model:
    - Each driver has a total race time = base_time + pace_delta × n_laps
    - Overtaking: drivers can overtake if their cumulative time is better
      and a random overtake check passes (depends on time gap and difficulty)

    Parameters
    ----------
    driver_pace : ndarray
        Pace deltas (seconds per lap) for each driver, shape (n_drivers,)
    grid_positions : ndarray
        Starting grid positions, shape (n_drivers,)
    n_laps : int
        Number of laps in the race
    overtake_difficulty : float
        Track overtake difficulty [0, 1]. Higher = harder to overtake.
    random_state : RandomState
        Random number generator

    Returns
    -------
    ndarray
        Finishing positions (1-indexed), shape (n_drivers,)
    """
    if random_state is None:
        random_state = np.random.RandomState()

    n_drivers = len(driver_pace)

    # Total race time (simplified: pace × laps + random component)
    race_times = driver_pace * n_laps + random_state.normal(0, 1.0, size=n_drivers)

    # Adjust for grid positions (starting position gives small advantage)
    # Better grid = slight time reduction
    grid_advantage = (20 - grid_positions) * 0.1  # ~0.1 sec per position better than P20
    race_times -= grid_advantage

    # Overtake probability based on time gaps
    # Drivers with much better race times have higher chance to overtake
    # This is a simplified heuristic

    # Sort by race time to get potential finishing order
    finish_order = np.argsort(race_times)

    # Apply overtake difficulty: small chance to NOT overtake even if faster
    # Randomly swap adjacent drivers based on difficulty
    for _ in range(int(n_drivers * overtake_difficulty)):
        i = random_state.randint(0, n_drivers - 1)
        if random_state.rand() < overtake_difficulty:
            # Swap
            finish_order[i], finish_order[i + 1] = finish_order[i + 1], finish_order[i]

    # Convert to positions (1-indexed)
    positions = np.empty(n_drivers, dtype=int)
    positions[finish_order] = np.arange(1, n_drivers + 1)

    return positions


def simulate_race_many_times(
    driver_pace: np.ndarray,
    grid_positions: np.ndarray,
    n_sims: int = 10000,
    n_laps: int = 60,
    overtake_difficulty: float = 0.3,
    random_seed: int = 42
) -> np.ndarray:
    """
    Run many race simulations and return finishing positions.

    Parameters
    ----------
    driver_pace : ndarray
        Pace deltas for each driver, shape (n_drivers,)
    grid_positions : ndarray
        Starting grid positions
    n_sims : int
        Number of simulations
    n_laps : int
        Laps per race
    overtake_difficulty : float
        Track overtake difficulty
    random_seed : int
        Random seed

    Returns
    -------
    ndarray
        Finishing positions, shape (n_sims, n_drivers)
    """
    rs = np.random.RandomState(random_seed)
    n_drivers = len(driver_pace)

    all_positions = np.zeros((n_sims, n_drivers), dtype=int)

    for i in range(n_sims):
        positions = simulate_race_finish(
            driver_pace, grid_positions, n_laps, overtake_difficulty, rs
        )
        all_positions[i, :] = positions

    return all_positions


def positions_to_points(positions: np.ndarray) -> np.ndarray:
    """
    Convert finishing positions to FIA points.

    Parameters
    ----------
    positions : ndarray
        Finishing positions, shape (n_sims, n_drivers) or (n_drivers,)

    Returns
    -------
    ndarray
        Points awarded, same shape as input
    """
    points = np.zeros_like(positions, dtype=float)

    for pos, pts in POINTS_SYSTEM.items():
        points[positions == pos] = pts

    return points


def compute_expected_points(positions: np.ndarray) -> np.ndarray:
    """
    Compute expected points from simulated positions.

    Parameters
    ----------
    positions : ndarray
        Finishing positions, shape (n_sims, n_drivers)

    Returns
    -------
    ndarray
        Expected points per driver, shape (n_drivers,)
    """
    points = positions_to_points(positions)
    expected_points = points.mean(axis=0)
    return expected_points


def compute_pwar_for_race(
    race_data: pd.DataFrame,
    driver_ability_samples: np.ndarray,
    replacement_samples: np.ndarray,
    encodings: Dict[str, Dict],
    n_sims: int = 10000,
    n_laps: int = 60,
    overtake_difficulty: float = 0.3,
    n_posterior_draws: int = 500,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Compute pWAR for a single race by comparing actual vs replacement driver.

    For each driver in the race:
    1. Simulate race outcomes with actual driver abilities
    2. Swap each driver with replacement-level driver
    3. Compute expected points difference
    4. Repeat across posterior draws to get uncertainty

    Parameters
    ----------
    race_data : DataFrame
        Race summary (one row per driver)
    driver_ability_samples : ndarray
        Posterior samples of driver abilities, shape (n_samples, n_drivers)
    replacement_samples : ndarray
        Replacement level per sample, shape (n_samples,)
    encodings : dict
        Encoding mappings
    n_sims : int
        Simulations per race configuration
    n_laps : int
        Laps in the race
    overtake_difficulty : float
        Track overtake difficulty
    n_posterior_draws : int
        Number of posterior draws to use for uncertainty
    random_seed : int
        Random seed

    Returns
    -------
    DataFrame
        pWAR results with columns: Driver, Team, GridPosition,
        pWAR_points_mean, pWAR_points_p5, pWAR_points_p95, etc.
    """
    n_total_samples = driver_ability_samples.shape[0]

    # Sample indices for posterior draws
    rs = np.random.RandomState(random_seed)
    draw_indices = rs.choice(n_total_samples, size=min(n_posterior_draws, n_total_samples), replace=False)

    # Get driver IDs in this race
    driver_ids = race_data['DriverID'].values
    grid_positions = race_data['GridPosition'].values

    n_drivers_in_race = len(driver_ids)

    # Storage for pWAR across draws
    pwar_points_all_draws = np.zeros((len(draw_indices), n_drivers_in_race))

    for draw_idx, sample_idx in enumerate(tqdm(draw_indices, desc=f"Race {race_data['EventName'].iloc[0]}", leave=False)):
        # Get abilities for this draw
        abilities = driver_ability_samples[sample_idx, :]
        replacement = replacement_samples[sample_idx]

        # Extract abilities for drivers in this race
        driver_pace = abilities[driver_ids]

        # Simulate actual race
        actual_positions = simulate_race_many_times(
            driver_pace, grid_positions, n_sims, n_laps, overtake_difficulty, random_seed + sample_idx
        )
        actual_points = compute_expected_points(actual_positions)

        # Compute pWAR for each driver
        for i, driver_id in enumerate(driver_ids):
            # Swap driver with replacement
            counterfactual_pace = driver_pace.copy()
            counterfactual_pace[i] = replacement

            # Simulate counterfactual race
            cf_positions = simulate_race_many_times(
                counterfactual_pace, grid_positions, n_sims, n_laps, overtake_difficulty, random_seed + sample_idx + 1000
            )
            cf_points = compute_expected_points(cf_positions)

            # pWAR = actual expected points - replacement expected points
            pwar_points_all_draws[draw_idx, i] = actual_points[i] - cf_points[i]

    # Compute summary statistics across draws
    results = []
    for i, (driver_id, driver_row) in enumerate(race_data.iterrows()):
        pwar_samples = pwar_points_all_draws[:, i]

        results.append({
            'Driver': driver_row['Driver'],
            'Team': driver_row['Team'],
            'GridPosition': driver_row['GridPosition'],
            'pWAR_points_mean': pwar_samples.mean(),
            'pWAR_points_sd': pwar_samples.std(),
            'pWAR_points_p5': np.percentile(pwar_samples, 5),
            'pWAR_points_p50': np.percentile(pwar_samples, 50),
            'pWAR_points_p95': np.percentile(pwar_samples, 95),
            'pWAR_wins_mean': pwar_samples.mean() / 25,
            'pWAR_wins_p5': np.percentile(pwar_samples, 5) / 25,
            'pWAR_wins_p95': np.percentile(pwar_samples, 95) / 25,
        })

    return pd.DataFrame(results)


def compute_pwar_all_races(
    race_summaries: pd.DataFrame,
    trace: az.InferenceData,
    encodings: Dict[str, Dict],
    n_sims: int = 10000,
    n_laps: int = 60,
    overtake_difficulty: float = 0.3,
    replacement_quantile: float = 0.25,
    n_posterior_draws: int = 500,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pWAR for all races in the season.

    Parameters
    ----------
    race_summaries : DataFrame
        Per-driver, per-race summary
    trace : InferenceData
        Posterior samples
    encodings : dict
        Encoding mappings
    n_sims : int
        Simulations per race
    n_laps : int
        Average laps per race
    overtake_difficulty : float
        Overtake difficulty parameter
    replacement_quantile : float
        Quantile for replacement level
    n_posterior_draws : int
        Posterior draws for uncertainty
    random_seed : int
        Random seed

    Returns
    -------
    pwar_by_race : DataFrame
        pWAR per race per driver
    pwar_by_season : DataFrame
        Season totals per driver
    """
    logger.info("Computing pWAR for all races...")

    # Extract driver ability samples
    driver_ability_samples = trace.posterior['driver_ability'].values
    driver_ability_samples = driver_ability_samples.reshape(-1, driver_ability_samples.shape[-1])

    # Compute replacement level per sample
    replacement_samples = compute_replacement_level(driver_ability_samples, replacement_quantile)

    # Group by race
    races = race_summaries.groupby('RoundNumber')

    all_race_results = []

    for round_num, race_df in races:
        race_df = race_df.copy()

        pwar_race = compute_pwar_for_race(
            race_df,
            driver_ability_samples,
            replacement_samples,
            encodings,
            n_sims,
            n_laps,
            overtake_difficulty,
            n_posterior_draws,
            random_seed + round_num
        )

        # Add race info
        pwar_race['RoundNumber'] = round_num
        pwar_race['EventName'] = race_df['EventName'].iloc[0]
        pwar_race['Season'] = race_df.get('Season', 2025).iloc[0] if 'Season' in race_df.columns else 2025

        all_race_results.append(pwar_race)

    pwar_by_race = pd.concat(all_race_results, ignore_index=True)

    # Aggregate to season totals
    pwar_by_season = pwar_by_race.groupby('Driver').agg({
        'Team': lambda x: ', '.join(sorted(set(x))),
        'pWAR_points_mean': 'sum',
        'pWAR_points_sd': lambda x: np.sqrt((x**2).sum()),  # Sum of variances -> sqrt for sd
        'pWAR_wins_mean': 'sum',
        'RoundNumber': 'count'
    }).reset_index()

    pwar_by_season = pwar_by_season.rename(columns={
        'Team': 'Teams',
        'RoundNumber': 'RacesCount'
    })

    # Recompute percentiles from race-level data
    race_level_sums = pwar_by_race.pivot_table(
        index='Driver',
        values=['pWAR_points_p5', 'pWAR_points_p95', 'pWAR_wins_p5', 'pWAR_wins_p95'],
        aggfunc='sum'
    ).reset_index()

    pwar_by_season = pwar_by_season.merge(race_level_sums, on='Driver', how='left')
    pwar_by_season['Season'] = 2025

    # Sort by pWAR
    pwar_by_season = pwar_by_season.sort_values('pWAR_points_mean', ascending=False)

    logger.info(f"Computed pWAR for {len(all_race_results)} races, {len(pwar_by_season)} drivers")

    return pwar_by_race, pwar_by_season
