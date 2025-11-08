"""
Bayesian hierarchical model for driver pace (pWAR).

Models demeaned lap times with random effects for:
- Driver ability (the key metric)
- Constructor × Race (car performance per weekend)
- Optional: Driver × Track, Driver × Constructor

Fixed effects for tyres, weather, stint, grid position.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

logger = logging.getLogger(__name__)


def build_pace_model(
    df: pd.DataFrame,
    include_driver_track: bool = False,
    include_driver_team: bool = False
) -> pm.Model:
    """
    Build Bayesian hierarchical model for lap times.

    Model:
        LapTime_demean ~ Normal(μ, σ)
        μ = intercept + fixed_effects + random_effects

    Random effects:
        - Driver ability (primary metric of interest)
        - Constructor × Race (car/strategy/setup per weekend)
        - Optionally: Driver × Track, Driver × Constructor

    Fixed effects:
        - Tyre compound, tyre life, tyre life squared
        - Weather: air temp, track temp, humidity
        - Stint number
        - Grid bucket

    Parameters
    ----------
    df : DataFrame
        Prepared lap data with encoded features
    include_driver_track : bool
        Include Driver × Track random effect
    include_driver_team : bool
        Include Driver × Constructor random effect

    Returns
    -------
    model : pm.Model
        PyMC model
    """
    logger.info("Building Bayesian hierarchical model...")

    # Extract data
    y = df['LapTime_demean'].values
    driver_id = df['DriverID'].values.astype(int)
    team_race_id = df['TeamRaceID'].values.astype(int)

    # Number of unique levels
    n_drivers = df['DriverID'].nunique()
    n_team_races = df['TeamRaceID'].nunique()

    logger.info(f"  Outcome: {len(y)} laps")
    logger.info(f"  Drivers: {n_drivers}")
    logger.info(f"  Team×Races: {n_team_races}")

    # Fixed effects design matrix
    fixed_cols = []

    # Tyre effects
    if 'TyreLife' in df.columns:
        fixed_cols.append('TyreLife')
    if 'TyreLife_sq' in df.columns:
        fixed_cols.append('TyreLife_sq')

    # Weather
    for col in ['AirTemp', 'TrackTemp', 'Humidity']:
        if col in df.columns and df[col].notna().sum() > 0:
            fixed_cols.append(col)

    # Stint
    if 'Stint' in df.columns:
        fixed_cols.append('Stint')

    # Compound (one-hot encoded)
    if 'CompoundID' in df.columns:
        # Create dummies
        compound_dummies = pd.get_dummies(df['CompoundID'], prefix='Compound', dtype=float)
        # Drop first to avoid collinearity
        compound_dummies = compound_dummies.iloc[:, 1:]
        fixed_cols.extend(compound_dummies.columns.tolist())
        df = pd.concat([df, compound_dummies], axis=1)

    # Grid bucket (one-hot)
    if 'GridBucketID' in df.columns:
        grid_dummies = pd.get_dummies(df['GridBucketID'], prefix='Grid', dtype=float)
        grid_dummies = grid_dummies.iloc[:, 1:]
        fixed_cols.extend(grid_dummies.columns.tolist())
        df = pd.concat([df, grid_dummies], axis=1)

    # Standardize fixed effects
    X_fixed = df[fixed_cols].fillna(0).values
    X_mean = X_fixed.mean(axis=0)
    X_std = X_fixed.std(axis=0)
    X_std[X_std == 0] = 1.0  # Avoid division by zero
    X_fixed = (X_fixed - X_mean) / X_std

    logger.info(f"  Fixed effects: {len(fixed_cols)} features")

    # Optional random effects
    if include_driver_track and 'DriverTrackID' in df.columns:
        driver_track_id = df['DriverTrackID'].values.astype(int)
        n_driver_tracks = df['DriverTrackID'].nunique()
        logger.info(f"  Driver×Track: {n_driver_tracks} combinations")
    else:
        driver_track_id = None
        n_driver_tracks = 0

    if include_driver_team and 'DriverTeamID' in df.columns:
        driver_team_id = df['DriverTeamID'].values.astype(int)
        n_driver_teams = df['DriverTeamID'].nunique()
        logger.info(f"  Driver×Team: {n_driver_teams} combinations")
    else:
        driver_team_id = None
        n_driver_teams = 0

    # Build PyMC model
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal('intercept', mu=0, sigma=1)

        # Fixed effects coefficients
        beta = pm.Normal('beta', mu=0, sigma=0.5, shape=len(fixed_cols))

        # Random effects standard deviations (half-normal priors)
        sigma_driver = pm.HalfNormal('sigma_driver', sigma=0.5)
        sigma_team_race = pm.HalfNormal('sigma_team_race', sigma=1.0)

        # Random effects
        driver_ability = pm.Normal('driver_ability', mu=0, sigma=sigma_driver, shape=n_drivers)
        team_race_effect = pm.Normal('team_race_effect', mu=0, sigma=sigma_team_race, shape=n_team_races)

        # Optional random effects
        if include_driver_track and n_driver_tracks > 0:
            sigma_driver_track = pm.HalfNormal('sigma_driver_track', sigma=0.3)
            driver_track_effect = pm.Normal('driver_track_effect', mu=0, sigma=sigma_driver_track, shape=n_driver_tracks)
        else:
            driver_track_effect = None

        if include_driver_team and n_driver_teams > 0:
            sigma_driver_team = pm.HalfNormal('sigma_driver_team', sigma=0.3)
            driver_team_effect = pm.Normal('driver_team_effect', mu=0, sigma=sigma_driver_team, shape=n_driver_teams)
        else:
            driver_team_effect = None

        # Residual standard deviation
        sigma = pm.HalfNormal('sigma', sigma=2.0)

        # Linear predictor
        mu = intercept + pm.math.dot(X_fixed, beta)
        mu += driver_ability[driver_id]
        mu += team_race_effect[team_race_id]

        if driver_track_effect is not None:
            mu += driver_track_effect[driver_track_id]

        if driver_team_effect is not None:
            mu += driver_team_effect[driver_team_id]

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    logger.info("Model built successfully")

    return model


def fit_pace_model(
    model: pm.Model,
    n_draws: int = 1000,
    n_tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: int = 42
) -> az.InferenceData:
    """
    Fit the pace model using NUTS sampler.

    Parameters
    ----------
    model : pm.Model
        PyMC model
    n_draws : int
        Number of posterior draws per chain
    n_tune : int
        Number of tuning steps
    target_accept : float
        Target acceptance probability for NUTS
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    trace : InferenceData
        ArviZ InferenceData object with posterior samples
    """
    logger.info(f"Fitting model with {n_draws} draws, {n_tune} tuning steps...")

    with model:
        trace = pm.sample(
            draws=n_draws,
            tune=n_tune,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True
        )

    logger.info("Sampling complete")

    return trace


def extract_driver_abilities(
    trace: az.InferenceData,
    encodings: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Extract driver ability posteriors and compute summary statistics.

    Parameters
    ----------
    trace : InferenceData
        Posterior samples
    encodings : dict
        Encoding mappings (must include 'Driver')

    Returns
    -------
    DataFrame
        Driver abilities with columns:
        Driver, ability_mean, ability_sd, ability_p5, ability_p95
    """
    # Get posterior samples for driver_ability
    driver_ability_samples = trace.posterior['driver_ability'].values  # shape: (chains, draws, n_drivers)

    # Flatten chains and draws
    driver_ability_samples = driver_ability_samples.reshape(-1, driver_ability_samples.shape[-1])  # (n_samples, n_drivers)

    # Reverse encoding
    driver_encoding = encodings.get('Driver', {})
    id_to_driver = {v: k for k, v in driver_encoding.items()}

    # Compute summary statistics
    results = []
    for driver_id in range(driver_ability_samples.shape[1]):
        samples = driver_ability_samples[:, driver_id]

        results.append({
            'DriverID': driver_id,
            'Driver': id_to_driver.get(driver_id, f'Driver_{driver_id}'),
            'ability_mean': samples.mean(),
            'ability_sd': samples.std(),
            'ability_p5': np.percentile(samples, 5),
            'ability_p25': np.percentile(samples, 25),
            'ability_p50': np.percentile(samples, 50),
            'ability_p75': np.percentile(samples, 75),
            'ability_p95': np.percentile(samples, 95)
        })

    df = pd.DataFrame(results).sort_values('ability_mean')

    logger.info(f"Extracted abilities for {len(df)} drivers")

    return df


def save_trace(trace: az.InferenceData, output_path: Path) -> None:
    """
    Save posterior trace to NetCDF format.

    Parameters
    ----------
    trace : InferenceData
        Posterior samples
    output_path : Path
        Output file path (.nc)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_netcdf(str(output_path))
    logger.info(f"Saved trace to {output_path}")


def load_trace(input_path: Path) -> az.InferenceData:
    """
    Load posterior trace from NetCDF format.

    Parameters
    ----------
    input_path : Path
        Input file path (.nc)

    Returns
    -------
    InferenceData
        Loaded posterior samples
    """
    trace = az.from_netcdf(str(input_path))
    logger.info(f"Loaded trace from {input_path}")
    return trace


def diagnose_trace(trace: az.InferenceData, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Run diagnostics on posterior samples.

    Parameters
    ----------
    trace : InferenceData
        Posterior samples
    output_path : Path, optional
        If provided, save diagnostics to CSV

    Returns
    -------
    DataFrame
        Summary statistics including R-hat and ESS
    """
    logger.info("Running diagnostics...")

    summary = az.summary(trace, var_names=['~driver_ability', '~team_race_effect'])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path)
        logger.info(f"Saved diagnostics to {output_path}")

    # Check for convergence issues
    if 'r_hat' in summary.columns:
        high_rhat = summary[summary['r_hat'] > 1.05]
        if len(high_rhat) > 0:
            logger.warning(f"WARNING: {len(high_rhat)} parameters have R-hat > 1.05")

    return summary
