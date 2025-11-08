"""
Data preparation module for F1 WAR metric.

Handles lap filtering, feature engineering, and encoding for modeling.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_race_laps(
    laps: pd.DataFrame,
    min_quantile: float = 0.01,
    max_quantile: float = 0.99
) -> pd.DataFrame:
    """
    Filter laps to include only valid race laps.

    Removes:
    - Pit in/out laps
    - Non-green flag laps (keep TrackStatus == '1')
    - Inaccurate laps
    - Extreme outliers (per race)

    Parameters
    ----------
    laps : DataFrame
        Raw lap data from FastF1
    min_quantile : float
        Lower quantile for lap time filtering (per race)
    max_quantile : float
        Upper quantile for lap time filtering (per race)

    Returns
    -------
    DataFrame
        Filtered lap data
    """
    logger.info(f"Starting with {len(laps)} total laps")

    # Create a copy to avoid modifying original
    df = laps.copy()

    # Filter to race laps only (not practice/qualifying)
    # This should already be done in data_fetch, but double-check
    initial_count = len(df)

    # Remove pit laps
    df = df[~df['PitInLap'].fillna(False)]
    df = df[~df['PitOutLap'].fillna(False)]
    logger.info(f"After pit lap removal: {len(df)} laps ({initial_count - len(df)} removed)")

    # Keep only green flag laps (TrackStatus == '1')
    if 'TrackStatus' in df.columns:
        df = df[df['TrackStatus'] == '1']
        logger.info(f"After track status filter: {len(df)} laps")

    # Keep only accurate laps
    if 'IsAccurate' in df.columns:
        df = df[df['IsAccurate'] == True]
        logger.info(f"After accuracy filter: {len(df)} laps")

    # Convert LapTime to seconds
    if 'LapTime' in df.columns:
        # LapTime is a timedelta
        df['LapTime_s'] = df['LapTime'].dt.total_seconds()

        # Remove invalid lap times (NaN or zero)
        df = df[df['LapTime_s'].notna()]
        df = df[df['LapTime_s'] > 0]

        # Remove extreme outliers per race
        def filter_outliers_per_race(group):
            lower = group['LapTime_s'].quantile(min_quantile)
            upper = group['LapTime_s'].quantile(max_quantile)
            return group[(group['LapTime_s'] >= lower) & (group['LapTime_s'] <= upper)]

        if 'RoundNumber' in df.columns:
            df = df.groupby('RoundNumber', group_keys=False).apply(filter_outliers_per_race)
        else:
            # Fallback: filter globally
            lower = df['LapTime_s'].quantile(min_quantile)
            upper = df['LapTime_s'].quantile(max_quantile)
            df = df[(df['LapTime_s'] >= lower) & (df['LapTime_s'] <= upper)]

        logger.info(f"After outlier removal: {len(df)} laps")

    logger.info(f"Final filtered dataset: {len(df)} laps")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for modeling.

    Features created:
    - LapTime_demean: lap time demeaned per race
    - TyreLife_sq: squared tyre life
    - GridBucket: grid position buckets
    - Team_RaceID: constructor × race identifier

    Parameters
    ----------
    df : DataFrame
        Filtered lap data

    Returns
    -------
    DataFrame
        Data with engineered features
    """
    df = df.copy()

    # Demean lap times per race (subtract race median)
    if 'RoundNumber' in df.columns and 'LapTime_s' in df.columns:
        df['RaceMedianLapTime'] = df.groupby('RoundNumber')['LapTime_s'].transform('median')
        df['LapTime_demean'] = df['LapTime_s'] - df['RaceMedianLapTime']
        logger.info("Created demeaned lap times (LapTime_demean)")

    # Squared tyre life
    if 'TyreLife' in df.columns:
        df['TyreLife_sq'] = df['TyreLife'] ** 2
        logger.info("Created squared tyre life feature")

    # Grid position buckets
    if 'GridPosition' in df.columns:
        df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
        df['GridBucket'] = pd.cut(
            df['GridPosition'],
            bins=[0, 4, 8, 20, 100],
            labels=['P1-4', 'P5-8', 'P9-20', 'Back'],
            include_lowest=True
        )
        logger.info("Created grid position buckets")

    # Constructor × Race identifier
    if 'Team' in df.columns and 'RoundNumber' in df.columns:
        df['Team_RaceID'] = df['Team'].astype(str) + '_R' + df['RoundNumber'].astype(str)
        logger.info("Created Team_RaceID (constructor × race)")

    # Driver × Track identifier (optional, for random effects)
    if 'Driver' in df.columns and 'EventName' in df.columns:
        df['Driver_Track'] = df['Driver'].astype(str) + '_' + df['EventName'].astype(str)

    # Driver × Constructor identifier
    if 'Driver' in df.columns and 'Team' in df.columns:
        df['Driver_Team'] = df['Driver'].astype(str) + '_' + df['Team'].astype(str)

    # Fill missing weather with race-level medians
    weather_cols = ['AirTemp', 'TrackTemp', 'Humidity']
    for col in weather_cols:
        if col in df.columns:
            df[col] = df.groupby('RoundNumber')[col].transform(
                lambda x: x.fillna(x.median())
            )

    # Ensure Stint is numeric
    if 'Stint' in df.columns:
        df['Stint'] = pd.to_numeric(df['Stint'], errors='coerce').fillna(1).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Encode categorical variables as integer indices for modeling.

    Parameters
    ----------
    df : DataFrame
        Data with features

    Returns
    -------
    df : DataFrame
        Data with encoded categoricals
    encodings : dict
        Mapping dictionaries for each categorical variable
    """
    df = df.copy()
    encodings = {}

    categorical_cols = {
        'Driver': 'DriverID',
        'Team': 'TeamID',
        'EventName': 'TrackID',
        'Team_RaceID': 'TeamRaceID',
        'Compound': 'CompoundID',
        'GridBucket': 'GridBucketID',
        'Driver_Track': 'DriverTrackID',
        'Driver_Team': 'DriverTeamID'
    }

    for col, id_col in categorical_cols.items():
        if col in df.columns:
            # Create encoding
            unique_vals = df[col].dropna().unique()
            encoding = {val: idx for idx, val in enumerate(sorted(unique_vals, key=str))}
            encodings[col] = encoding

            # Apply encoding
            df[id_col] = df[col].map(encoding)

            logger.info(f"Encoded {col}: {len(encoding)} unique values")

    return df, encodings


def prepare_modeling_data(
    laps: pd.DataFrame,
    min_quantile: float = 0.01,
    max_quantile: float = 0.99
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Main preparation pipeline: filter, engineer features, and encode.

    Parameters
    ----------
    laps : DataFrame
        Raw lap data
    min_quantile : float
        Lower quantile for outlier filtering
    max_quantile : float
        Upper quantile for outlier filtering

    Returns
    -------
    prepared_data : DataFrame
        Clean, feature-engineered, encoded data ready for modeling
    encodings : dict
        Encoding mappings for categorical variables
    """
    logger.info("=== Starting data preparation pipeline ===")

    # Step 1: Filter laps
    filtered = filter_race_laps(laps, min_quantile, max_quantile)

    # Step 2: Engineer features
    featured = engineer_features(filtered)

    # Step 3: Encode categoricals
    prepared, encodings = encode_categoricals(featured)

    # Remove rows with missing critical values
    critical_cols = ['LapTime_demean', 'DriverID', 'TeamID', 'TeamRaceID']
    before = len(prepared)
    prepared = prepared.dropna(subset=[c for c in critical_cols if c in prepared.columns])
    after = len(prepared)

    if before > after:
        logger.info(f"Removed {before - after} rows with missing critical values")

    logger.info(f"=== Preparation complete: {len(prepared)} laps ready for modeling ===")

    return prepared, encodings


def create_race_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-driver, per-race summary statistics for simulation.

    Parameters
    ----------
    df : DataFrame
        Prepared lap data

    Returns
    -------
    DataFrame
        Summary with columns: RoundNumber, EventName, Driver, Team, GridPosition,
        MeanLapTime, MedianLapTime, LapCount, etc.
    """
    summary = df.groupby(['RoundNumber', 'EventName', 'Driver', 'Team']).agg({
        'LapTime_s': ['mean', 'median', 'std', 'count'],
        'LapTime_demean': ['mean', 'median'],
        'GridPosition': 'first',
        'DriverID': 'first',
        'TeamID': 'first',
        'TeamRaceID': 'first'
    }).reset_index()

    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in summary.columns]

    # Rename for clarity
    summary = summary.rename(columns={
        'LapTime_s_mean': 'MeanLapTime',
        'LapTime_s_median': 'MedianLapTime',
        'LapTime_s_std': 'StdLapTime',
        'LapTime_s_count': 'LapCount',
        'LapTime_demean_mean': 'MeanLapTime_demean',
        'LapTime_demean_median': 'MedianLapTime_demean',
        'GridPosition_first': 'GridPosition',
        'DriverID_first': 'DriverID',
        'TeamID_first': 'TeamID',
        'TeamRaceID_first': 'TeamRaceID'
    })

    return summary
