"""
Report generation module for F1 WAR metric.

Creates CSV outputs, visualizations, and HTML/Markdown reports.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_csv_outputs(
    pwar_by_race: pd.DataFrame,
    pwar_by_season: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Save pWAR results to CSV files.

    Parameters
    ----------
    pwar_by_race : DataFrame
        Per-race pWAR results
    pwar_by_season : DataFrame
        Season totals
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-race CSV
    race_csv = output_dir / "pwar_by_race_2025.csv"
    race_cols = [
        'Season', 'RoundNumber', 'EventName', 'Driver', 'Team', 'GridPosition',
        'pWAR_points_mean', 'pWAR_points_p5', 'pWAR_points_p95',
        'pWAR_wins_mean', 'pWAR_wins_p5', 'pWAR_wins_p95'
    ]
    pwar_by_race[[c for c in race_cols if c in pwar_by_race.columns]].to_csv(race_csv, index=False)
    logger.info(f"Saved per-race pWAR to {race_csv}")

    # Per-season CSV
    season_csv = output_dir / "pwar_by_season_2025.csv"
    season_cols = [
        'Season', 'Driver', 'Teams', 'RacesCount',
        'pWAR_points_mean', 'pWAR_points_p5', 'pWAR_points_p95',
        'pWAR_wins_mean', 'pWAR_wins_p5', 'pWAR_wins_p95'
    ]
    pwar_by_season[[c for c in season_cols if c in pwar_by_season.columns]].to_csv(season_csv, index=False)
    logger.info(f"Saved season pWAR to {season_csv}")


def plot_top_drivers(
    pwar_by_season: pd.DataFrame,
    output_path: Path,
    top_n: int = 10
) -> None:
    """
    Create bar chart of top N drivers by pWAR.

    Parameters
    ----------
    pwar_by_season : DataFrame
        Season pWAR results
    output_path : Path
        Output file path
    top_n : int
        Number of top drivers to show
    """
    top_drivers = pwar_by_season.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))

    drivers = top_drivers['Driver'].values
    means = top_drivers['pWAR_points_mean'].values
    p5 = top_drivers['pWAR_points_p5'].values
    p95 = top_drivers['pWAR_points_p95'].values

    errors = np.array([means - p5, p95 - means])

    ax.barh(range(len(drivers)), means, xerr=errors, capsize=5, alpha=0.7)
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers)
    ax.invert_yaxis()
    ax.set_xlabel('pWAR (Championship Points Above Replacement)')
    ax.set_title(f'Top {top_n} Drivers by pWAR - 2025 Season')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved top drivers chart to {output_path}")


def plot_driver_abilities(
    driver_abilities: pd.DataFrame,
    output_path: Path,
    top_n: int = 15
) -> None:
    """
    Create error bar chart of driver abilities (pace in sec/lap).

    Parameters
    ----------
    driver_abilities : DataFrame
        Driver ability posteriors
    output_path : Path
        Output file path
    top_n : int
        Number of drivers to show
    """
    # Sort by ability (negative = faster)
    top_drivers = driver_abilities.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))

    drivers = top_drivers['Driver'].values
    means = top_drivers['ability_mean'].values
    p5 = top_drivers['ability_p5'].values
    p95 = top_drivers['ability_p95'].values

    errors = np.array([means - p5, p95 - means])

    ax.barh(range(len(drivers)), means, xerr=errors, capsize=5, alpha=0.7, color='steelblue')
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers)
    ax.invert_yaxis()
    ax.set_xlabel('Driver Ability (sec/lap, negative = faster)')
    ax.set_title(f'Top {top_n} Drivers by Pace Ability - 2025 Season')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved driver abilities chart to {output_path}")


def plot_per_race_contributions(
    pwar_by_race: pd.DataFrame,
    output_path: Path,
    top_n: int = 10
) -> None:
    """
    Create stacked bar chart showing per-race pWAR contributions.

    Parameters
    ----------
    pwar_by_race : DataFrame
        Per-race pWAR results
    output_path : Path
        Output file path
    top_n : int
        Number of top drivers to show
    """
    # Get top drivers by total pWAR
    top_drivers = pwar_by_race.groupby('Driver')['pWAR_points_mean'].sum().nlargest(top_n).index

    # Pivot to get races as columns
    pivot = pwar_by_race[pwar_by_race['Driver'].isin(top_drivers)].pivot_table(
        index='Driver',
        columns='EventName',
        values='pWAR_points_mean',
        fill_value=0
    )

    # Reorder drivers by total
    pivot = pivot.loc[top_drivers]

    fig, ax = plt.subplots(figsize=(14, 8))

    pivot.plot(kind='barh', stacked=True, ax=ax, legend=False, width=0.8)

    ax.set_xlabel('Cumulative pWAR (Championship Points)')
    ax.set_ylabel('Driver')
    ax.set_title(f'Per-Race pWAR Contributions - Top {top_n} Drivers')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved per-race contributions chart to {output_path}")


def generate_html_report(
    pwar_by_race: pd.DataFrame,
    pwar_by_season: pd.DataFrame,
    driver_abilities: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_path: Path,
    metadata: Dict
) -> None:
    """
    Generate HTML report with tables, charts, and methodology.

    Parameters
    ----------
    pwar_by_race : DataFrame
        Per-race pWAR
    pwar_by_season : DataFrame
        Season pWAR
    driver_abilities : DataFrame
        Driver ability posteriors
    diagnostics : DataFrame
        Model diagnostics
    output_path : Path
        Output HTML file path
    metadata : dict
        Run metadata (parameters, versions, etc.)
    """
    logger.info("Generating HTML report...")

    # Create charts
    charts_dir = output_path.parent / "charts"
    charts_dir.mkdir(exist_ok=True)

    plot_top_drivers(pwar_by_season, charts_dir / "top_drivers.png")
    plot_driver_abilities(driver_abilities, charts_dir / "driver_abilities.png")
    plot_per_race_contributions(pwar_by_race, charts_dir / "per_race_contributions.png")

    # Top 10 leaderboard
    top10 = pwar_by_season.head(10)[['Driver', 'Teams', 'RacesCount', 'pWAR_points_mean', 'pWAR_wins_mean']].copy()
    top10['pWAR_points_mean'] = top10['pWAR_points_mean'].round(2)
    top10['pWAR_wins_mean'] = top10['pWAR_wins_mean'].round(2)
    top10_html = top10.to_html(index=False, classes='table')

    # Diagnostics summary
    # Compute values first to avoid f-string formatting issues
    n_params = len(diagnostics)
    high_rhat_count = (diagnostics['r_hat'] > 1.05).sum() if 'r_hat' in diagnostics.columns else 'N/A'
    mean_ess = f"{diagnostics['ess_bulk'].mean():.0f}" if 'ess_bulk' in diagnostics.columns else 'N/A'

    diag_summary = f"""
    <p><strong>Convergence diagnostics:</strong></p>
    <ul>
        <li>Parameters checked: {n_params}</li>
        <li>R-hat > 1.05: {high_rhat_count}</li>
        <li>Mean ESS: {mean_ess}</li>
    </ul>
    """

    # Build HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>F1 WAR Report - 2025 Season</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #e10600; }}
            h2 {{ color: #333; border-bottom: 2px solid #e10600; padding-bottom: 5px; }}
            .table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .table th {{ background-color: #e10600; color: white; }}
            .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            .metadata {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #e10600; }}
        </style>
    </head>
    <body>
        <h1>Formula 1 Wins Above Replacement (pWAR) Report</h1>
        <h2>2025 Season</h2>

        <h2>Executive Summary</h2>
        <p>
            This report presents the <strong>pace-based Wins Above Replacement (pWAR)</strong> metric
            for the 2025 Formula 1 season. pWAR measures the championship points a driver contributes
            above what a replacement-level driver would achieve in the same seat and conditions.
        </p>

        <h3>Top 10 Leaderboard</h3>
        {top10_html}

        <h2>Visualizations</h2>

        <h3>Top Drivers by pWAR</h3>
        <img src="charts/top_drivers.png" alt="Top Drivers">

        <h3>Driver Pace Abilities (sec/lap)</h3>
        <img src="charts/driver_abilities.png" alt="Driver Abilities">

        <h3>Per-Race Contributions</h3>
        <img src="charts/per_race_contributions.png" alt="Per-Race Contributions">

        <h2>Methodology</h2>

        <h3>Overview</h3>
        <p>
            The pWAR metric isolates driver skill from car performance using a Bayesian hierarchical model
            of lap times. The model accounts for:
        </p>
        <ul>
            <li><strong>Car performance:</strong> Constructor × Race random effects absorb car/setup/strategy variance</li>
            <li><strong>Driver ability:</strong> Driver random effects capture individual pace (the key metric)</li>
            <li><strong>Context:</strong> Fixed effects for tyre compound, tyre age, weather, stint, grid position</li>
        </ul>

        <h3>Data Preparation</h3>
        <p>Lap filtering criteria:</p>
        <ul>
            <li>Race laps only (no practice/qualifying)</li>
            <li>Excluded pit in/out laps</li>
            <li>Green flag laps only (TrackStatus = 1)</li>
            <li>Accurate timing only (IsAccurate = True)</li>
            <li>Outliers removed: kept laps within [{metadata.get('min_lap_quantile', 0.01)*100:.0f}th, {metadata.get('max_lap_quantile', 0.99)*100:.0f}th] percentile per race</li>
        </ul>

        <h3>Model Specification</h3>
        <p>
            <strong>Outcome:</strong> Demeaned lap time (LapTime - race median)<br>
            <strong>Random effects:</strong>
        </p>
        <ul>
            <li>Driver ability ~ Normal(0, σ_driver)</li>
            <li>Constructor × Race ~ Normal(0, σ_team_race)</li>
        </ul>
        <p><strong>Fixed effects:</strong> Tyre compound, tyre life (+ squared), weather (air/track temp, humidity), stint, grid bucket</p>

        <h3>Replacement Level</h3>
        <p>
            Replacement level defined as the <strong>{metadata.get('replacement_quantile', 0.25)*100:.0f}th percentile</strong>
            of driver abilities per posterior draw.
        </p>

        <h3>Race Simulation</h3>
        <p>
            For each race, we simulate {metadata.get('n_sims', 10000):,} race outcomes using driver pace abilities
            and grid positions. Overtaking probability depends on pace gaps and a track-difficulty parameter
            (default: {metadata.get('overtake_difficulty', 0.3)}).
        </p>
        <p>
            pWAR = expected points with actual driver - expected points with replacement-level driver.
        </p>

        <h2>Model Diagnostics</h2>
        {diag_summary}

        <h2>Appendix: Run Metadata</h2>
        <div class="metadata">
            <pre>{json.dumps(metadata, indent=2)}</pre>
        </div>

        <hr>
        <p><em>Generated by war-f1 pipeline</em></p>
    </body>
    </html>
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)

    logger.info(f"Saved HTML report to {output_path}")


def save_metadata(metadata: Dict, output_path: Path) -> None:
    """
    Save run metadata as JSON.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary
    output_path : Path
        Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, indent=2, fp=f)

    logger.info(f"Saved metadata to {output_path}")
