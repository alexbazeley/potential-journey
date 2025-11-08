#!/usr/bin/env python3
"""
Main orchestrator script for F1 WAR metric pipeline.

Runs the complete pipeline: data fetch → preparation → modeling → simulation → reporting.

Usage:
    python scripts/run_2025.py --draws 1000 --tune 1000 --sims 10000 --out out/

For all options:
    python scripts/run_2025.py --help
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from war_f1 import data_fetch, prep, models_pace, simulate, report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('war_f1_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='F1 WAR Metric Pipeline for 2025 Season',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data parameters
    parser.add_argument('--year', type=int, default=2025,
                        help='Season year')
    parser.add_argument('--cache', type=Path, default=Path('./cache'),
                        help='FastF1 cache directory')

    # Filtering parameters
    parser.add_argument('--min_lap_quantile', type=float, default=0.01,
                        help='Lower quantile for lap time filtering')
    parser.add_argument('--max_lap_quantile', type=float, default=0.99,
                        help='Upper quantile for lap time filtering')

    # Model parameters
    parser.add_argument('--draws', '--n_draws', type=int, default=1000, dest='n_draws',
                        help='Number of posterior draws per chain')
    parser.add_argument('--tune', '--n_tune', type=int, default=1000, dest='n_tune',
                        help='Number of tuning steps')
    parser.add_argument('--target_accept', type=float, default=0.9,
                        help='Target acceptance probability for NUTS')
    parser.add_argument('--include_driver_track', action='store_true',
                        help='Include Driver×Track random effects')
    parser.add_argument('--include_driver_team', action='store_true',
                        help='Include Driver×Constructor random effects')

    # Simulation parameters
    parser.add_argument('--sims', type=int, default=10000,
                        help='Number of race simulations per configuration')
    parser.add_argument('--n_laps', type=int, default=60,
                        help='Average number of laps per race')
    parser.add_argument('--overtake_difficulty', type=float, default=0.3,
                        help='Track overtake difficulty [0, 1]')
    parser.add_argument('--rep_quantile', '--replacement_quantile', type=float, default=0.25,
                        dest='replacement_quantile',
                        help='Quantile for replacement level')
    parser.add_argument('--n_posterior_draws', type=int, default=500,
                        help='Posterior draws to use for uncertainty in WAR')

    # Output parameters
    parser.add_argument('--out', type=Path, default=Path('./out'),
                        help='Output directory')
    parser.add_argument('--html', type=Path, default=None,
                        help='HTML report output path (default: out/report_2025.html)')

    # Optional features
    parser.add_argument('--include_results_model', action='store_true',
                        help='Include results-based rWAR model (not yet implemented)')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Skip steps (for testing/debugging)
    parser.add_argument('--skip_fetch', action='store_true',
                        help='Skip data fetching (use cached data)')
    parser.add_argument('--skip_modeling', action='store_true',
                        help='Skip modeling (load existing trace)')
    parser.add_argument('--trace_path', type=Path, default=None,
                        help='Path to existing trace file (if skip_modeling)')

    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("F1 WAR METRIC PIPELINE - 2025 SEASON")
    logger.info("=" * 80)

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)

    # Set HTML path if not provided
    if args.html is None:
        args.html = args.out / "report_2025.html"

    # Metadata for reproducibility
    metadata = {
        'pipeline_version': '0.1.0',
        'run_timestamp': datetime.now().isoformat(),
        'year': args.year,
        'parameters': {
            'min_lap_quantile': args.min_lap_quantile,
            'max_lap_quantile': args.max_lap_quantile,
            'n_draws': args.n_draws,
            'n_tune': args.n_tune,
            'target_accept': args.target_accept,
            'n_sims': args.sims,
            'n_laps': args.n_laps,
            'overtake_difficulty': args.overtake_difficulty,
            'replacement_quantile': args.replacement_quantile,
            'n_posterior_draws': args.n_posterior_draws,
            'random_seed': args.seed,
            'include_driver_track': args.include_driver_track,
            'include_driver_team': args.include_driver_team,
        }
    }

    try:
        # Step 1: Data Fetching
        if not args.skip_fetch:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1: DATA FETCHING")
            logger.info("=" * 80)

            raw_data = data_fetch.load_f1_data(args.year, args.cache)
            laps_df = raw_data['laps']

            if laps_df.empty:
                logger.error("No lap data fetched. Exiting.")
                return 1

            logger.info(f"Fetched {len(laps_df)} laps from {args.year} season")
        else:
            logger.info("Skipping data fetch (using cached data)")
            # Would need to load from disk in real implementation
            raise NotImplementedError("Skip fetch requires loading cached data")

        # Step 2: Data Preparation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DATA PREPARATION")
        logger.info("=" * 80)

        prepared_data, encodings = prep.prepare_modeling_data(
            laps_df,
            args.min_lap_quantile,
            args.max_lap_quantile
        )

        if prepared_data.empty:
            logger.error("No data after preparation. Exiting.")
            return 1

        logger.info(f"Prepared {len(prepared_data)} laps for modeling")

        # Create race summaries for simulation
        race_summaries = prep.create_race_summary(prepared_data)
        logger.info(f"Created summaries for {len(race_summaries)} driver-race combinations")

        # Step 3: Pace Model
        if not args.skip_modeling:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: PACE MODEL (pWAR)")
            logger.info("=" * 80)

            # Build model
            model = models_pace.build_pace_model(
                prepared_data,
                args.include_driver_track,
                args.include_driver_team
            )

            # Fit model
            trace = models_pace.fit_pace_model(
                model,
                args.n_draws,
                args.n_tune,
                args.target_accept,
                args.seed
            )

            # Save trace
            trace_path = args.out / "pace_trace_2025.nc"
            models_pace.save_trace(trace, trace_path)

            # Run diagnostics
            diagnostics = models_pace.diagnose_trace(trace, args.out / "diagnostics.csv")

        else:
            logger.info("Skipping modeling (loading existing trace)")
            if args.trace_path is None:
                args.trace_path = args.out / "pace_trace_2025.nc"

            trace = models_pace.load_trace(args.trace_path)
            diagnostics = models_pace.diagnose_trace(trace)

        # Extract driver abilities
        driver_abilities = models_pace.extract_driver_abilities(trace, encodings)
        logger.info(f"\nTop 5 drivers by pace ability:")
        for _, row in driver_abilities.head(5).iterrows():
            logger.info(f"  {row['Driver']}: {row['ability_mean']:.4f} ± {row['ability_sd']:.4f} sec/lap")

        # Step 4: Race Simulation & pWAR Calculation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: RACE SIMULATION & pWAR CALCULATION")
        logger.info("=" * 80)

        pwar_by_race, pwar_by_season = simulate.compute_pwar_all_races(
            race_summaries,
            trace,
            encodings,
            args.sims,
            args.n_laps,
            args.overtake_difficulty,
            args.replacement_quantile,
            args.n_posterior_draws,
            args.seed
        )

        logger.info(f"\nTop 5 drivers by pWAR:")
        for _, row in pwar_by_season.head(5).iterrows():
            logger.info(f"  {row['Driver']}: {row['pWAR_points_mean']:.2f} points ({row['pWAR_wins_mean']:.2f} wins)")

        # Step 5: Results Model (optional)
        if args.include_results_model:
            logger.warning("Results-based model (rWAR) not yet implemented. Skipping.")

        # Step 6: Reporting
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: GENERATING REPORTS")
        logger.info("=" * 80)

        # Save CSVs
        report.save_csv_outputs(pwar_by_race, pwar_by_season, args.out)

        # Generate HTML report
        report.generate_html_report(
            pwar_by_race,
            pwar_by_season,
            driver_abilities,
            diagnostics,
            args.html,
            metadata
        )

        # Save metadata
        report.save_metadata(metadata, args.out / "run_metadata.json")

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nOutputs saved to: {args.out}")
        logger.info(f"  - CSV files: pwar_by_race_2025.csv, pwar_by_season_2025.csv")
        logger.info(f"  - Posterior trace: pace_trace_2025.nc")
        logger.info(f"  - HTML report: {args.html}")
        logger.info(f"  - Metadata: run_metadata.json")

        return 0

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
