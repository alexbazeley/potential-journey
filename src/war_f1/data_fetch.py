"""
Data fetching module for Formula 1 race data using FastF1.

This module handles downloading race data, caching it locally, and
returning structured dataframes for further processing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import fastf1
import pandas as pd
from tqdm import tqdm

# Suppress FastF1 warnings
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('fastf1').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def setup_fastf1_cache(cache_dir: Path = Path("./cache")) -> None:
    """
    Configure FastF1 cache directory for storing downloaded data.

    Parameters
    ----------
    cache_dir : Path
        Directory to store cached FastF1 data.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    logger.info(f"FastF1 cache enabled at: {cache_dir}")


def fetch_race_session(
    year: int,
    event: Union[int, str],
    session: str = "R"
) -> Optional[fastf1.core.Session]:
    """
    Fetch a single race session with error handling.

    Parameters
    ----------
    year : int
        Season year (e.g., 2025)
    event : int or str
        Event number or name
    session : str
        Session identifier ('R' for race, 'Q' for qualifying, etc.)

    Returns
    -------
    Session or None
        FastF1 Session object, or None if loading failed.
    """
    try:
        session_obj = fastf1.get_session(year, event, session)
        session_obj.load()
        return session_obj
    except Exception as e:
        logger.warning(f"Failed to load {year} event {event} session {session}: {e}")
        return None


def fetch_season_races(year: int = 2025) -> Tuple[List[fastf1.core.Session], pd.DataFrame]:
    """
    Fetch all race sessions for a given season.

    Parameters
    ----------
    year : int
        Season year to fetch

    Returns
    -------
    sessions : List[Session]
        List of loaded FastF1 Session objects
    schedule : DataFrame
        Season schedule with event information
    """
    logger.info(f"Fetching season schedule for {year}...")

    try:
        schedule = fastf1.get_event_schedule(year)
        # Filter to only race events (exclude testing, etc.)
        race_schedule = schedule[schedule['EventFormat'] != 'testing'].copy()
    except Exception as e:
        logger.error(f"Failed to fetch schedule for {year}: {e}")
        return [], pd.DataFrame()

    sessions = []
    logger.info(f"Loading {len(race_schedule)} race sessions for {year}...")

    for idx, event_info in tqdm(race_schedule.iterrows(), total=len(race_schedule), desc="Loading races"):
        event_name = event_info.get('EventName', event_info.get('OfficialEventName', f"Event {idx+1}"))
        round_num = event_info.get('RoundNumber', idx + 1)

        session = fetch_race_session(year, round_num, "R")

        if session is not None:
            sessions.append(session)
            logger.info(f"  ✓ Loaded: {event_name}")
        else:
            logger.warning(f"  ✗ Skipped: {event_name}")

    logger.info(f"Successfully loaded {len(sessions)}/{len(race_schedule)} races")

    return sessions, race_schedule


def extract_laps_data(sessions: List[fastf1.core.Session]) -> pd.DataFrame:
    """
    Extract lap-level data from all sessions into a single dataframe.

    Parameters
    ----------
    sessions : List[Session]
        List of FastF1 Session objects

    Returns
    -------
    DataFrame
        Combined lap data with race metadata
    """
    all_laps = []

    for session in sessions:
        try:
            laps = session.laps

            # Add race metadata
            laps = laps.copy()
            laps['Season'] = session.event['EventDate'].year
            laps['RoundNumber'] = session.event.get('RoundNumber', 0)
            laps['EventName'] = session.event.get('EventName', session.event.get('OfficialEventName', 'Unknown'))
            laps['Country'] = session.event.get('Country', 'Unknown')
            laps['Location'] = session.event.get('Location', 'Unknown')

            # Add weather data (session-level)
            try:
                weather = session.weather_data
                if weather is not None and len(weather) > 0:
                    # Get median weather for the race
                    laps['AirTemp'] = weather['AirTemp'].median()
                    laps['TrackTemp'] = weather['TrackTemp'].median()
                    laps['Humidity'] = weather['Humidity'].median()
                else:
                    laps['AirTemp'] = None
                    laps['TrackTemp'] = None
                    laps['Humidity'] = None
            except:
                laps['AirTemp'] = None
                laps['TrackTemp'] = None
                laps['Humidity'] = None

            all_laps.append(laps)

        except Exception as e:
            logger.warning(f"Failed to extract laps from {session.event.get('EventName', 'unknown')}: {e}")
            continue

    if not all_laps:
        logger.error("No lap data extracted from any session!")
        return pd.DataFrame()

    combined = pd.concat(all_laps, ignore_index=True)
    logger.info(f"Extracted {len(combined)} total laps from {len(all_laps)} races")

    return combined


def fetch_results_data(sessions: List[fastf1.core.Session]) -> pd.DataFrame:
    """
    Extract race results (finishing positions) from sessions.

    Parameters
    ----------
    sessions : List[Session]
        List of FastF1 Session objects

    Returns
    -------
    DataFrame
        Race results with finishing positions and points
    """
    all_results = []

    for session in sessions:
        try:
            results = session.results

            # Add race metadata
            results = results.copy()
            results['Season'] = session.event['EventDate'].year
            results['RoundNumber'] = session.event.get('RoundNumber', 0)
            results['EventName'] = session.event.get('EventName', session.event.get('OfficialEventName', 'Unknown'))
            results['Country'] = session.event.get('Country', 'Unknown')

            all_results.append(results)

        except Exception as e:
            logger.warning(f"Failed to extract results from {session.event.get('EventName', 'unknown')}: {e}")
            continue

    if not all_results:
        logger.error("No results data extracted!")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(f"Extracted results for {len(combined)} driver-race combinations")

    return combined


def load_f1_data(
    year: int = 2025,
    cache_dir: Path = Path("./cache")
) -> Dict[str, pd.DataFrame]:
    """
    Main entry point: load all F1 data for a season.

    Parameters
    ----------
    year : int
        Season year
    cache_dir : Path
        Cache directory for FastF1

    Returns
    -------
    dict
        Dictionary containing:
        - 'laps': lap-level timing data
        - 'results': race results
        - 'schedule': season schedule
    """
    setup_fastf1_cache(cache_dir)

    sessions, schedule = fetch_season_races(year)

    if not sessions:
        logger.error(f"No sessions loaded for {year}")
        return {'laps': pd.DataFrame(), 'results': pd.DataFrame(), 'schedule': schedule}

    laps = extract_laps_data(sessions)
    results = fetch_results_data(sessions)

    return {
        'laps': laps,
        'results': results,
        'schedule': schedule
    }
