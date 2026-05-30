"""Dataset module for loading and processing PFF data."""

import gandula
from src.tracking import pff_frames_to_dataframe
from gandula.features.pff import add_ball_speed, add_players_speed
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from src.events import parse_names, parse_events
from src.tracking import change_events_side

# Constants
ROOT_DIR = Path(__file__).parent.parent
ALLOWED_COMPETITIONS = {"PL", "UCL", "WC", "BR"}
ALLOWED_SEASONS = {"23", "24", "22-23", "23-24", "24-25"}

class PFFDataset:
    """
    Dataset class for loading and processing PFF data.

    Handles loading of tracking and event data, processing, and storing
    intermediate results for efficient reuse.

    Attributes:
        competition: Competition identifier (e.g., 'PL', 'UCL')
        season: Season identifier (e.g., '23-24', '24-25')
        match_ids: List of available match IDs
        players: List of player DataFrames for each loaded match
        tracking: List of tracking DataFrames for each loaded match
        events: List of event DataFrames for each loaded match
    """

    def __init__(self, competition: str, season: str):
        """
        Initialize the PFFDataset.

        Args:
            competition: Competition code (PL, UCL, WC, BR)
            season: Season code (e.g., '23-24', '24-25')

        Raises:
            ValueError: If competition or season is not valid
        """
        self._validate_inputs(competition, season)

        self.competition = competition
        self.season = season

        # Setup directory structure
        self.data_path = ROOT_DIR / "data"
        self.tracking_path = self.data_path / "raw" / competition / season / "tracking"
        self.events_path = self.data_path / "raw" / competition / season / "events"
        self.save_path = self.data_path / "interim" / competition / season

        self._create_directories()
        self.match_ids = self._discover_match_ids()

        # Data storage
        self.players: list[pd.DataFrame] = []
        self.tracking: list[pd.DataFrame] = []
        self.events: list[pd.DataFrame] = []

        self.data_version = 'v2'

    def _validate_inputs(self, competition: str, season: str) -> None:
        """Validate competition and season inputs."""
        if competition not in ALLOWED_COMPETITIONS:
            raise ValueError(
                f"Competition must be one of {ALLOWED_COMPETITIONS}, got '{competition}'"
            )
        if season not in ALLOWED_SEASONS:
            raise ValueError(
                f"Season must be one of {ALLOWED_SEASONS}, got '{season}'"
            )

    def _create_directories(self) -> None:
        """Create necessary directory structure."""
        for path in [self.data_path, self.tracking_path, self.events_path, self.save_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _discover_match_ids(self) -> list[str]:
        """Discover available match IDs from tracking or events directories."""
        # Try tracking directory first
        if list(self.tracking_path.iterdir()):
            return [f.stem.split('.')[0] for f in self.tracking_path.iterdir()]

        # Fall back to events directory
        return [f.stem.split('_')[0] for f in self.events_path.iterdir()]

    def load_data(
        self,
        n_matches: Optional[int] = None,
        match_ids: Optional[list[str]] = None,
        add_velocity: bool = False,
        event_type: str = 'all',
        save: bool = True,
        overwrite: bool = False,
        filter_tracking_to_event_frames: bool = False,
    ) -> None:
        """
        Load and process data for specified matches.

        Args:
            n_matches: Number of matches to load (from the start of match_ids list)
            match_ids: Specific match IDs to load (overrides n_matches)
            add_velocity: Whether to calculate velocity features
            event_type: Type of events to filter ('shot', 'pass', 'carry', or 'all')
            save: Whether to save processed data
            overwrite: Whether to overwrite existing processed data
        """
        matches_to_load = self._determine_matches_to_load(n_matches, match_ids)

        self._load_data(
            matches_to_load,
            add_velocity,
            event_type,
            save,
            overwrite,
            filter_tracking_to_event_frames,
        )

    def _load_data(
        self,
        matches_to_load: list[str],
        add_velocity: bool,
        event_type: str,
        save: bool,
        overwrite: bool,
        filter_tracking_to_event_frames: bool,
    ) -> None:
        """Load data sequentially (original implementation)."""
        for match_id in tqdm(matches_to_load, desc="Loading matches"):
            try:
                if self._can_load_cached(match_id) and not overwrite:
                    try:
                        tracking_df, events_df, players_info = self._load_cached_data(match_id)
                    except ValueError:
                        tracking_df, events_df, players_info = self._process_match(
                            match_id,
                            add_velocity,
                            event_type,
                            save,
                            filter_tracking_to_event_frames,
                        )
                else:
                    tracking_df, events_df, players_info = self._process_match(
                        match_id,
                        add_velocity,
                        event_type,
                        save,
                        filter_tracking_to_event_frames,
                    )

                self.players.append(players_info)
                self.tracking.append(tracking_df)
                self.events.append(events_df)
            except Exception as e:
                import traceback
                print(f"Error processing match {match_id}: {e}")
                print(f"Full traceback:")
                traceback.print_exc()
                continue

    def _determine_matches_to_load(
        self,
        n_matches: Optional[int],
        match_ids: Optional[list[str]]
    ) -> list[str]:
        """Determine which matches to load based on parameters."""
        if match_ids is not None:
            return match_ids
        if n_matches is not None:
            return self.match_ids[:n_matches]
        return self.match_ids

    def _can_load_cached(self, match_id: str) -> bool:
        """Check if cached data exists for a match."""
        match_dir = self.save_path / match_id
        return (match_dir / "events.parquet").exists()

    def _load_cached_data(self, match_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cached processed data for a match."""
        match_dir = self.save_path / match_id
        tracking_df = pd.read_parquet(match_dir / "tracking.parquet")
        events_df = pd.read_parquet(match_dir / "events.parquet")
        players_info = pd.read_parquet(match_dir / "players.parquet")

        if 'data_version' not in events_df.columns or not (events_df['data_version'] == self.data_version).all():
            raise ValueError(f"Cached events version mismatch for match {match_id}")

        return tracking_df, events_df, players_info

    def _process_match(
        self,
        match_id: str,
        add_velocity: bool,
        event_type: str,
        save: bool,
        filter_tracking_to_event_frames: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process a single match from raw data."""
        # Load raw data
        metadata_df, tracking_df = self.load_tracking(match_id, add_velocity=add_velocity)
        events_df = self.load_events(match_id)

        # Process events and get player info
        events_df, players_info = self._process_events(events_df, event_type=event_type)

        # Merge metadata with events
        events_df = self._merge_event_metadata(events_df, metadata_df)

        # Keep continuous tracking by default. Event-frame filtering is optional.
        if filter_tracking_to_event_frames:
            tracking_df = self._filter_tracking_frames(tracking_df, events_df)
        tracking_df = self._make_serializable(tracking_df)

        # Standardize event ID columns
        events_df = self._standardize_id_columns(events_df)

        # Merge player information into tracking data
        tracking_df = self._merge_player_info(tracking_df, players_info)

        # Adjust coordinate system based on team orientation
        home_team_start_left = events_df['home_team_start_left'].iloc[0]
        tracking_df = change_events_side(tracking_df, events_df, home_team_start_left)

        # Add ball location to events
        events_df = self._add_event_location(tracking_df, events_df)

        events_df['data_version'] = self.data_version

        # Get columns for tracking and events data
        tracking_columns = self._get_tracking_columns(tracking_df, add_velocity)
        event_columns = self._get_event_columns(events_df)

        # Filter dataframes to selected columns
        clean_tracking_df = tracking_df[tracking_columns]
        events_df = events_df[event_columns]

        if save:
            self.save_data(match_id, clean_tracking_df, events_df, players_info)

        return clean_tracking_df, events_df, players_info

    def _merge_event_metadata(
        self,
        events_df: pd.DataFrame,
        metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge event data with tracking metadata."""
        metadata_subset = metadata_df[["match_id", "event_id", "possession_id", "frame_id"]].copy()
        metadata_subset["match_id"] = metadata_subset["match_id"].astype(int)
        metadata_subset["possession_id"] = metadata_subset["possession_id"].astype(int)
        metadata_subset["frame_id"] = metadata_subset["frame_id"].astype(int)

        frame_bounds = (
            metadata_subset
            .groupby(["match_id", "possession_id"], as_index=False)
            .agg(start_frame_id=("frame_id", "min"), end_frame_id=("frame_id", "max"))
        )

        merged = events_df.merge(
            frame_bounds,
            on=['match_id', 'possession_id'],
            how='inner',
        )
        # Keep frame_id as start-frame alias for backward compatibility.
        merged["frame_id"] = merged["start_frame_id"]
        return merged

    def _filter_tracking_frames(
        self,
        tracking_df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter tracking data to only include frames with events."""
        return tracking_df[tracking_df['frame_id'].isin(events_df['frame_id'])]

    def _standardize_id_columns(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize ID columns to integer type."""

        EVENT_ID_COLUMNS = [
            "match_id",
            "event_id",
            "possession_id",
            "frame_id",
            "start_frame_id",
            "end_frame_id",
        ]

        events_df = events_df.dropna(subset=EVENT_ID_COLUMNS).reset_index(drop=True)

        for col in EVENT_ID_COLUMNS:
            events_df[col] = events_df[col].astype(int)
        return events_df

    def _merge_player_info(
        self,
        tracking_df: pd.DataFrame,
        players_info: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge player information into tracking data."""
        player_columns = ["player_id", "team_id", "team_side", "shirt_number"]

        tracking_df['shirt'] = tracking_df['shirt'].astype(str)
        players_info['shirt_number'] = players_info['shirt_number'].astype(str)

        return tracking_df.merge(
            players_info[player_columns],
            left_on=['shirt', 'team'],
            right_on=['shirt_number', 'team_side'],
            how='left'
        )

    def _add_event_location(
        self,
        tracking_df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add ball location from tracking data to events."""
        ball_frame_df = (
            tracking_df[['match_id', 'frame_id', 'period', 'elapsed_seconds', 'ball_x', 'ball_y', 'ball_z']]
            .drop_duplicates(subset=['match_id', 'frame_id'])
            .reset_index(drop=True)
        )

        player_start_df = tracking_df[['match_id', 'frame_id', 'player_id', 'x', 'y']].copy()
        events_df = events_df.merge(
            player_start_df,
            left_on=['match_id', 'start_frame_id', 'player_id'],
            right_on=['match_id', 'frame_id', 'player_id'],
            how='left',
            suffixes=('', '_player_start'),
        )
        if 'frame_id_player_start' in events_df.columns:
            events_df = events_df.drop(columns=['frame_id_player_start'])

        ball_start_df = ball_frame_df.rename(columns={
            'frame_id': 'start_frame_id',
            'period': 'period_start',
            'elapsed_seconds': 'elapsed_seconds_start',
            'ball_x': 'ball_x_start',
            'ball_y': 'ball_y_start',
            'ball_z': 'ball_z_start',
        })
        events_df = events_df.merge(ball_start_df, on=['match_id', 'start_frame_id'], how='left')

        ball_end_df = ball_frame_df.rename(columns={
            'frame_id': 'end_frame_id',
            'period': 'period_end',
            'elapsed_seconds': 'elapsed_seconds_end',
            'ball_x': 'ball_x_end',
            'ball_y': 'ball_y_end',
            'ball_z': 'ball_z_end',
        })
        events_df = events_df.merge(ball_end_df, on=['match_id', 'end_frame_id'], how='left')

        # Preserve previous fields as aliases to start-frame values.
        events_df['period'] = events_df.get('period_start')
        events_df['elapsed_seconds'] = events_df.get('elapsed_seconds_start')
        events_df['ball_x'] = events_df.get('ball_x_start')
        events_df['ball_y'] = events_df.get('ball_y_start')
        events_df['ball_z'] = events_df.get('ball_z_start')

        return events_df

    def _get_tracking_columns(
        self,
        tracking_df: pd.DataFrame,
        add_velocity: bool
    ) -> list[str]:
        """
        Get the columns to include in tracking data based on available columns and velocity settings.

        Args:
            tracking_df: Tracking DataFrame to check for available columns
            add_velocity: Whether to include velocity columns

        Returns:
            List of column names to include in tracking data
        """
        base_columns = [
            'match_id', 'frame_id', 'period', 'elapsed_seconds',
            'team_id', 'player_id', 'x', 'y', 'ball_x', 'ball_y', 'ball_z',
            'shirt', 'team_side', 'team_phase'
        ]

        velocity_cols = [
            'vx', 'vy', 'ax', 'ay', 'speed',
            'ball_vx', 'ball_vy', 'ball_vz', 'ball_speed',
            'ball_ax', 'ball_ay', 'ball_az'
        ] if add_velocity else []

        # Combine base and velocity columns, filtering for available columns
        all_columns = base_columns + velocity_cols
        return [col for col in all_columns if col in tracking_df.columns]

    def _get_event_columns(self, events_df: pd.DataFrame) -> list[str]:
        """
        Get the columns to include in event data based on event type and available columns.

        Args:
            events_df: Event DataFrame to check for available columns

        Returns:
            List of column names to include in event data
        """
        # Define base columns that are always included
        base_columns = [
            'match_id', 'event_id', 'possession_id', 'frame_id', 'start_frame_id', 'end_frame_id',
            'period', 'elapsed_seconds',
            'team_id', 'player_id', 'possession_type', 'x', 'y',
            'ball_x', 'ball_y', 'ball_z', 'ball_height', 'set_piece',
            'ball_x_start', 'ball_y_start', 'ball_z_start',
            'ball_x_end', 'ball_y_end', 'ball_z_end',
            'video_url', 'team_side', 'home_team_start_left', 'data_version'
        ]

        # Define event-specific column groups
        event_specific_columns = {
            'shot': ['shot_outcome', 'shot_type', 'body_part', 'body_movement_type', 'ball_moving', 'nature_type'],
            'pass': ['pass_type', 'pass_accuracy_type', 'receiver_player_id', 'target_player_id', 'no_look', 'creates_space', 'pressure_type', 'lines_broken_type', 'cross_type', 'cross_zone_type', 'cross_outcome', 'pass_outcome'],
            'carry': ['carry_outcome', 'carry_type', 'carry_intent', 'carry_success', 'dribble_type', 'dribble_outcome']
        }

        # Collect all available event-specific columns
        available_event_columns = []
        for _, columns in event_specific_columns.items():
            available_event_columns.extend([col for col in columns if col in events_df.columns])

        # Combine all columns and filter for available ones
        all_columns = base_columns + available_event_columns
        return [col for col in all_columns if col in events_df.columns]

    def save_data(
        self,
        match_id: str,
        tracking_df: pd.DataFrame,
        events_df: pd.DataFrame,
        players_df: pd.DataFrame
    ) -> None:
        """
        Save processed data to parquet files.

        Args:
            match_id: Match identifier
            tracking_df: Tracking data
            events_df: Event data
            players_df: Player information
        """
        match_dir = self.save_path / match_id
        match_dir.mkdir(parents=True, exist_ok=True)

        players_df.to_parquet(match_dir / "players.parquet", engine='pyarrow')
        events_df.to_parquet(match_dir / "events.parquet", engine='fastparquet')
        tracking_df.to_parquet(match_dir / "tracking.parquet", engine='fastparquet')

    def load_events(self, match_id: str) -> pd.DataFrame:
        """
        Load raw event data for a match.

        Args:
            match_id: Match identifier

        Returns:
            DataFrame with raw event data
        """
        events_file = self.events_path / f"{match_id}_events.json"
        try:
            return pd.read_json(events_file, lines=True)
        except Exception as e:
            print(f"Error loading events for match {match_id}: {e}")
            print(f"Events file: {events_file}")
            raise

    def load_tracking(
        self,
        match_id: str,
        add_velocity: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and process tracking data for a match.

        Args:
            match_id: Match identifier
            add_velocity: Whether to calculate velocity features

        Returns:
            Tuple of (metadata_df, tracking_df)
        """
        metadata_df, tracking_df = pff_frames_to_dataframe(
            gandula.get_frames(str(self.tracking_path), match_id),
            smoothed=True
        )

        metadata_df, tracking_df = self._process_tracking(metadata_df, tracking_df)

        if add_velocity:
            tracking_df = add_players_speed(tracking_df)
            tracking_df = add_ball_speed(tracking_df)

        return metadata_df, tracking_df

    def _process_events(
        self,
        events_df: pd.DataFrame,
        event_type: str | list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process event data and extract player information.

        Args:
            events_df: Raw event data
            event_type: Type of events to filter

        Returns:
            Tuple of (filtered_events, players_info)
        """
        try:
            game_data = events_df['game'].iloc[0]
            match_events = pd.DataFrame(game_data['gameEvents'])
            home_team_id = int(game_data['homeTeam']['id']) if game_data['homeTeam'] else None
            home_team_start_left = game_data['homeTeamStartLeft'] if game_data['homeTeamStartLeft'] else None
            match_id = int(game_data['id']) if game_data['id'] else None
        except Exception as e:
            print(f"Error processing game data: {e}")
            print(f"Events_df shape: {events_df.shape}")
            print(f"Events_df columns: {events_df.columns.tolist()}")
            if 'game' in events_df.columns:
                print(f"Game column type: {type(events_df['game'].iloc[0])}")
            raise

        # Extract player information
        players_info = self._extract_player_info(game_data['rosters'], home_team_id)

        match_events['match_id'] = match_id
        match_events['team_id'] = match_events['team'].apply(
            lambda x: int(x.get('id')) if x else None
        )
        match_events['player_id'] = match_events['player'].apply(
            lambda x: int(x.get('id')) if x else None
        )

        # Process possession events
        possession_events = (
            match_events[['match_id','player_id','team_id', 'possessionEvents', 'setpieceType', 'videoUrl']]
            .explode('possessionEvents')
            .dropna(subset=['match_id','player_id','team_id', 'possessionEvents'])
            .reset_index(drop=True)
        )


        # Filter by event type
        filtered_events = self._filter_by_event_type(
            possession_events, event_type, home_team_id
        )
        filtered_events['home_team_start_left'] = home_team_start_left

        return filtered_events, players_info

    def _extract_player_info(
        self,
        rosters: list,
        home_team_id: int
    ) -> pd.DataFrame:
        """Extract and format player information from rosters."""

        try:
            columns = [
                'player.id', 'player.nickname', 'positionGroupType', 'shirtNumber',
                'team.id', 'team.name', 'player.preferredFoot', 'player.height', 'player.weight'
            ]
            players_info = pd.json_normalize(rosters)[columns]

            # Rename columns for consistency
            players_info.rename(columns={
                'player.id': 'player_id',
                'player.nickname': 'player_name',
                'team.id': 'team_id',
                'team.name': 'team_name',
                'player.preferredFoot': 'preferred_foot',
                'positionGroupType': 'position_name',
                'shirtNumber': 'shirt_number',
                'player.height': 'height',
                'player.weight': 'weight'
            }, inplace=True)

            players_info['team_id'] = players_info['team_id'].astype(int)
            players_info['shirt_number'] = players_info['shirt_number'].astype(int)
            players_info['player_id'] = players_info['player_id'].astype(int)

            # Add team side indicator
            players_info['team_side'] = players_info['team_id'].apply(
                lambda x: 'home' if int(x) == home_team_id else 'away'
            )

            return players_info
        except Exception as e:
            print(f"Error extracting player info: {e}")
            print(f"Rosters: {rosters}")
            print(f"Home team ID: {home_team_id}")
            raise

    def _filter_by_event_type(
        self,
        possession_events: pd.DataFrame,
        event_type: str | list[str],
        home_team_id: int
    ) -> pd.DataFrame:
        """Filter events by type and add team side information."""

        try:
            parsed_events = pd.json_normalize(
                possession_events.apply(parse_events, axis=1).dropna()
            )

            parsed_events = parse_names(parsed_events)

            if isinstance(event_type, str) and event_type in ['shot', 'pass', 'carry', 'all']:
                if event_type == 'all':
                    filtered_events = parsed_events
                else:
                    filtered_events = parsed_events[parsed_events['possession_type'] == event_type].reset_index(drop=True)
            elif isinstance(event_type, list):
                filtered_events = parsed_events[parsed_events['possession_type'].isin(event_type)].reset_index(drop=True)
            else:
                # Placeholder for other event types
                raise NotImplementedError(f"Event type '{event_type}' is not yet supported")

            filtered_events['team_side'] = filtered_events['team_id'].apply(
                lambda x: 'home' if x is not None and int(x) == home_team_id else 'away'
            )
            return filtered_events
        except Exception as e:
            print(f"Error filtering events: {e}")
            print(f"Possession events: {possession_events.shape}")
            print(f"Event type: {event_type}")
            print(f"Home team ID: {home_team_id}")
            raise

    def _process_tracking(
        self,
        metadata_df: pd.DataFrame,
        tracking_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process and clean tracking data.

        Args:
            metadata_df: Metadata DataFrame
            tracking_df: Players tracking DataFrame

        Returns:
            Tuple of processed (metadata_df, tracking_df)
        """
        # Process metadata
        metadata_df['possession_type'] = metadata_df['possession_type'].astype(str)
        metadata_df['match_id'] = metadata_df['match_id'].ffill()
        metadata_df = metadata_df.dropna(subset=['possession_id'])
        metadata_df = metadata_df.drop_duplicates(subset=['frame_id', 'match_id']).reset_index(drop=True)
        metadata_df['possession_id'] = metadata_df['possession_id'].astype(int)
        metadata_df['frame_id'] = metadata_df['frame_id'].astype(int)

        # Process player tracking
        tracking_df['match_id'] = tracking_df['match_id'].ffill()
        tracking_df = tracking_df.dropna(subset=['frame_id', 'match_id'], how='any')
        tracking_df = tracking_df.drop_duplicates(subset=['frame_id', 'match_id', 'team', 'shirt']).reset_index(drop=True)
        tracking_df['frame_id'] = tracking_df['frame_id'].astype(int)
        tracking_df['match_id'] = tracking_df['match_id'].astype(int)
        tracking_df['shirt'] = tracking_df['shirt'].astype(int)

        return metadata_df, tracking_df

    def _make_serializable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to serializable types for parquet storage."""
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(
                    lambda x: x.name if hasattr(x, "name") else str(x)
                )
        return df

    def __len__(self) -> int:
        """Return the number of available matches."""
        return len(self.match_ids)

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"PFFDataset(competition='{self.competition}', "
            f"season='{self.season}', matches={len(self)}, "
            f"version='{self.data_version}')"
        )