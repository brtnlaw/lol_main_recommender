import json
import math
import os
import pickle as pkl
import time
from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from map_helper import MapHelper
from riot_api_helper import RiotApiHelper

PROJECT_ROOT = os.getenv("PROJECT_ROOT")


class SummonerData:
    """Class for generating and manipulating user data with champions played."""

    def __init__(self):
        """Initializes an API Wrapper to ping Riot's API."""
        self.riot_api_helper = RiotApiHelper()

    def dump_matches_for_puuid(self, puuid: str, ct: int = 100) -> None:
        """
        Given a puuid, dumps match info into a json.

        Args:
            puuid (str): Puuid of interest.
            ct (int, optional): Number of matches, needs to be uniform. Defaults to 100.
        """
        file_path = os.path.join(
            PROJECT_ROOT,
            f"src/summoner_data/{puuid}.json",
        )
        if os.path.exists(file_path):
            return
        match_data = []
        print(f"Loading data for puuid: {puuid}")
        match_ids = self.riot_api_helper.get_player_matches(puuid, ct)
        print(len(match_ids))
        # Each json is to have 100 games by default
        for i in range(len(match_ids)):
            match_id = match_ids[i]
            print(f"Match {i} loaded, id: {match_id}...")
            match_data.append(self.riot_api_helper.get_match_info(match_id))
            time.sleep(1)
        with open(file_path, "w") as f:
            json.dump(match_data, f, indent=2)
        print(f"Successfully saved data for puuid: {puuid}")

    def save_challenger_data(self, ct: int = 100) -> None:
        """
        Loops through all the puuids of Challenger. Gets matches per puuid and saves into a json in the summoner_data folder.

        Args:
            ct (int, optional): Number of matches per json.
        """
        puuids = self.riot_api_helper.get_challenger_puuids()
        for puuid in puuids:
            file_path = os.path.join(
                PROJECT_ROOT,
                f"src/summoner_data/{puuid}.json",
            )
            if os.path.exists(file_path):
                continue
            self.dump_matches_for_puuid(puuid, ct)

    def aggregate_json(self, rewrite: bool = False) -> dict:
        """
        Using all the jsons, constructs a nested dictionary of champion data per puuid. If pkl exists, simply loads.

        Args:
            rewrite (bool, optional): Whether or not to rewrite the existing pkl file. Defaults to False.

        Returns:
            dict: Nested dictionary of champion data per puuid.
        """
        pkl_path = os.path.join(PROJECT_ROOT, "src/champion_data/raw_data.pkl")
        if rewrite == False and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                return pkl.load(f)

        print("Re-aggregating json...")
        json_folder_path = os.path.join(PROJECT_ROOT, "src/summoner_data/")

        # {player_id: {champion_id: count}}
        player_champions = defaultdict(lambda: defaultdict(int))
        for json_file in os.listdir(json_folder_path):
            puuid = json_file.split(".json")[0]
            # Load json for puuid
            with open(os.path.join(json_folder_path, json_file), "r") as f:
                match_data = json.load(f)

            # Get necessary info from match data
            for match in match_data:
                participants = match.get("info", {}).get("participants", [])
                for participant in participants:
                    if participant.get("puuid") == puuid:
                        # Store information to get rating
                        champion_id = participant.get("championId")
                        win = participant.get("win")
                        kills = participant.get("kills")
                        deaths = participant.get("deaths")
                        assists = participant.get("assists")

                        player_champions[puuid][champion_id] += 1
                        player_champions[puuid][f"{champion_id}_kills"] += kills
                        player_champions[puuid][f"{champion_id}_assists"] += deaths
                        player_champions[puuid][f"{champion_id}_deaths"] += assists
                        if win == 1:
                            player_champions[puuid][f"{champion_id}_wins"] += 1

        # Write to json
        with open(pkl_path, "wb") as f:
            pkl.dump(dict(player_champions), f)
        return dict(player_champions)

    def load_rating(self, rewrite: bool = False) -> pd.DataFrame:
        """
        Loads the rating DataFrame with puuid, champion, and a proprietary rating system.

        Args:
            rewrite (bool, optional): Whether or not to overwrite rating pkl. Defaults to False.

        Returns:
            pd.DataFrame: Rating DataFrame.
        """
        pkl_path = os.path.join(PROJECT_ROOT, "src/champion_data/rating_data.pkl")
        if rewrite == False and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                return pkl.load(f)

        print("Loading new ratings...")
        player_champions = self.aggregate_json(rewrite)

        player_champion_stats = defaultdict(lambda: defaultdict(int))
        for puuid, champ_data in player_champions.items():
            for champion_id, count in champ_data.items():
                # Searches for the id key to get stats
                if isinstance(champion_id, int):
                    kills = player_champions[puuid][f"{champion_id}_kills"]
                    assists = player_champions[puuid][f"{champion_id}_assists"]
                    deaths = player_champions[puuid][f"{champion_id}_deaths"]
                    wins = player_champions[puuid].get(f"{champion_id}_wins", 0)
                    win_pct = (wins / count) if count > 0 else 0
                    kda = (kills + assists) / (deaths if deaths != 0 else 1)

                    # Scale the win % to within 0.4-0.7 and have a logarithmic scale for KDA, capped above by 6
                    scaled_win_pct = (min(max(win_pct, 0.4), 0.7) - 0.4) / (0.7 - 0.4)
                    scaled_kda = math.log(min(kda, 6) + 1) / math.log(7)

                    # Normalizes a performance and enthusiasm score to attain a rating from 1-10
                    performance_score = math.log(count + 1) / math.log(101)
                    enthusiasm_score = scaled_win_pct * scaled_kda
                    rating = 10 * performance_score * enthusiasm_score
                    player_champion_stats[puuid][champion_id] = rating

        # If no games, assign 0
        rating_df = pd.DataFrame(player_champion_stats).transpose().fillna(0)

        # Writes to pkl file
        with open(pkl_path, "wb") as f:
            pkl.dump(rating_df, f)
        return rating_df

    def load_clean_df_encoders(
        self, rewrite: bool = False
    ) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
        """
        Cleans up the rating DataFrame and encodes.

        Args:
            rewrite (bool, optional): Whether or not to write over jsons. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]: Cleaned DataFrame, user encoder, champion encoder.
        """
        df = self.load_rating(rewrite)
        df.index.name = "puuid"
        df.reset_index(inplace=True)
        df = pd.melt(
            df, id_vars=["puuid"], var_name="orig_champ_id", value_name="rating"
        )
        map_helper = MapHelper()
        champ_map = map_helper.get_champ_id_to_name()
        df["champion"] = df["orig_champ_id"].apply(lambda x: champ_map.get(str(x)))
        le_user = LabelEncoder()
        df["user_id"] = le_user.fit_transform(df["puuid"].values)
        le_champion = LabelEncoder()
        df["champ_id"] = le_champion.fit_transform(df["orig_champ_id"].values)
        return df, le_user, le_champion
