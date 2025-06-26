import datetime as dt
import os
import pickle as pkl
from collections import defaultdict

import pandas as pd

from ..data_loaders.summoner_mastery_loader import SummonerMasteryLoader
from ..utils.map_helper import MapHelper
from .base_processor import BaseProcessor


class SummonerMasteryProcessor(BaseProcessor):
    """Class for manipulating raw data and creating ratings per user."""

    def __init__(self):
        super().__init__()
        self.summoner_mastery_loader = SummonerMasteryLoader()
        self.map_helper = MapHelper()

    def aggregate_summoner_pkls(self, overwrite_aggregate: bool = False) -> dict:
        """
        Using all the summoner_pkls, constructs a nested dictionary of champion data per puuid. If pkl exists, simply loads.

        Args:
            overwrite_aggregate (bool, optional): Whether or not to rewrite the existing pkl file. Defaults to False.

        Returns:
            dict: Nested dictionary of champion data per puuid.
        """
        pkl_path = os.path.join(
            self.project_root, "data/cache/aggregate_mastery_data.pkl"
        )
        if os.path.exists(pkl_path) and not overwrite_aggregate:
            with open(pkl_path, "rb") as f:
                return pkl.load(f)

        else:
            print("Re-aggregating summoner mastery data...")
            pkl_folder_path = self.summoner_mastery_loader.json_folder_path
            id_champ_map = self.map_helper.get_id_champ_map()

            combined_puuid_dict = {}
            for pkl_file in os.listdir(pkl_folder_path):
                puuid = pkl_file.split(".pkl")[0]
                old_puuid_dict = self.summoner_mastery_loader.load_dict_from_pkl(puuid)
                new_puuid_dict = {}
                id_champ_map = self.map_helper.get_id_champ_map()
                for key, value in old_puuid_dict.items():
                    champ_id_str, suffix = key.split("_", 1)
                    champ_name = id_champ_map.get(int(champ_id_str), champ_id_str)
                    new_puuid_dict[f"{champ_name}_{suffix}"] = value
                combined_puuid_dict[puuid] = new_puuid_dict
            with open(pkl_path, "wb") as f:
                pkl.dump(dict(combined_puuid_dict), f)
            print(f"Successfully pickled combined puuid_dict.")
        return combined_puuid_dict

    def load_ratings(
        self, overwrite_rating: bool = False, overwrite_aggregate: bool = False
    ) -> pd.DataFrame:
        """
        Loads the rating DataFrame with puuid, champion, and a proprietary rating system.

        Args:
            overwrite_rating (bool, optional): Whether or not to overwrite rating pkl. Defaults to False.
            overwrite_aggregate (bool, optional): Whether or not to overwrite the aggregate dict. Defaults to False.
        Returns:
            pd.DataFrame: Rating DataFrame.
        """

        pkl_path = os.path.join(self.project_root, "data/cache/mastery_rating_data.pkl")
        if os.path.exists(pkl_path) and not overwrite_rating:
            with open(pkl_path, "rb") as f:
                return pkl.load(f)

        else:
            print("Loading new ratings...")
            # Note that overwriting the ratings_df is not the same as overwriting the aggregate json
            player_champions = self.aggregate_summoner_pkls(overwrite_aggregate)

            player_champion_stats = defaultdict(lambda: defaultdict(int))
            for puuid, champ_data in player_champions.items():
                total_rating = 0
                dict_keys = champ_data.keys()
                player_played_champions = sorted(
                    {dict_key.split("_")[0] for dict_key in dict_keys}
                )

                for champion_name in player_played_champions:
                    level = player_champions[puuid][f"{champion_name}_level"]
                    last_play_time = player_champions[puuid][
                        f"{champion_name}_last_play_time"
                    ]
                    days_since_last_play = (
                        dt.datetime.now()
                        - dt.datetime.fromtimestamp(last_play_time / 1000)
                    ).days
                    rating = level * (1 - 0.5 * min(days_since_last_play, 365) / 365)
                    player_champion_stats[puuid][champion_name] = rating
                    total_rating += rating

                # Normalize
                for champion_name in player_champion_stats[puuid]:
                    player_champion_stats[puuid][champion_name] /= total_rating / 1000

            # If no games, assign 0
            rating_df = pd.DataFrame(player_champion_stats).transpose().fillna(0)

            with open(pkl_path, "wb") as f:
                pkl.dump(rating_df, f)
            return rating_df
