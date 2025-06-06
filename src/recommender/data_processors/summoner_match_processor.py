import math
import os
import pickle as pkl
from collections import defaultdict

import pandas as pd

from ..data_loaders.summoner_match_loader import SummonerMatchLoader
from ..data_processors.summoner_data_processor import SummonerDataProcessor


class SummonerMatchProcessor(SummonerDataProcessor):
    """Class for manipulating raw data and creating ratings per user."""

    def __init__(self):
        super().__init__()
        self.sdl = SummonerMatchLoader()

    def aggregate_summoner_pkls(self, overwrite_aggregate: bool = False) -> dict:
        """
        Using all the summoner_pkls, constructs a nested dictionary of champion data per puuid. If pkl exists, simply loads.

        Args:
            overwrite_aggregate (bool, optional): Whether or not to rewrite the existing pkl file. Defaults to False.

        Returns:
            dict: Nested dictionary of champion data per puuid.
        """
        pkl_path = os.path.join(
            self.project_root, "src/cache/aggregate_summoner_data.pkl"
        )
        if os.path.exists(pkl_path) and not overwrite_aggregate:
            with open(pkl_path, "rb") as f:
                return pkl.load(f)

        else:
            print("Re-aggregating summoner data...")
            pkl_folder_path = os.path.join(self.project_root, "src/summoner_pkls/")

            combined_puuid_dict = {}
            for pkl_file in os.listdir(pkl_folder_path):
                puuid = pkl_file.split(".pkl")[0]
                combined_puuid_dict[puuid] = self.sdl.load_dict_from_pkl(puuid)
            with open(pkl_path, "wb") as f:
                pkl.dump(dict(combined_puuid_dict), f)
            print(f"Successfully pickled combined puuid_dict.")
        return combined_puuid_dict

    def load_rating(
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

        pkl_path = os.path.join(self.project_root, "src/cache/rating_data.pkl")
        if os.path.exists(pkl_path) and not overwrite_rating:
            with open(pkl_path, "rb") as f:
                return pkl.load(f)

        else:
            print("Loading new ratings...")
            # Note that overwriting the ratings_df is not the same as overwriting the aggregate json
            player_champions = self.aggregate_summoner_pkls(overwrite_aggregate)

            player_champion_stats = defaultdict(lambda: defaultdict(int))
            for puuid, champ_data in player_champions.items():
                for champion_name, count in champ_data.items():
                    # Searches for the champion name to get stats
                    if "_" not in champion_name:
                        kills = player_champions[puuid][f"{champion_name}_kills"]
                        assists = player_champions[puuid][f"{champion_name}_assists"]
                        deaths = player_champions[puuid][f"{champion_name}_deaths"]
                        wins = player_champions[puuid].get(f"{champion_name}_wins", 0)
                        win_pct = (wins / count) if count > 0 else 0
                        kda = (kills + assists) / (deaths if deaths != 0 else 1)

                        # Scale the win % to within 0.4-0.7 and have a logarithmic scale for KDA, capped above by 6
                        scaled_win_pct = (min(max(win_pct, 0.4), 0.7) - 0.4) / (
                            0.7 - 0.4
                        )
                        scaled_kda = math.log(min(kda, 6) + 1) / math.log(7)

                        # Normalizes a performance and enthusiasm score to attain a rating from 1-10, irrelevant of games played total
                        usage_score = usage_score = count / (count + 10)
                        performance_score = scaled_win_pct * scaled_kda

                        rating = 10 * usage_score * performance_score
                        player_champion_stats[puuid][champion_name] = rating

            # If no games, assign 0
            rating_df = pd.DataFrame(player_champion_stats).transpose().fillna(0)

            # Writes to pkl file
            with open(pkl_path, "wb") as f:
                pkl.dump(rating_df, f)
            return rating_df
