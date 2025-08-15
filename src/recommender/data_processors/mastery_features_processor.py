import os
import pickle as pkl
import random
import time
from collections import defaultdict

import pandas as pd
from sklearn.calibration import LabelEncoder

from ..data_loaders.summoner_mastery_loader import SummonerMasteryLoader
from ..data_loaders.summoner_match_loader import SummonerMatchLoader
from ..utils.map_helper import MapHelper
from ..utils.riot_api_helper import RiotApiHelper
from .base_processor import BaseProcessor
from .summoner_mastery_processor import SummonerMasteryProcessor


class MasteryFeaturesProcessor(BaseProcessor):
    """Includes feature data about summoner and champion for two tower model."""

    def __init__(self):
        super().__init__()
        self.summoner_mastery_loader = SummonerMasteryLoader()
        self.summoner_match_loader = SummonerMatchLoader()
        self.summoner_mastery_processor = SummonerMasteryProcessor()
        self.map_helper = MapHelper()
        self.riot_api_helper = RiotApiHelper()

    def aggregate_summoner_pkls(self):
        """Not applicable, as relies on other loaders."""
        pass

    def load_rating(self):
        """Not applicable, as relies on other loaders."""
        pass

    async def async_load_encoded_ratings(
        self, overwrite_aggregate=False
    ) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
        """
        Asynchronously generates the encoded ratings, adding features to the existing aggregate mastery pkl.

        Args:
            overwrite_aggregate (bool, optional): Whether or not to overwrite aggregate. Defaults to False.

        Returns:
            pd.DataFrame: Ratings DataFrame with features.
        """
        rating_df, le_user, le_champion = (
            self.summoner_mastery_processor.load_encoded_ratings()
        )

        pkl_path = os.path.join(
            self.project_root, "data/cache/agg_mastery_features_data.pkl"
        )
        if os.path.exists(pkl_path) and not overwrite_aggregate:
            with open(pkl_path, "rb") as f:
                return pkl.load(f), le_user, le_champion

        # Start with the existing mastery
        print("Re-aggregating summoner mix data...")
        match_folder_path = self.summoner_match_loader.json_folder_path
        champ_metadata = self.map_helper.get_lolstaticdata_champ_id_mapping()

        puuid_dict = defaultdict(dict)
        champ_dict = defaultdict(dict)
        puuids = rating_df["puuid"].unique()

        # Build champ features
        for champ in champ_metadata:
            champ_dict[champ]["adaptive_type"] = champ_metadata[champ]["adaptiveType"]
            champ_dict[champ]["attack_type"] = champ_metadata[champ]["attackType"]

            # TODO: Eventually include QWER CD? Range? Damage? Utility?
            champ_dict[champ]["resource"] = champ_metadata[champ]["resource"]
            champ_dict[champ]["positions"] = champ_metadata[champ]["positions"]
            champ_dict[champ]["roles"] = champ_metadata[champ]["roles"]

        # Build summoner features
        for puuid in puuids:
            print(f"Generating data for puuid {puuid}")
            rank = self.riot_api_helper.get_player_rank(puuid)[0]["tier"]
            time.sleep(0.025)

            if not os.path.exists(os.path.join(match_folder_path, f"{puuid}.pkl")):
                await self.summoner_match_loader.async_dump_data_for_puuid(puuid)
            match_history = self.summoner_match_loader.load_dict_from_pkl(puuid)
            lane_counts = defaultdict(int)
            role_counts = defaultdict(int)
            # Average KDA between champions
            kdas = []
            for champion in match_history:
                if "_" in champion:
                    continue
                kdas.append(
                    (
                        match_history[f"{champion}_kills"]
                        + match_history[f"{champion}_assists"]
                    )
                    / max(1, match_history[f"{champion}_deaths"])
                )

                games_played = match_history[champion]
                if champion == "FiddleSticks":
                    champion = "Fiddlesticks"
                for lane in champ_metadata[champion]["positions"]:
                    lane_counts[lane] += games_played
                for role in champ_metadata[champion]["roles"]:
                    role_counts[role] += games_played

            # Break ties of lanes randomly
            sorted_lanes = sorted(
                lane_counts.items(), key=lambda x: (-x[1], random.random())
            )
            sorted_roles = sorted(
                role_counts.items(), key=lambda x: (-x[1], random.random())
            )

            puuid_dict[puuid]["rank"] = rank
            puuid_dict[puuid]["lane"] = sorted_lanes[0][0]
            puuid_dict[puuid]["role"] = sorted_roles[0][0]
            puuid_dict[puuid]["avg_kda"] = sum(kdas) / len(kdas)

        # Just loops through second layer
        for attribute in puuid_dict[next(iter(puuid_dict))]:
            rating_df[f"summoner_{attribute}"] = rating_df["puuid"].map(
                lambda x: puuid_dict.get(x, {}).get(attribute)
            )

        for attribute in champ_dict[next(iter(champ_dict))]:
            rating_df[f"champ_{attribute}"] = rating_df["champ_name"].map(
                lambda x: champ_dict.get(x, {}).get(attribute)
            )

        with open(pkl_path, "wb") as f:
            pkl.dump(rating_df, f)
        return rating_df, le_user, le_champion
