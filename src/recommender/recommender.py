import os

import torch

from .data_loaders.summoner_mastery_loader import SummonerMasteryLoader
from .data_processors.summoner_mastery_processor import SummonerMasteryProcessor
from .user_item_collab_filter import UserItemCollabFilter
from .utils.map_helper import MapHelper


class Recommender:
    """Class to encapsulate the recommendation."""

    def __init__(self):
        self.project_root = os.getenv("PROJECT_ROOT")
        self.map_helper = MapHelper()
        self.summoner_mastery_loader = SummonerMasteryLoader()
        self.summoner_mastery_processor = SummonerMasteryProcessor()

    def recommend_champions(self):
        """Prompts summoner id and then recommends three champions."""
        # Prompt in game name
        puuid = self.map_helper.get_puuid_mapping()

        filter = UserItemCollabFilter()
        predicted_ratings, le_champion = filter.recommend_champions(puuid)

        _, top_indices = torch.topk(predicted_ratings, k=3)
        top_indices_list = top_indices.tolist()
        _, bottom_indices = torch.topk(-predicted_ratings, k=3)
        bottom_indices_list = bottom_indices.tolist()

        # Load champ order
        champ_order = [
            x for x in le_champion.inverse_transform(range(le_champion.classes_))
        ]
        print("=================================")
        top_ranking = 1
        bottom_ranking = 1
        for idx in top_indices_list:
            print(f"You should try playing {top_ranking}: {champ_order[idx]}")
            top_ranking += 1
        print("=================================")
        for idx in bottom_indices_list:
            print(f"You should avoid playing {bottom_ranking}: {champ_order[idx]}")
            bottom_ranking += 1
        print("=================================")


if __name__ == "__main__":
    # python -m src.recommender.recommender
    recommender = Recommender()
    recommender.recommend_champions()
