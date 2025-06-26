import os
from abc import abstractmethod

import torch
import torch.nn as nn

from ..data_loaders.summoner_mastery_loader import SummonerMasteryLoader
from ..data_processors.summoner_mastery_processor import SummonerMasteryProcessor
from ..utils.map_helper import MapHelper


class DotProduct(nn.Module):
    """Model that dot products the summoner and champion factors. Matrix approach of collaborative filtering."""

    def __init__(self, num_summoners, num_champions, num_factors):
        """Module with summoner and champion factors as Embedding objects with learned factors per summoner/champion."""
        super().__init__()
        self.summoner_factors = nn.Embedding(num_summoners, num_factors)
        self.champion_factors = nn.Embedding(num_champions, num_factors)

    def forward(self, summoner_ids, champ_ids):
        """Simply takes the dot product at each forward step to get a predicted rating."""
        summoner_embedding = self.summoner_factors(summoner_ids)
        champion_embedding = self.champion_factors(champ_ids)
        return (summoner_embedding * champion_embedding).sum(dim=1)


class BaseRecommender:
    """Base class that loads and requires a recommend_champions."""

    def __init__(self):
        """Basic loaders, processors, utils."""
        self.project_root = os.getenv("PROJECT_ROOT")
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.summoner_mastery_loader = SummonerMasteryLoader()
        self.summoner_mastery_processor = SummonerMasteryProcessor()
        self.map_helper = MapHelper()

    @abstractmethod
    def get_predicted_ratings(self, puuid: str, *args, **kwargs) -> dict:
        """GEts a rating to champion dictionary to determine what to recommend."""
        pass

    def recommend_champions(self, puuid: str, *args, **kwargs) -> None:
        """Takes in a puuid and returns an ordered list of champions."""
        predicted_ratings_dict = self.get_predicted_ratings(puuid, *args, **kwargs)
        ordered_champs = sorted(
            enumerate(predicted_ratings_dict.keys()), key=lambda x: x[1], reverse=True
        )
        print("=================================")
        for list_rank in range(1, 4):
            print(
                f"You should try playing {list_rank}: {predicted_ratings_dict[ordered_champs[list_rank-1][1]]}"
            )
        print("=================================")
        for list_rank in range(1, 4):
            print(
                f"You should avoid playing {list_rank}: {predicted_ratings_dict[ordered_champs[-(list_rank)][1]]}"
            )
