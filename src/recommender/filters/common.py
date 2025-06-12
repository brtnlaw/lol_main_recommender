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
        summoner_embedded = self.summoner_factors(summoner_ids)
        champion_embedded = self.champion_factors(champ_ids)
        return (summoner_embedded * champion_embedded).sum(dim=1)


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
    def recommend_champions(puuid: str, *args, **kwargs) -> list[str]:
        """Takes in a puuid and returns an ordered list of champions."""
        pass
