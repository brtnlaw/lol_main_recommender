import torch
import torch.nn as nn


class SummonerTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_factors = 10
        # 10 ranks from Iron to Challenger
        self.rank_factors = nn.Embedding(10, 8)
        # 5 lanes
        self.lane_factors = nn.Embedding(5, 8)
        self.input_dim = (
            self.rank_factors.embedding_dim + self.lane_factors.embedding_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_factors),
        )

    def forward(self, rank_ids, lane_ids):
        x = torch.concat(
            [self.rank_factors(rank_ids), self.lane_factors(lane_ids)], dim=-1
        )
        return self.mlp(x)


# Negative sampling somewhere
class ChampTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_factors = 10
        # not lanes, idk how many
        self.role_factors = nn.Embedding(5, 8)
        self.attack_type_factors = nn.Embedding(2, 8)
        self.input_dim = (
            self.role_factors.embedding_dim + self.attack_type_factors.embedding_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_factors),
        )

    def forward(self, role_ids, attack_type_ids):
        x = torch.concat(
            [self.role_factors(role_ids), self.attack_type_factors(attack_type_ids)],
            dim=-1,
        )
        return self.mlp(x)


# Already Dot Product?
class TwoTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.summoner_tower = SummonerTower()
        self.champ_tower = ChampTower()

    def forward(self, summoner_tensor_tuple, champ_tensor_tuple):
        summoner_embedding = self.summoner_tower(*summoner_tensor_tuple)
        champion_embedding = self.champ_tower(*champ_tensor_tuple)
        score = (summoner_embedding * champion_embedding).sum(dim=-1)
        return score
