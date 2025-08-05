import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder
from torch.utils.data import DataLoader

from recommender.data_processors.mastery_features_processor import (
    MasteryFeaturesProcessor,
)

from ..utils.data_utils import (
    ChampionsDataset,
    ChampionsFeaturesDataset,
    MultiEpochsDataLoader,
)
from .common import BaseRecommender, DotProduct


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

        # Melee Ranged
        self.attack_type_factors = nn.Embedding(2, 8)
        # AD AP
        self.adaptive_type_factors = nn.Embedding(2, 8)
        # Mana, Blood Well, etc.
        self.resource_factors = nn.Embedding(15, 8)

        # For Multi-Hot Encoding, we use a small MLP to synthesize vector into embedding representation
        self.role_mlp = nn.Sequential(nn.Linear(17, 16), nn.ReLU(), nn.Linear(16, 8))
        self.position_mlp = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 8))

        self.input_dim = (
            self.attack_type_factors.embedding_dim
            + self.adaptive_type_factors.embedding_dim
            + self.resource_factors.embedding_dim
            + self.role_mlp[-1].out_features
            + self.position_mlp[-1].out_features
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_factors),
        )

    def forward(
        self,
        attack_type_ids,
        adaptive_type_ids,
        resource_ids,
        role_multihots,
        position_multihots,
    ):
        x = torch.concat(
            [
                self.attack_type_factors(attack_type_ids),
                self.adaptive_type_factors(adaptive_type_ids),
                self.resource_factors(resource_ids),
                self.role_mlp(role_multihots),
                self.position_mlp(position_multihots),
            ],
            dim=-1,
        )
        return self.mlp(x)


# NOTE: A bit redundant with DotProduct Module
class TwoTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.summoner_tower = SummonerTower()
        self.champ_tower = ChampTower()

    def forward(self, summoner_tensor_tuple, champ_tensor_tuple):
        summoner_embedding = self.summoner_tower(*summoner_tensor_tuple)
        champion_embedding = self.champ_tower(*champ_tensor_tuple)

        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # score = cos(summoner_embedding, champion_embedding)
        score = (summoner_embedding * champion_embedding).sum(dim=-1)
        return score


class TwoTowerRecommender(BaseRecommender):
    def evaluate_model(self, model, test_loader):
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for summoner_tuple, champion_tuple, ratings in test_loader:
                preds = model(summoner_tuple, champion_tuple)
                y_pred.extend(preds.numpy())
                y_true.extend(ratings.numpy())
            mse = mean_squared_error(y_pred, y_true)
            print(f"Test MSE: {mse}")

    def train_and_evaluate_model(
        self, model, train_loader, test_loader, epochs=10, lr=0.05
    ):
        start_time = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for summoner_tuple, champion_tuple, ratings in train_loader:
                optimizer.zero_grad()
                preds = model(summoner_tuple, champion_tuple)
                loss = criterion(preds, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
            self.evaluate_model(model, test_loader)
        print(f"Model training completed in {(time.time() - start_time)} seconds.")

    async def get_predicted_ratings(
        self,
        puuid,
        test_size=0.2,
        num_workers: int = 3,
        batch_size: int = 10,
        epochs: int = 1,
        lr: float = 0.05,
    ):
        mfp = MasteryFeaturesProcessor()
        puuid_path = os.path.join(
            self.project_root, f"data/summoner_mastery_pkls/{puuid}.pkl"
        )
        if not os.path.exists(puuid_path):
            self.summoner_mastery_loader.dump_data_for_puuid(puuid)

        df, _, le_champion = await mfp.async_load_encoded_ratings()
        categorical_cols = [
            "summoner_rank",
            "summoner_lane",
            "champ_attack_type",
            "champ_adaptive_type",
            # New
            "champ_resource",
        ]
        df[categorical_cols] = (
            OrdinalEncoder().fit_transform(df[categorical_cols]).astype(int)
        )

        list_cols = ["champ_positions", "champ_roles"]
        mlb = MultiLabelBinarizer()
        for list_col in list_cols:
            position_onehot = pd.DataFrame(
                mlb.fit_transform(df[list_col]),
                columns=[f"{list_col}_{cls}" for cls in mlb.classes_],
                index=df.index,
            )
            df = pd.concat([df.drop(columns=[list_col]), position_onehot], axis=1)

        role_cols = sorted([col for col in df.columns if col.startswith("champ_roles")])
        df["champ_roles"] = tuple(zip(*(df[role_col] for role_col in role_cols)))
        df.drop(columns=role_cols, inplace=True)

        position_cols = sorted(
            [col for col in df.columns if col.startswith("champ_positions")]
        )
        df["champ_positions"] = tuple(
            zip(*(df[position_col] for position_col in position_cols))
        )
        df.drop(columns=position_cols, inplace=True)

        train_df, test_df = train_test_split(df, test_size=test_size)
        train_dataset = ChampionsFeaturesDataset(train_df)
        test_dataset = ChampionsFeaturesDataset(test_df)

        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
        test_loader = MultiEpochsDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
        )

        model = TwoTowerModel().to(self.device)
        self.train_and_evaluate_model(model, train_loader, test_loader, epochs, lr)

        # TODO: Clean this part up
        summoner_df = df[df["puuid"] == puuid].iloc[0]

        rank_ids = torch.tensor(summoner_df["summoner_rank"], dtype=torch.long)
        lane_ids = torch.tensor(summoner_df["summoner_lane"], dtype=torch.long)
        summoner_tensor_tuple = (rank_ids, lane_ids)

        champ_df = (
            df[
                [
                    "champ_name",
                    "champ_attack_type",
                    "champ_adaptive_type",
                    "champ_resource",
                    "champ_roles",
                    "champ_positions",
                ]
            ]
            .drop_duplicates()
            .sort_values(by="champ_name")
        )
        attack_type_ids = torch.tensor(
            champ_df["champ_attack_type"].values, dtype=torch.long
        )
        adaptive_type_ids = torch.tensor(
            champ_df["champ_adaptive_type"].values, dtype=torch.long
        )
        resource_ids = torch.tensor(champ_df["champ_resource"].values, dtype=torch.long)
        role_multihots = torch.tensor(
            np.stack(champ_df["champ_roles"].values), dtype=torch.float
        )
        position_multihots = torch.tensor(
            np.stack(champ_df["champ_positions"].values), dtype=torch.float
        )

        champ_tensor_tuple = (
            attack_type_ids,
            adaptive_type_ids,
            resource_ids,
            role_multihots,
            position_multihots,
        )

        predicted_ratings = model(summoner_tensor_tuple, champ_tensor_tuple)
        champ_order = torch.argsort(predicted_ratings, descending=True)
        predicted_ratings_dict = {
            predicted_ratings[champ.item()].item(): le_champion.inverse_transform(
                [champ.item()]
            )[0]
            for champ in champ_order
        }
        return predicted_ratings_dict
