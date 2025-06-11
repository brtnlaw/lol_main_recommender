import os

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .data_loaders.summoner_mastery_loader import SummonerMasteryLoader
from .data_processors.summoner_mastery_processor import SummonerMasteryProcessor
from .utils.model_utils import MultiEpochsDataLoader, train_and_evaluate_model


class ChampionsDataset(Dataset):
    """Custom Dataset with puuid, champion, and rating."""

    def __init__(self, df):
        """Dummy for inheritance."""
        self.puuids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.champions = torch.tensor(df["champ_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float)

    def __len__(self):
        """Dummy for inheritance."""
        return len(self.puuids)

    def __getitem__(self, idx):
        """Gets puuid, champion, and rating."""
        return self.puuids[idx], self.champions[idx], self.ratings[idx]


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


class UserItemCollabFilter:
    """Model-based matrix factorization collaborative filter."""

    def __init__(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.project_root = os.getenv("PROJECT_ROOT")
        self.summoner_mastery_loader = SummonerMasteryLoader()
        self.summoner_mastery_processor = SummonerMasteryProcessor()

    # return ratings and champ_encoder
    def recommend_champions(
        self,
        puuid: str,
        test_size: float = 0.2,
        num_workers: int = 3,
        batch_size: int = 10,
        num_factors: int = 5,
        epochs: int = 20,
    ):
        overwrite = False

        # Check if data exists as json
        puuid_path = os.path.join(
            self.project_root, f"data/summoner_mastery_pkls/{puuid}.pkl"
        )
        if not os.path.exists(puuid_path):
            overwrite = True
            self.summoner_mastery_loader.dump_data_for_puuid(puuid)

        # Get the rating data
        df, le_user, le_champion = self.summoner_mastery_processor.load_encoded_ratings(
            overwrite, overwrite
        )

        # Train a model
        train_df, test_df = train_test_split(df, test_size=test_size)
        train_dataset = ChampionsDataset(train_df)
        test_dataset = ChampionsDataset(test_df)

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

        num_summoners = len(df["puuid"].unique())
        num_champions = len(df["champ_id"].unique())

        model = DotProduct(num_summoners, num_champions, num_factors).to(self.device)
        train_and_evaluate_model(model, train_loader, test_loader, epochs=epochs)

        # Get relevant user index
        user_idx = le_user.transform([puuid])

        # Get preds
        all_champions = torch.tensor([champ_idx for champ_idx in range(num_champions)])
        user_id = torch.tensor(list(user_idx) * len(all_champions))
        predicted_ratings = model(user_id, all_champions)

        champ_order = torch.argsort(predicted_ratings, descending=True)
        return list(le_champion.inverse_transform(champ_order.numpy()))


# maybe a memory based just for posterity
