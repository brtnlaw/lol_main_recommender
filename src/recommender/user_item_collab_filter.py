import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

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


# TODO: If I want to implement this, I need to change the way that I load the data
# right now, pytorch SGD is mini-batch gradient descent; so namely, it takes however large our batch is from DataLoader
# It then adjusts the gradient based on wahtever minibatch you throw into it
class AlternatingLeastSquares(torch.optim.Optimizer):
    # Batch size needs to be the whole thing
    def __init__(self, params, mastery_tensor):
        super().__init__(params, defaults={})
        self.alternating_idx = 0
        # n summoners x m champions
        self.mastery_tensor = mastery_tensor

    def _solve_matrix(self, variable_factors, fixed_factors):
        # should be n x k or m x k
        update_tensor = torch.zeros_like(variable_factors)
        k = fixed_factors.shape[1]
        for i in range(len(variable_factors)):
            if self.alternating_idx == 0:
                # ith summoner ratings
                rating_tensor = self.mastery_tensor[i, :]
            else:
                # ith champion ratings
                rating_tensor = self.mastery_tensor[:, i]
            sum_tensor = torch.zeros([k, k])
            sum_r_tensor = torch.zeros(k)
            for j in range(len(fixed_factors)):
                # sum of y_iy_i^T + lambda I, lambda = 0
                # (kx1)(kx1)^T = kxk
                sum_tensor += torch.outer(fixed_factors[j], fixed_factors[j])
                # sum of r_ij y_i
                sum_r_tensor += rating_tensor[j] * fixed_factors[j]
            # (kxk)(kx1) = (kx1)
            update_tensor[i, :] = torch.matmul(torch.inverse(sum_tensor), sum_r_tensor)
        return update_tensor

    def step(self):
        summoner_factors, champion_factors = self.param_groups[0]["params"]
        if self.alternating_idx == 0:
            summoner_factors.data = self._solve_matrix(
                summoner_factors.data, champion_factors.data
            )
        else:
            champion_factors.data = self._solve_matrix(
                champion_factors.data, summoner_factors.data
            )
        self.alternating_idx = 1 - self.alternating_idx


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


def train_and_evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 20,
    lr: float = 0.05,
    mastery_tensor=None,
):
    """
    Trains and evaluates a model for a given number of epochs.

    Args:
        model (nn.Module): Model of interest.
        train_loader (DataLoader): DataLoader of train data.
        test_loader (DataLoader): DataLoader of test data.
        epochs (int, optional): Number of epochs. Defaults to 20.
    """
    start_time = time.time()
    optimizer = AlternatingLeastSquares(model.parameters(), mastery_tensor)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        # Training mode and reset loss
        model.train()
        total_loss = 0
        for user_ids, champ_ids, ratings in train_loader:
            # Zero out the gradient, make predictions, backward propagate the losses, and step forward with the optimizer
            optimizer.zero_grad()
            preds = model(user_ids, champ_ids)
            # Criterion betweens the forward run and the true rating
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
        # evaluate_model(model, test_loader)
    print(f"Model training completed in {(time.time() - start_time)} seconds.")


# maybe a memory based just for posterity
