import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..utils.data_utils import ChampionsDataset, MultiEpochsDataLoader
from .common import BaseRecommender, DotProduct


class SGDCollabFilter(BaseRecommender):
    """Model-based matrix factorization collaborative filter."""

    def __init__(self):
        """Initializes the device and means of obtaining data."""
        super().__init__()

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader):
        """
        Evaluate model metrics.

        Args:
            model (torch.Module): Model of interest.
            test_loader (DataLoader): Test DataLoader.
        """
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for summoner_ids, champ_ids, ratings in test_loader:
                preds = model(summoner_ids, champ_ids)
                y_pred.extend(preds.numpy())
                y_true.extend(ratings.numpy())
            mse = mean_squared_error(y_true, y_pred)
            print(f"Test MSE: {mse}")

    def train_and_evaluate_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 20,
        lr: float = 0.05,
    ):
        """
        Trains and evaluates a model for a given number of epochs.

        Args:
            model (nn.Module): Model of interest.
            train_loader (DataLoader): DataLoader of train data.
            test_loader (DataLoader): DataLoader of test data.
            epochs (int, optional): Number of epochs. Defaults to 20.
            lr (float, optional): Learning rate. Defaults to 0.05.
        """
        start_time = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for summoner_ids, champ_ids, ratings in train_loader:
                optimizer.zero_grad()
                preds = model(summoner_ids, champ_ids)
                loss = criterion(preds, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
            self.evaluate_model(model, test_loader)
        print(f"Model training completed in {(time.time() - start_time)} seconds.")

    def recommend_champions(
        self,
        puuid: str,
        test_size: float = 0.2,
        num_workers: int = 3,
        batch_size: int = 10,
        num_factors: int = 5,
        epochs: int = 20,
        lr: float = 0.05,
    ) -> dict:
        """
        Given a puuid, returns an ordered list of recommended champions.

        Args:
            puuid (str): Puuid of interest.
            test_size (float, optional): Proportion of test data. Defaults to 0.2.
            num_workers (int, optional): Number of DataLoader workers. Defaults to 3.
            batch_size (int, optional): Batch size of DataLoader for mini-batch SGD. Defaults to 10.
            num_factors (int, optional): Number of latent factors. Defaults to 5.
            epochs (int, optional): Number of training epochs. Defaults to 20.
            lr (float, optional): Learning rate. Defaults to 0.05.

        Returns:
            dict: Rating to champion dict.
        """
        overwrite = False
        puuid_path = os.path.join(
            self.project_root, f"data/summoner_mastery_pkls/{puuid}.pkl"
        )
        if not os.path.exists(puuid_path):
            overwrite = True
            self.summoner_mastery_loader.dump_data_for_puuid(puuid)

        df, le_user, le_champion = self.summoner_mastery_processor.load_encoded_ratings(
            overwrite, overwrite
        )

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
        self.train_and_evaluate_model(model, train_loader, test_loader, epochs, lr)

        # Get ratings for given user
        user_idx = le_user.transform([puuid])
        all_champions = torch.tensor([champ_idx for champ_idx in range(num_champions)])
        user_id = torch.tensor(list(user_idx) * len(all_champions))
        predicted_ratings = model(user_id, all_champions)

        champ_order = torch.argsort(predicted_ratings, descending=True)
        # Avoid ties
        epsilon = 1e-6
        for i in range(len(predicted_ratings)):
            predicted_ratings[i] -= i * epsilon

        predicted_ratings_dict = {
            predicted_ratings[champ.item()].item(): le_champion.inverse_transform(
                [champ.item()]
            )[0]
            for champ in champ_order
        }
        return predicted_ratings_dict
