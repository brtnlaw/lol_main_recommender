import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..utils.data_utils import ChampionsDataset, MultiEpochsDataLoader
from .common import BaseRecommender, DotProduct


class AlternatingLeastSquares(torch.optim.Optimizer):
    """Optimizer class that computes ALS."""

    def __init__(self, params, mastery_tensor):
        """Initializes the current alternating state as well as the tensor of ratings."""
        super().__init__(params, defaults={})

        # (n summoners x m champions)
        self.mastery_tensor = mastery_tensor
        self.alternating_idx = 0

    def _solve_matrix(
        self, variable_factors: torch.tensor, fixed_factors: torch.tensor
    ) -> torch.tensor:
        """
        Completes one step of ALS.

        Args:
            variable_factors (torch.tensor): The tensor which we are differentiating MSE with respect to.
            fixed_factors (torch.tensor): The tensor we are holding fixed.

        Returns:
            torch.tensor: The updated variable_factors tensor.
        """
        # (n x k) or (m x k)
        update_tensor = torch.zeros_like(variable_factors)
        k = fixed_factors.shape[1]
        for i in range(len(variable_factors)):
            if self.alternating_idx == 0:
                # i-th summoner ratings, (m x 1)
                row_tensor = self.mastery_tensor[i, :]
            else:
                # i-th champion ratings, (n x 1)
                row_tensor = self.mastery_tensor[:, i]
            mat_sum_tensor = torch.zeros([k, k])
            row_sum_tensor = torch.zeros(k)

            for j in range(len(fixed_factors)):
                # sum of (fixed_j)(fixed_j)^T + (lambda)(I), lambda = 0, (k x 1)(k x 1)^T = (k x k)
                mat_sum_tensor += torch.outer(fixed_factors[j], fixed_factors[j])
                # sum of mastery[i][j]* (fixed_j), a(k x 1)
                sum_row_row_sum_tensortensor += row_tensor[j] * fixed_factors[j]

            # (k x k)(k x 1) = (k x 1)
            update_tensor[i, :] = torch.matmul(
                torch.inverse(mat_sum_tensor), row_sum_tensor
            )
        return update_tensor

    def step(self):
        """Completes one step of ALS."""
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


class ALSCollabFilter(BaseRecommender):
    """Model-based matrix factorization collaborative filter solved using Alternating Least Squares."""

    def __init__(self):
        super().__init__()

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader):
        """
        Evaluate model metrics.

        Args:
            model (torch.Module): Model of interest.
            test_loader (DataLoader): DataLoader of test data.
        """
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for user_ids, champ_ids, ratings in test_loader:
                preds = model(user_ids, champ_ids)
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
            model.train()
            total_loss = 0
            with torch.no_grad():
                for user_ids, champ_ids, ratings in train_loader:
                    preds = model(user_ids, champ_ids)
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
        num_factors: int = 5,
        epochs: int = 20,
    ) -> list[str]:
        """
        Given a puuid, returns an ordered list of recommended champions.

        Args:
            puuid (str): Puuid of interest.
            test_size (float, optional): Proportion of test data. Defaults to 0.2.
            num_factors (int, optional): Number of latent factors. Defaults to 5.
            epochs (int, optional): Number of training epochs. Defaults to 20.

        Returns:
            list[str]: List of champions.
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

        # Necessarily uses the entire train and test datasets
        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            shuffle=True,
        )
        test_loader = MultiEpochsDataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            shuffle=True,
        )

        num_summoners = len(df["puuid"].unique())
        num_champions = len(df["champ_id"].unique())

        model = DotProduct(num_summoners, num_champions, num_factors).to(self.device)
        self.train_and_evaluate_model(model, train_loader, test_loader, epochs)

        # Get ratings for given user
        user_idx = le_user.transform([puuid])
        all_champions = torch.tensor([champ_idx for champ_idx in range(num_champions)])
        user_id = torch.tensor(list(user_idx) * len(all_champions))
        predicted_ratings = model(user_id, all_champions)

        champ_order = torch.argsort(predicted_ratings, descending=True)
        return list(le_champion.inverse_transform(champ_order.numpy()))
