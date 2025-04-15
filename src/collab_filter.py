import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch import optim
from torch.utils.data import DataLoader, Dataset


class ChampionsDataset(Dataset):
    """Custom Dataset with puuid, champion, and rating."""

    def __init__(self, df):
        """Dummy for inheritance."""
        self.df = df

    def __len__(self):
        """Dummy for inheritance."""
        return len(self.df)

    def __getitem__(self, idx):
        """Gets puuid, champion, and rating."""
        puuid = torch.tensor(self.df.iloc[idx]["user_id"], dtype=torch.long)
        champion = torch.tensor(self.df.iloc[idx]["champ_id"], dtype=torch.long)
        rating = torch.tensor(self.df.iloc[idx]["rating"], dtype=torch.float)
        return puuid, champion, rating


class DotProduct(nn.Module):
    """Model that dot products the summoner and champion factors"""

    def __init__(self, num_summoners, num_champions, num_factors):
        """Module with summoner and champion factors as Embedding objects."""
        super().__init__()
        self.summoner_factors = nn.Embedding(num_summoners, num_factors)
        self.champion_factors = nn.Embedding(num_champions, num_factors)

    def forward(self, summoner_ids, champ_ids):
        """Simply takes the dot product at each forward step."""
        summoner_embedded = self.summoner_factors(summoner_ids)
        champion_embedded = self.champion_factors(champ_ids)
        return (summoner_embedded * champion_embedded).sum(dim=1)


def evaluate_model(model: nn.Module, test_loader: DataLoader):
    """
    Evaluate model metrics.

    Args:
        model (torch.Module): Model of interest.
        test_loader (DataLoader): DataLoader of test data.
    """
    # Model evaluation mode with no gradient
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
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
):
    """
    Trains and evaluates a model for a given number of epochs.

    Args:
        model (nn.Module): Model of interest.
        train_loader (DataLoader): DataLoader of train data.
        test_loader (DataLoader): DataLoader of test data.
        epochs (int, optional): Number of epochs. Defaults to 50.
    """
    optimizer = optim.SGD(model.parameters())
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        # Training mode and reset loss
        model.train()
        total_loss = 0
        for user_ids, champ_ids, ratings in train_loader:
            # Zero out the gradient, make predictions, backward propagate the losses, and step forward with the optimizer
            optimizer.zero_grad()
            preds = model(user_ids, champ_ids)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
        evaluate_model(model, test_loader)
