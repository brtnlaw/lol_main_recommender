import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch import optim
from torch.utils.data import Dataset


class ChampionsDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        puuid = torch.tensor(self.df.iloc[idx]["user_id"], dtype=torch.long)
        champion = torch.tensor(self.df.iloc[idx]["champ_id"], dtype=torch.long)
        rating = torch.tensor(self.df.iloc[idx]["rating"], dtype=torch.float)
        return puuid, champion, rating


class DotProduct(nn.Module):
    def __init__(self, num_summoners, num_champions, num_factors):
        super().__init__()
        self.summoner_factors = nn.Embedding(num_summoners, num_factors)
        self.champion_factors = nn.Embedding(num_champions, num_factors)

    def forward(self, summoner_ids, champ_ids):
        summoner_embedded = self.summoner_factors(summoner_ids)
        champion_embedded = self.champion_factors(champ_ids)
        return torch.sigmoid((summoner_embedded * champion_embedded).sum(dim=1))


def evaluate_model(model, test_loader):
    # Turns on eval mode
    model.eval()
    y_pred, y_true = [], []
    # Just a mode that turns off gradients
    with torch.no_grad():
        for user_ids, champ_ids, ratings in test_loader:
            preds = model(user_ids, champ_ids)
            y_pred.extend(preds.numpy())
            y_true.extend(ratings.numpy())
        mse_rounded = mean_squared_error(y_true, np.rint(y_pred))
        print(f"Test MSE (Rounded Predictions): {mse_rounded}")


def train_and_evaluate_model(model, train_loader, test_loader, epochs=10):
    optimizer = optim.SGD(model.parameters())
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        # Model training persists outside of function
        model.train()
        total_loss = 0
        for user_ids, champ_ids, ratings in train_loader:
            optimizer.zero_grad()
            preds = model(user_ids, champ_ids)
            loss = criterion(preds, ratings)
            # Identify which parameters i.e. embeddings to adjust via backpropogation then step forward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
        evaluate_model(model, test_loader)
