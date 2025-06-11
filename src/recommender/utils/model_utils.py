import time

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader


class MultiEpochsDataLoader(DataLoader):
    """
    Better performance DataLoader over multiple epochs.
    Credit: https://github.com/huggingface/pytorch-image-models/pull/140
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Credit: https://github.com/huggingface/pytorch-image-models/pull/140

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


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
    """
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        evaluate_model(model, test_loader)
    print(f"Model training completed in {(time.time() - start_time)} seconds.")
