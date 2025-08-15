import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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


class ChampionsFeaturesDataset(Dataset):
    """Custom Dataset for two-towers."""

    def __init__(self, df):
        # ===== Summoners ======
        self.ranks = torch.tensor(df["summoner_rank"].values, dtype=torch.long)
        self.lanes = torch.tensor(df["summoner_lane"].values, dtype=torch.long)
        self.sum_roles = torch.tensor(df["summoner_role"].values, dtype=torch.long)
        self.avg_kdas = torch.tensor(df["summoner_avg_kda"].values, dtype=torch.long)

        # ====== Champions ======
        self.attack_types = torch.tensor(
            df["champ_attack_type"].values, dtype=torch.long
        )
        self.adaptive_types = torch.tensor(
            df["champ_adaptive_type"].values, dtype=torch.long
        )
        self.resources = torch.tensor(df["champ_resource"].values, dtype=torch.long)

        # Encoded tensors
        self.roles = torch.tensor(np.stack(df["champ_roles"].values), dtype=torch.float)
        self.positions = torch.tensor(
            np.stack(df["champ_positions"].values), dtype=torch.float
        )

        # Desired rating
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float)

    def __len__(self):
        """Dummy for inheritance."""
        return len(self.ratings)

    def __getitem__(self, idx):
        """Gets puuid, champion, and rating."""
        summoner_features = (
            self.ranks[idx],
            self.lanes[idx],
            self.sum_roles[idx],
            # Needs to be [batch, 1]
            self.avg_kdas[idx].unsqueeze(-1),
        )
        champion_features = (
            self.attack_types[idx],
            self.adaptive_types[idx],
            self.resources[idx],
            self.roles[idx],
            self.positions[idx],
        )
        rating = self.ratings[idx]
        return summoner_features, champion_features, rating


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
