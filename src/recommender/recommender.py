import os

import torch
from sklearn.model_selection import train_test_split

from .collab_filter import ChampionsDataset, DotProduct, train_and_evaluate_model
from .data_loaders.summoner_mastery_loader import SummonerMasteryLoader
from .data_processors.summoner_mastery_processor import SummonerMasteryProcessor
from .map_helper import MapHelper
from .multi_epochs_data_loader import MultiEpochsDataLoader

PROJECT_ROOT = os.getenv("PROJECT_ROOT")


class Recommender:
    """Class to encapsulate the recommendation."""

    def __init__(self):
        self.map_helper = MapHelper()
        self.summoner_mastery_loader = SummonerMasteryLoader()
        self.summoner_mastery_processor = SummonerMasteryProcessor()

    def recommend_champions(self):
        """Prompts summoner id and then recommends three champions."""
        overwrite = False

        # Prompt in game name
        puuid = self.map_helper.get_puuid_mapping()

        # Check if data exists as json
        puuid_path = os.path.join(
            PROJECT_ROOT, f"data/summoner_mastery_pkls/{puuid}.pkl"
        )
        if not os.path.exists(puuid_path):
            overwrite = True
            self.summoner_mastery_loader.dump_data_for_puuid(puuid)

        # Get the rating data
        df, le_user, le_champion = self.summoner_mastery_processor.load_encoded_ratings(
            overwrite, overwrite
        )

        # TODO: move to collab_filter.py
        # Train a model
        train_df, test_df = train_test_split(df, test_size=0.2)
        train_dataset = ChampionsDataset(train_df)
        test_dataset = ChampionsDataset(test_df)
        num_workers = 3

        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=10,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
        test_loader = MultiEpochsDataLoader(
            test_dataset,
            batch_size=10,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
        )

        num_summoners = len(df["puuid"].unique())
        num_champions = len(df["champ_id"].unique())
        num_factors = 5

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = DotProduct(num_summoners, num_champions, num_factors).to(device)
        train_and_evaluate_model(model, train_loader, test_loader, epochs=20)

        # Get relevant user index
        user_idx = le_user.transform([puuid])

        # Get preds
        all_champions = torch.tensor([champ_idx for champ_idx in range(num_champions)])
        user_id = torch.tensor(list(user_idx) * len(all_champions))
        predicted_ratings = model(user_id, all_champions)
        _, top_indices = torch.topk(predicted_ratings, k=3)
        top_indices_list = top_indices.tolist()
        _, bottom_indices = torch.topk(-predicted_ratings, k=3)
        bottom_indices_list = bottom_indices.tolist()

        # Load champ order
        champ_order = [x for x in le_champion.inverse_transform(range(num_champions))]
        print("=================================")
        ranking = 1
        for idx in top_indices_list:
            print(f"You should try playing {ranking}: {champ_order[idx]}")
            ranking += 1
        print("=================================")
        ranking = 1
        for idx in bottom_indices_list:
            print(f"You should avoid playing {ranking}: {champ_order[idx]}")
            ranking += 1
        print("=================================")


if __name__ == "__main__":
    # python -m src.recommender.recommender
    recommender = Recommender()
    recommender.recommend_champions()
