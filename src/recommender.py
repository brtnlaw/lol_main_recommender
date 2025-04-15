import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from collab_filter import ChampionsDataset, DotProduct, train_and_evaluate_model
from map_helper import MapHelper
from summoner_data import SummonerData

PROJECT_ROOT = os.getenv("PROJECT_ROOT")


class Recommender:
    """Class to encapsulate the recommendation."""

    def recommend_champions(self):
        """Prompts summoner id and then recommends three champions."""
        map_helper = MapHelper()
        summoner_data = SummonerData()
        json_rewrite = False

        # Prompt in game name
        puuid = map_helper.get_puuid_mapping()

        # Check if data exists as json
        puuid_path = os.path.join(PROJECT_ROOT, f"src/summoner_data/{puuid}.json")
        if not os.path.exists(puuid_path):
            json_rewrite = True
            summoner_data.dump_matches_for_puuid(puuid)

        # Get the rating data
        df, le_user, le_champion = summoner_data.load_clean_df_encoders(json_rewrite)

        # Train a model
        train_df, test_df = train_test_split(df, test_size=0.2)
        train_dataset = ChampionsDataset(train_df)
        test_dataset = ChampionsDataset(test_df)
        train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=25, shuffle=True)

        num_summoners = len(df["puuid"].unique())
        num_champions = len(df["champ_id"].unique())
        num_factors = 10

        model = DotProduct(num_summoners, num_champions, num_factors)
        train_and_evaluate_model(model, train_loader, test_loader, epochs=100)

        # Get relevant user index
        user_idx = le_user.transform([puuid])

        # Get preds
        all_champions = torch.tensor([champ_idx for champ_idx in range(num_champions)])
        user_id = torch.tensor(list(user_idx) * len(all_champions))
        predicted_ratings = model(user_id, all_champions)
        _, top_indices = torch.topk(predicted_ratings, k=3)
        top_indices_list = top_indices.tolist()

        # Load champ order
        champ_map = map_helper.get_champ_id_to_name()
        champ_order = [
            champ_map.get(str(x))
            for x in le_champion.inverse_transform(range(num_champions))
        ]
        for idx in top_indices_list:
            print(f"WE PRESCRIBE THAT YOU PLAY: {champ_order[idx]}")


if __name__ == "__main__":
    # python src/recommender.py
    recommender = Recommender()
    recommender.recommend_champions()
