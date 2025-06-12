import torch

from .filters.als_collab_filter import ALSCollabFilter
from .filters.content_based_filter import ContentBasedFilter
from .filters.sgd_collab_filter import SGDCollabFilter


class Recommender:
    """Class to encapsulate the recommendation process."""

    def __init__(self, filter_str):
        assert filter_str in [
            "sgd_collab",
            "als_collab",
            "content_based",
        ], "Not valid filter."
        match filter_str:
            case "sgd_collab":
                self.filter = SGDCollabFilter
            case "als_collab":
                self.filter = ALSCollabFilter
            case "content_based":
                self.filter = ContentBasedFilter

    def recommend_champions(self) -> None:
        """Prompts summoner id and then recommends three champions."""
        puuid = self.map_helper.get_puuid_mapping()

        predicted_ratings, le_champion = self.filter.recommend_champions(puuid)

        _, top_indices = torch.topk(predicted_ratings, k=3)
        top_indices_list = top_indices.tolist()
        _, bottom_indices = torch.topk(-predicted_ratings, k=3)
        bottom_indices_list = bottom_indices.tolist()

        champ_order = [
            x for x in le_champion.inverse_transform(range(le_champion.classes_))
        ]
        print("=================================")
        top_ranking = 1
        bottom_ranking = 1
        for idx in top_indices_list:
            print(f"You should try playing {top_ranking}: {champ_order[idx]}")
            top_ranking += 1
        print("=================================")
        for idx in bottom_indices_list:
            print(f"You should avoid playing {bottom_ranking}: {champ_order[idx]}")
            bottom_ranking += 1
        print("=================================")


if __name__ == "__main__":
    # python -m src.recommender.recommender
    recommender = Recommender()
    recommender.recommend_champions()
