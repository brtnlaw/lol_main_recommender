import torch

from .filters.als_collab_filter import ALSCollabFilter
from .filters.content_based_filter import ContentBasedFilter
from .filters.sgd_collab_filter import SGDCollabFilter
from .utils.map_helper import MapHelper


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
                self.filter = SGDCollabFilter()
            case "als_collab":
                self.filter = ALSCollabFilter()
            case "content_based":
                self.filter = ContentBasedFilter()
        self.map_helper = MapHelper()

    def recommend_champions(self) -> None:
        """Prompts summoner id and then recommends three champions."""
        puuid = self.map_helper.get_puuid_mapping("BSIZZLEMONEY#0000")
        self.filter.recommend_champions(puuid)


if __name__ == "__main__":
    # python -m src.recommender.recommender
    recommender = Recommender("content_based")
    recommender.recommend_champions()
