from .filters.als_collab_filter import ALSCollabFilter
from .filters.content_based_filter import ContentBasedFilter
from .filters.sgd_collab_filter import SGDCollabFilter
from .filters.simple_hybrid import SimpleHybridFilter
from .utils.map_helper import MapHelper


class Recommender:
    """Class to encapsulate the recommendation process."""

    def __init__(self, filter_str):
        """Sets up the filter type."""
        assert filter_str in [
            "sgd_collab",
            "als_collab",
            "content_based",
            "simple_hybrid",
        ], "Not valid filter."
        match filter_str:
            case "sgd_collab":
                self.filter = SGDCollabFilter()
            case "als_collab":
                self.filter = ALSCollabFilter()
            case "content_based":
                self.filter = ContentBasedFilter()
            case "simple_hybrid":
                self.filter = SimpleHybridFilter()
        self.map_helper = MapHelper()

    def recommend_champions(self) -> None:
        """Prompts summoner id and then recommends three champions."""
        puuid = self.map_helper.get_puuid_mapping("BSIZZLEMONEY#0000")
        self.filter.recommend_champions(puuid)


if __name__ == "__main__":
    # python -m src.recommender.recommender
    recommender = Recommender("simple_hybrid")
    recommender.recommend_champions()
