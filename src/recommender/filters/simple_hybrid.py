from .common import BaseRecommender


class SimpleHybrid(BaseRecommender):
    """Simple hybrid that combines content-based filtering with user-item collaboration filtering."""

    def __init__(self):
        super().__init__()

    def recommend_champions(
        self,
        puuid: str,
        test_size: float = 0.2,
        num_factors: int = 5,
        epochs: int = 20,
    ) -> list[str]:
        pass
