from .als_collab_filter import ALSCollabFilter
from .common import BaseRecommender
from .content_based_filter import ContentBasedFilter


class SimpleHybridFilter(BaseRecommender):
    """Simple hybrid that combines content-based filtering with user-item collaboration filtering."""

    def __init__(self):
        super().__init__()
        self.als_collab_filter = ALSCollabFilter()
        self.content_based_filter = ContentBasedFilter()
        self.weight = 0.5

    def get_predicted_ratings(self, puuid: str):
        als_preds = self.als_collab_filter.get_predicted_ratings(puuid)
        cbf_preds = self.content_based_filter.get_predicted_ratings(puuid)

        als_inv = {champ: rating for rating, champ in als_preds}
        cbf_inv = {champ: rating for rating, champ in cbf_preds}

        hybrid_pred_dict = {}
        for champ in als_inv.keys():
            hybrid_pred_dict[
                self.weight * als_inv[champ] + (1 - self.weight) * cbf_inv[champ]
            ] = champ

        return hybrid_pred_dict
