import os
from abc import abstractmethod
from typing import Tuple

import pandas as pd
from sklearn.calibration import LabelEncoder


class SummonerDataProcessor:
    """Class for manipulating raw data and creating ratings per user."""

    def __init__(self):
        self.project_root = os.getenv("PROJECT_ROOT")

    @abstractmethod
    def aggregate_summoner_pkls(self, overwrite_aggregate: bool = False) -> dict:
        """
        Using all the summoner_pkls, constructs a nested dictionary of champion data per puuid. If pkl exists, simply loads.

        Args:
            overwrite_aggregate (bool, optional): Whether or not to rewrite the existing pkl file. Defaults to False.

        Returns:
            dict: Nested dictionary of champion data per puuid.
        """
        pass

    @abstractmethod
    def load_rating(
        self, overwrite_rating: bool = False, overwrite_aggregate: bool = False
    ) -> pd.DataFrame:
        """
        Loads the rating DataFrame with puuid, champion, and a proprietary rating system.

        Args:
            overwrite_rating (bool, optional): Whether or not to overwrite rating pkl. Defaults to False.
            overwrite_aggregate (bool, optional): Whether or not to overwrite the aggregate dict. Defaults to False.
        Returns:
            pd.DataFrame: Rating DataFrame.
        """
        pass

    def load_encoded_ratings(
        self, overwrite_rating: bool = False, overwrite_aggregate: bool = False
    ) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
        """
        Cleans up the rating DataFrame and encodes.

        Args:
            overwrite_rating (bool, optional): Whether or not to overwrite rating pkl. Defaults to False.
            overwrite_aggregate (bool, optional): Whether or not to overwrite the aggregate dict. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]: Cleaned DataFrame, user encoder, champion encoder.
        """
        rating_df = self.load_rating(overwrite_rating, overwrite_aggregate)
        rating_df.index.name = "puuid"
        rating_df.reset_index(inplace=True)
        rating_df = pd.melt(
            rating_df, id_vars=["puuid"], var_name="champ_name", value_name="rating"
        )
        rating_df.rename(columns={"champ_name": "champion"})
        le_user = LabelEncoder()
        rating_df["user_id"] = le_user.fit_transform(rating_df["puuid"].values)
        le_champion = LabelEncoder()
        rating_df["champ_id"] = le_champion.fit_transform(
            rating_df["champ_name"].values
        )
        return rating_df, le_user, le_champion
