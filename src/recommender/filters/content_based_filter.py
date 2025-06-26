from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder
from torch.nn.functional import cosine_similarity

from .common import BaseRecommender


class ContentBasedFilter(BaseRecommender):
    """Filter that recommends based on champion metadata."""

    def __init__(self):
        """Loads metadata encoders."""
        super().__init__()
        self.multi_label_binarizer = MultiLabelBinarizer()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.min_max_scaler = MinMaxScaler()

    def get_champ_df(self) -> pd.DataFrame:
        """
        Constructs DataFrame of champions and attributes.

        Returns:
            pd.DataFrame: Champion attribute DataFrame.
        """
        metadata = self.map_helper.get_lolstaticdata_champ_id_mapping()
        champ_dict = defaultdict(dict)

        for key in metadata.keys():
            champ_dict[key]["resource"] = metadata[key]["resource"]
            champ_dict[key]["attack_type"] = metadata[key]["attackType"]
            champ_dict[key]["adaptive_type"] = metadata[key]["adaptiveType"]
            champ_dict[key]["positions"] = metadata[key]["positions"]
            champ_dict[key]["roles"] = metadata[key]["roles"]
            champ_dict[key]["damage"] = metadata[key]["attributeRatings"]["damage"]
            champ_dict[key]["toughness"] = metadata[key]["attributeRatings"][
                "toughness"
            ]
            champ_dict[key]["control"] = metadata[key]["attributeRatings"]["control"]
            champ_dict[key]["mobility"] = metadata[key]["attributeRatings"]["mobility"]
            champ_dict[key]["utility"] = metadata[key]["attributeRatings"]["utility"]
            champ_dict[key]["ability_reliance"] = metadata[key]["attributeRatings"][
                "abilityReliance"
            ]
            champ_dict[key]["difficulty"] = metadata[key]["attributeRatings"][
                "difficulty"
            ]

        champ_df = pd.DataFrame(champ_dict).transpose()
        tags_one_hot = pd.DataFrame(
            self.multi_label_binarizer.fit_transform(champ_df["roles"]),
            columns=[
                f"tag_{class_name}"
                for class_name in self.multi_label_binarizer.classes_
            ],
            index=champ_df.index,
        )
        pos_one_hot = pd.DataFrame(
            self.multi_label_binarizer.fit_transform(champ_df["positions"]),
            columns=[
                f"positions_{class_name}"
                for class_name in self.multi_label_binarizer.classes_
            ],
            index=champ_df.index,
        )
        champ_df["resource"] = champ_df["resource"].where(
            champ_df["resource"].isin(["MANA", "ENERGY", "HEALTH", "NONE"]), "OTHER"
        )
        champ_df = pd.concat(
            [champ_df.drop(columns=["positions", "roles"]), tags_one_hot, pos_one_hot],
            axis=1,
        )

        cat_cols = self.one_hot_encoder.fit_transform(
            champ_df[["resource", "attack_type", "adaptive_type"]]
        )
        cat_col_names = self.one_hot_encoder.get_feature_names_out(
            ["resource", "attack_type", "adaptive_type"]
        )
        champ_df = pd.concat(
            [
                champ_df.drop(columns=["resource", "attack_type", "adaptive_type"]),
                pd.DataFrame(cat_cols, columns=cat_col_names, index=champ_df.index),
            ],
            axis=1,
        )
        champ_df = pd.DataFrame(
            self.min_max_scaler.fit_transform(champ_df),
            columns=champ_df.columns,
            index=champ_df.index,
        )
        return champ_df

    def get_predicted_ratings(self, puuid: str) -> dict:
        """
        Recommends champions based on metadata and existing user play.

        Args:
            puuid (str): Puuid of interest.

        Returns:
            dict: Rating to champion dict.
        """
        champ_df = self.get_champ_df()
        summoner_dict = self.summoner_mastery_loader.load_dict_from_pkl(puuid)
        id_champ_map = self.map_helper.get_id_champ_map()

        summoner_prop_dict = {}
        for key, value in summoner_dict.items():
            champ_id_str, suffix = key.split("_", 1)
            champ_name = id_champ_map.get(int(champ_id_str), champ_id_str)
            if suffix == "points":
                summoner_prop_dict[f"{champ_name}_{suffix}"] = value
        tot_points = sum(summoner_prop_dict.values())
        for key, value in summoner_prop_dict.items():
            summoner_prop_dict[key] = value / tot_points

        # Possibly out of order
        champion_names = [
            champ_key.replace("_points", "") for champ_key in summoner_prop_dict.keys()
        ]
        user_prefs = np.average(
            champ_df.loc[champion_names].values,
            axis=0,
            weights=list(summoner_prop_dict.values()),
        )
        user_tensor = torch.tensor(user_prefs.astype(np.float64), dtype=torch.float64)
        champ_tensor = torch.tensor(
            champ_df.values.astype(np.float64), dtype=torch.float64
        )
        sim = cosine_similarity(user_tensor, champ_tensor)
        champ_order = torch.argsort(sim, descending=True)
        # Avoid ties
        epsilon = 1e-6
        for i in range(len(sim)):
            sim[i] -= i * epsilon
        predicted_ratings_dict = {
            sim[champ.item()].item(): champ_df.index[champ.item()]
            for champ in champ_order
        }
        return predicted_ratings_dict
