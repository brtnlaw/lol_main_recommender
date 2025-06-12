import os
import pickle as pkl
import time
from collections import defaultdict

from .base_data_loader import BaseDataLoader


class SummonerMasteryLoader(BaseDataLoader):
    """Class for generating user mastery data."""

    def __init__(self):
        """Initializes an API Wrapper to ping Riot's API."""
        self.project_root = os.getenv("PROJECT_ROOT")
        super().__init__(
            puuid_json_path=os.path.join(
                self.project_root,
                f"data/cache/pending_mastery_puuids.json",
            ),
            json_folder_path=os.path.join(
                self.project_root, "data/summoner_mastery_pkls/"
            ),
        )

    def dump_data_for_puuid(self, puuid: str, overwrite: bool = False) -> None:
        """
        Given a puuid, dumps mastery info into a pkl.

        Args:
            puuid (str): Puuid of interest.
            overwrite (bool, optional): Whether or not to overwrite the puuid. Defaults to False.
        """
        pkl_path = os.path.join(self.json_folder_path, f"{puuid}.pkl")
        if os.path.exists(pkl_path) and not overwrite:
            return
        summoner_dict = defaultdict(int)
        print(f"Loading data for puuid: {puuid}")
        masteries = self.riot_api_helper.get_player_mastery(puuid)

        for champion_dict in masteries:
            champ_id = champion_dict.get("championId")
            summoner_dict[f"{champ_id}_level"] = champion_dict.get("championLevel")
            summoner_dict[f"{champ_id}_points"] = champion_dict.get("championPoints")
            summoner_dict[f"{champ_id}_last_play_time"] = champion_dict.get(
                "lastPlayTime"
            )

        time.sleep(0.05)
        with open(pkl_path, "wb") as f:
            pkl.dump(dict(summoner_dict), f)

        # Need snowballing of some fashion
        match_id = self.riot_api_helper.get_player_matches(puuid, 1)[0]
        match = self.riot_api_helper.get_match_info(match_id)
        participants = match.get("info", {}).get("participants", [])
        for participant in participants:
            match_puuid = participant.get("puuid")
            if match_puuid not in self.processed_puuids and len(match_puuid) == 78:
                self.pending_puuids.add(match_puuid)
                self.save_puuid_json()

        print(f"Successfully saved data for puuid: {puuid}")


if __name__ == "__main__":
    """To generate further data"""
    # python -m src.recommender.data_loaders.summoner_mastery_loader
    instance = SummonerMasteryLoader()
    instance.loop_puuid_data()
