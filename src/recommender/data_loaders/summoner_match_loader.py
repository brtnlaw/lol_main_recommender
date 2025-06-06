import json
import os
import pickle as pkl
import time
from collections import defaultdict

from .summoner_data_loader import SummonerDataLoader


class SummonerMatchLoader(SummonerDataLoader):
    """Class for generating user data with champions played."""

    def __init__(self):
        """Initializes an API Wrapper to ping Riot's API."""
        self.project_root = os.getenv("PROJECT_ROOT")
        super().__init__(
            puuid_json_path=os.path.join(
                self.project_root,
                f"data/cache/pending_match_puuids.json",
            ),
            json_folder_path=os.path.join(
                self.project_root, "data/summoner_match_pkls/"
            ),
        )

    def dump_matches_for_puuid(
        self, puuid: str, ct: int = 25, overwrite: bool = False
    ) -> None:
        """
        Given a puuid, dumps ranked match info into a json.

        Args:
            puuid (str): Puuid of interest.
            ct (int, optional): Number of matches. Defaults to 25.
            overwrite (bool, optional): Whether or not to overwrite the puuid. Defaults to False.
        """
        pkl_path = os.path.join(
            self.project_root,
            f"src/summoner_match_pkls/{puuid}.pkl",
        )
        if os.path.exists(pkl_path) and not overwrite:
            return

        summoner_dict = defaultdict(int)
        print(f"Loading data for puuid: {puuid}")
        match_ids = self.riot_api_helper.get_player_matches(puuid, ct)

        for i in range(len(match_ids)):
            match_id = match_ids[i]
            print(f"Match {i} loaded, id: {match_id}...")
            match = self.riot_api_helper.get_match_info(match_id)
            try:
                participants = match.get("info", {}).get("participants", [])
            except:
                continue
            for participant in participants:
                match_puuid = participant.get("puuid")
                if match_puuid == puuid:
                    champion_name = participant.get("championName")
                    win = participant.get("win")
                    kills = participant.get("kills")
                    deaths = participant.get("deaths")
                    assists = participant.get("assists")

                    summoner_dict[champion_name] += 1
                    summoner_dict[f"{champion_name}_kills"] += kills
                    summoner_dict[f"{champion_name}_assists"] += assists
                    summoner_dict[f"{champion_name}_deaths"] += deaths
                    if win:
                        summoner_dict[f"{champion_name}_wins"] += 1
                elif (
                    match_puuid not in self.processed_puuids and len(match_puuid) == 78
                ):
                    self.pending_puuids.add(match_puuid)
                    self.save_puuid_json()
            time.sleep(0.05)

        with open(pkl_path, "wb") as f:
            pkl.dump(dict(summoner_dict), f)
        print(f"Successfully saved data for puuid: {puuid}")


if __name__ == "__main__":
    """To generate further data"""
    # python -m src.recommender.data_loaders.summoner_match_loader
    instance = SummonerMatchLoader()
    instance.loop_puuid_data()
