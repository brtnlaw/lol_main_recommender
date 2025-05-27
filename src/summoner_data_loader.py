import json
import os
import pickle as pkl
import time
from collections import defaultdict

from riot_api_helper import RiotApiHelper


class SummonerDataLoader:
    """Class for generating user data with champions played."""

    def __init__(self):
        """Initializes an API Wrapper to ping Riot's API."""
        self.riot_api_helper = RiotApiHelper()
        self.project_root = os.getenv("PROJECT_ROOT")
        self.puuid_json_path = os.path.join(
            self.project_root,
            f"src/cache/pending_puuids.json",
        )
        self.pending_puuids = set()
        self.processed_puuids = set()
        self.load_puuid_json()
        self.load_processed_puuids()
        if not self.pending_puuids and self.riot_api_helper:
            challenger_puuids = self.riot_api_helper.get_challenger_puuids()
            self.pending_puuids.update(challenger_puuids)
            self.save_puuid_json()

    def load_processed_puuids(self):
        json_folder_path = os.path.join(self.project_root, "src/summoner_pkls/")
        for json_file in os.listdir(json_folder_path):
            puuid = json_file.split(".json")[0]
            self.processed_puuids.add(puuid)

    def load_puuid_json(self):
        try:
            with open(self.puuid_json_path, "r") as f:
                data = json.load(f)
                self.pending_puuids = set(data.get("pending", []))
        except FileNotFoundError:
            self.pending_puuids = set()

    def save_puuid_json(self):
        data = {
            "pending": list(self.pending_puuids),
        }
        with open(self.puuid_json_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_num_puuids(self) -> int:
        """Simple function to retrieve the number of saved summoner pkls."""
        json_folder_path = os.path.join(self.project_root, "src/summoner_pkls/")
        num_puuids = len(
            [
                f
                for f in os.listdir(json_folder_path)
                if os.path.isfile(os.path.join(json_folder_path, f))
            ]
        )
        return num_puuids

    def dump_matches_for_puuid(
        self, puuid: str, ct: int = 25, overwrite: bool = False
    ) -> None:
        """
        Given a puuid, dumps match info into a json.

        Args:
            puuid (str): Puuid of interest.
            ct (int, optional): Number of matches. Defaults to 25.
            overwrite (bool, optional): Whether or not to overwrite the puuid. Defaults to False.
        """
        pkl_path = os.path.join(
            self.project_root,
            f"src/summoner_pkls/{puuid}.pkl",
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

    def loop_puuid_match_data(self):
        num_puuids = self.get_num_puuids()
        while num_puuids < 5000:
            puuid = next(iter(self.pending_puuids))
            if not self.pending_puuids:
                print("Out of puuids!")
                break
            self.dump_matches_for_puuid(puuid)
            self.pending_puuids.discard(puuid)
            self.processed_puuids.add(puuid)
            num_puuids += 1
            if num_puuids % 5 == 0:
                print(f"Have stored {num_puuids} puuids.")

    def load_dict_from_pkl(self, puuid: int) -> dict:
        """
        Simple wrapper to load individual puuid dictionary.

        Args:
            puuid (int): Puuid of interest

        Raises:
            Exception: If puuid not loaded, one needs to load first.

        Returns:
            dict: Player's match dictionary.
        """
        pkl_path = os.path.join(
            self.project_root,
            f"src/summoner_pkls/{puuid}.pkl",
        )
        try:
            with open(pkl_path, "rb") as f:
                return pkl.load(f)
        except Exception:
            raise Exception("Puuid data not loaded. Load first or check the file.")


if __name__ == "__main__":
    """To generate further data"""
    # python src/summoner_data_loader.py
    sd = SummonerDataLoader()
    sd.loop_puuid_match_data()
