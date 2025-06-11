import json
import os
import pickle as pkl
from abc import abstractmethod

from ..utils.riot_api_helper import RiotApiHelper


class SummonerDataLoader:
    """Class for generating user data with champions played."""

    def __init__(self, puuid_json_path, json_folder_path):
        """Initializes an API Wrapper to ping Riot's API."""
        self.puuid_json_path = puuid_json_path
        self.json_folder_path = json_folder_path
        os.makedirs(self.json_folder_path, exist_ok=True)

        self.riot_api_helper = RiotApiHelper()

        self.pending_puuids = set()
        self.processed_puuids = set()
        self.load_puuid_json()
        self.load_processed_puuids()
        if not self.pending_puuids and self.riot_api_helper:
            challenger_puuids = self.riot_api_helper.get_challenger_puuids()
            self.pending_puuids.update(challenger_puuids)
            self.save_puuid_json()

    def load_processed_puuids(self) -> None:
        """Adds all the already-proecssed puuids to the self.processed_puuids."""
        for json_file in os.listdir(self.json_folder_path):
            puuid = json_file.split(".json")[0]
            self.processed_puuids.add(puuid)

    def load_puuid_json(self) -> None:
        """Loads the pending puuids from json."""
        try:
            with open(self.puuid_json_path, "r") as f:
                data = json.load(f)
                self.pending_puuids = set(data.get("pending", []))
        except FileNotFoundError:
            self.pending_puuids = set()

    def save_puuid_json(self) -> None:
        """Updates the pending puuids to json from current self.pending_puuids."""
        data = {
            "pending": list(self.pending_puuids),
        }
        with open(self.puuid_json_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_num_puuids(self) -> int:
        """Simple function to retrieve the number of saved summoner pkls."""
        num_puuids = len(
            [
                f
                for f in os.listdir(self.json_folder_path)
                if os.path.isfile(os.path.join(self.json_folder_path, f))
            ]
        )
        return num_puuids

    @abstractmethod
    def dump_data_for_puuid(self, puuid: str, *args) -> None:
        """
        Dumps data for a puuid into a pkl.

        Args:
            puuid (int): Puuid of interest.
        """
        pass

    def loop_puuid_data(self) -> None:
        """
        Loops through puuid data generation.
        """
        num_puuids = self.get_num_puuids()
        while num_puuids < 5000:
            puuid = next(iter(self.pending_puuids))
            if not self.pending_puuids:
                print("Out of puuids!")
                break
            self.dump_data_for_puuid(puuid)
            self.pending_puuids.discard(puuid)
            self.processed_puuids.add(puuid)
            num_puuids += 1
            if num_puuids % 5 == 0:
                print(f"Have stored data for {num_puuids} puuids.")

    def load_dict_from_pkl(self, puuid: str) -> dict:
        """
        Simple wrapper to load individual puuid dictionary.

        Args:
            puuid (int): Puuid of interest

        Raises:
            Exception: If puuid not loaded, one needs to load first.

        Returns:
            dict: Player's dictionary.
        """
        pkl_path = os.path.join(self.json_folder_path, f"{puuid}.pkl")
        try:
            with open(pkl_path, "rb") as f:
                return pkl.load(f)
        except Exception:
            raise Exception("Puuid data not loaded. Load first or check the file.")
