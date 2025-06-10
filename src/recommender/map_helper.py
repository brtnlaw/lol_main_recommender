import json
import os
from typing import Optional

import requests

from .riot_api_helper import RiotApiHelper

PROJECT_ROOT = os.getenv("PROJECT_ROOT")


class MapHelper:
    """Simple class to assist and store various mappings."""

    def __init__(self):
        """Initializes Riot API helper class."""
        self.riot_api_helper = RiotApiHelper()

    # NOTE: For acquiring champion metadata
    def get_riot_champ_data(self, version="15.7.1") -> Optional[dict]:
        """
        Grabs the champion mapping from Riot.

        Args:
            version (str, optional): Version of the data. Defaults to "15.7.1".

        Returns:
            Optional[dict]: Returns the entire mapping if available.
        """
        url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
        file_path = os.path.join(PROJECT_ROOT, "data/cache/champ_id_mapping.json")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return json.load(f)

        response = requests.get(url)
        if response.status_code == 200:
            with open(
                file_path,
                "wb",
            ) as f:
                f.write(response.content)
                print("Champ_id map downloaded successfully.")
            with open(file_path, "rb") as f:
                return json.load(f)
        else:
            print("Failed to retrieve the file. Status code:", response.status_code)
            return None

    def get_id_champ_map(self) -> dict:
        """
        Gets the id to champion map.

        Returns:
            dict: Id to champion map.
        """
        riot_dict = self.get_riot_champ_data()
        id_champ_map = {}
        for _, champ_info in riot_dict["data"].items():
            champ_id = int(champ_info["key"])
            champ_name = champ_info["id"]
            id_champ_map[champ_id] = champ_name
        return id_champ_map

    def get_puuid_mapping(self, summoner_id: str = None) -> int:
        """
        Prompts user for summoner info. If not in json, grabs info and dumps to mapping json. Then returns puuid.

        Returns:
            int: Puuid for summoner.
        """
        file_path = os.path.join(PROJECT_ROOT, "data/cache/summoner_puuid_mapping.json")
        if not summoner_id:
            summoner_id = input("Input your summoner name and tag (Faker#NA1): \n")
        if not os.path.exists(file_path):
            print("Mapping does not exist. Generating...")
            puuid = self.riot_api_helper.get_puuid_from_summoner_id(summoner_id)
            summoner_puuid_dict = {summoner_id: puuid}
            with open(file_path, "w") as f:
                json.dump(summoner_puuid_dict, f, indent=2)
            print("Successfully generated json.")
        else:
            with open(file_path, "r") as f:
                summoner_puuid_dict = json.load(f)
            if summoner_id in summoner_puuid_dict:
                puuid = summoner_puuid_dict.get(summoner_id)
            else:
                puuid = self.riot_api_helper.get_puuid_from_summoner_id(summoner_id)
                summoner_puuid_dict[summoner_id] = puuid
                with open(file_path, "w") as f:
                    json.dump(summoner_puuid_dict, f, indent=2)
                print("Successfully added mapping.")
        return puuid
