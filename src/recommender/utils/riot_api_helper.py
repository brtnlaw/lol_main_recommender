import asyncio
import os
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from aiolimiter import AsyncLimiter


class RiotApiHelper:
    """Class that handles all of the API retrieval with Riot."""

    def __init__(self):
        """Initializes the API Key and defaults to NA/America."""
        self.riot_api_key = os.getenv("RIOT_API_KEY")
        self.headers = {"X-Riot-Token": self.riot_api_key}
        self.region = "na1"
        self.match_region = "americas"
        self.mastery_region = "NA1"
        self.short_rate_limit = AsyncLimiter(49, 10)
        self.long_rate_limit = AsyncLimiter(29999, 600)

    def get_challenger_puuids(
        self, queue: str = "RANKED_SOLO_5x5", ct: int = 200
    ) -> List[str]:
        """
        Gets puuids from api of the top Challenger players in ranked solo.

        Args:
            queue (str, optional): Type of queue. Defaults to "RANKED_SOLO_5x5".
            ct (int, optional): Number of Challenger players. Defaults to 200.

        Returns:
            List[str]: List of puuids.
        """
        url = f"https://{self.region}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{queue}"
        res = requests.get(url, headers=self.headers)
        if res.status_code == 200:
            data = res.json()
            entries = data.get("entries", [])
            entries = sorted(entries, key=lambda x: x["leaguePoints"], reverse=True)[
                :ct
            ]
            return [entry["puuid"] for entry in entries]
        else:
            print(f"Error fetching data: {res.status_code} - {res.text}")
            return []

    def get_player_matches(self, puuid: str, ct: int = 25) -> List[str]:
        """
        Pulls ranked match ids for a given puuid.

        Args:
            puuid (str): User puuid.
            ct (int, optional): Number of matches per player. Defaults to 25.

        Returns:
            List[str]: List of match ids.
        """
        url = f"https://{self.match_region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count={ct}&queue=420"
        res = requests.get(url, headers=self.headers)
        if res.status_code == 200:
            return res.json()
        else:
            print(f"Error fetching data: {res.status_code} - {res.text}")
            return None

    def get_match_info(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets the match information for a given match id.

        Args:
            match_id (str): Match identifier

        Returns:
            Optional[Dict[str, Any]]: Json results of match.
        """
        url = f"https://{self.match_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        res = requests.get(url, headers=self.headers)
        if res.status_code == 200:
            return res.json()
        else:
            print(f"Error fetching data: {res.status_code} - {res.text}")
            return None

    def get_puuid_from_summoner_id(self, summoner_id: str) -> int:
        """
        Given a summoner ID, returns the puuid.

        Args:
            summoner_id (str): Summoner ID of the form ABC#NA1.

        Returns:
            int: Puuid of the summoner.
        """
        game_name, tag_line = summoner_id.split("#")
        url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        res = requests.get(url, headers=self.headers)
        if res.status_code == 200:
            puuid = res.json()["puuid"]
            return puuid
        else:
            print(f"Error fetching data: {res.status_code} - {res.text}")
            return 0

    def get_player_mastery(
        self,
        puuid: str,
    ):
        url = f"https://{self.mastery_region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}"
        res = requests.get(url, headers=self.headers)
        if res.status_code == 200:
            return res.json()
        else:
            print(f"Error fetching data: {res.status_code} - {res.text}")
            return None

    def get_player_rank(self, puuid: str):
        url = f"https://{self.mastery_region}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
        res = requests.get(url, headers=self.headers)
        if res.status_code == 200:
            return res.json()
        else:
            print(f"Error fetching data: {res.status_code} - {res.text}")
            return None

    async def async_get_match_info(self, session, match_id, semaphore, max_retries=3):
        url = f"https://{self.match_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        async with semaphore:
            for _ in range(max_retries):
                try:
                    async with self.short_rate_limit:
                        async with self.long_rate_limit:
                            async with session.get(url, headers=self.headers) as resp:
                                if resp.status == 429:
                                    retry_after = int(
                                        resp.headers.get("Retry-After", 1)
                                    )
                                    print(
                                        f"Rate limited. Retry after {retry_after} seconds..."
                                    )
                                    await asyncio.sleep(retry_after)
                                    continue
                                elif resp.status != 200:
                                    print(f"Failed with status {resp.status}")
                                    return None
                                print(f"Match {match_id} loaded successfully.")
                                return await resp.json()
                except aiohttp.ClientError as e:
                    print(f"ClientError: {e}")
                    await asyncio.sleep(1)
            print(f"Failed after {max_retries} attempts.")
            return None
