import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

API_KEY = os.getenv("RIOT_API_KEY")
PROJECT_ROOT = "/Users/brtnl/OneDrive/Desktop/code/lol_main_recommender"
HEADERS = {"X-Riot-Token": API_KEY}
REGION = "na1"
MATCH_REGION = "americas"


def get_challenger_puuids(queue: str = "RANKED_SOLO_5x5", ct: int = 200) -> List[str]:
    """
    Gets puuids from api of the top Challenger players in ranked solo.

    Args:
        queue (str, optional): Type of queue. Defaults to "RANKED_SOLO_5x5".
        ct (int, optional): Number of Challenger players. Defaults to 200.

    Returns:
        List[str]: List of puuids.
    """
    url = f"https://{REGION}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{queue}"
    res = requests.get(url, headers=HEADERS)
    data = res.json()
    entries = data.get("entries", [])
    entries = sorted(entries, key=lambda x: x["leaguePoints"], reverse=True)[:ct]
    return [entry["puuid"] for entry in entries]


def get_player_matches(puuid: str, ct: int = 100) -> List[str]:
    """
    Pulls the matches for a given puuid

    Args:
        puuid (str): User puuid.
        ct (int, optional): Number of matches per player. Defaults to 100.

    Returns:
        List[str]: List of match ids.
    """
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count={ct}"
    res = requests.get(url, headers=HEADERS)
    return res.json()


def get_match_info(match_id: str) -> Optional[Dict[str, Any]]:
    """
    Gets the match information for a given match id.

    Args:
        match_id (str): Match identifier

    Returns:
        Optional[Dict[str, Any]]: Json results of match.
    """
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    res = requests.get(url, headers=HEADERS)
    return res.json()


def save_challenger_data():
    """Loops through all the puuids of Challenger. Gets 100 matches per puuid and saves into a json in the summoner_data folder."""
    puuids = get_challenger_puuids()
    for puuid in puuids:
        file_path = os.path.join(
            PROJECT_ROOT,
            f"src/summoner_data/{puuid}.json",
        )
        if os.path.exists(file_path):
            continue
        match_data = []
        print(f"Loading data for puuid: {puuid}")
        match_ids = get_player_matches(puuid)
        # Each json is to have 100 games by default
        for i in range(len(match_ids)):
            match_id = match_ids[i]
            print(f"Match {i} loaded, id: {match_id}...")
            match_data.append(get_match_info(match_id))
            time.sleep(1)
        with open(file_path, "w") as f:
            json.dump(match_data, f, indent=2)
        print(f"Successfully saved data for puuid: {puuid}")


if __name__ == "__main__":
    # python src/get_summoner_data.py
    save_challenger_data()
