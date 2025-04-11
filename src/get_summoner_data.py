import json
import os
import pickle as pkl
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

from get_champ_id_mapping import get_champ_id_to_name

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


def json_to_df() -> pd.DataFrame:
    """
    Turns the folder of jsons into a useable DataFrame.

    Returns:
        pd.DataFrame: Combined json DataFrame.
    """
    pkl_path = os.path.join(PROJECT_ROOT, "src/data.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pkl.load(f)

    json_folder_path = os.path.join(PROJECT_ROOT, "src/summoner_data/")

    player_champions = defaultdict(
        lambda: defaultdict(int)
    )  # {player_id: {champion_id: count}}
    for json_file in os.listdir(json_folder_path):
        puuid = json_file.split(".json")[0]
        with open(os.path.join(json_folder_path, json_file), "r") as f:
            match_data = json.load(f)
        for match in match_data:
            participants = match.get("info", {}).get("participants", [])
            for participant in participants:
                if participant.get("puuid") == puuid:
                    # Store information to get rating
                    champion_id = participant.get("championId")
                    win = participant.get("win")
                    kills = participant.get("kills")
                    deaths = participant.get("deaths")
                    assists = participant.get("assists")

                    # rating = (win_percentage_weight * win_percentage) + (kda_weight * kda) + (matches_played_weight * matches_played)
                    player_champions[puuid][champion_id] += 1
                    player_champions[puuid][f"{champion_id}_kills"] += kills
                    player_champions[puuid][f"{champion_id}_assists"] += deaths
                    player_champions[puuid][f"{champion_id}_deaths"] += assists
                    if win == 1:
                        player_champions[puuid][f"{champion_id}_wins"] += 1
            player_champion_stats = defaultdict(lambda: defaultdict(int))

    # Insert own rating
    player_champion_stats = defaultdict(lambda: defaultdict(int))
    # calculate kda
    for puuid, champ_data in player_champions.items():
        for champion_id, count in champ_data.items():
            # If the key is equal to the id, can construct the stats
            if isinstance(champion_id, int):
                kills = player_champions[puuid][f"{champion_id}_kills"]
                assists = player_champions[puuid][f"{champion_id}_assists"]
                deaths = player_champions[puuid][f"{champion_id}_deaths"]
                wins = player_champions[puuid].get(f"{champion_id}_wins", 0)
                win_pct = (wins / count) * 100 if count > 0 else 0
                kda = (kills + assists) / (deaths if deaths != 0 else 1)

                player_champion_stats[puuid][champion_id] = kda + win_pct + count

    data_df = pd.DataFrame(player_champion_stats).transpose().fillna(0)
    with open(pkl_path, "wb") as f:
        pkl.dump(data_df, f)
    return data_df


def load_clean_df():
    # TODO: include champ name mapping, perhaps
    df = json_to_df()
    df.index.name = "puuid"
    df.reset_index(inplace=True)
    df = pd.melt(df, id_vars=["puuid"], var_name="orig_champ_id", value_name="rating")
    champ_map = get_champ_id_to_name()
    df["champion"] = df["orig_champ_id"].apply(lambda x: champ_map.get(str(x)))
    le_user = LabelEncoder()
    df["user_id"] = le_user.fit_transform(df["puuid"].values)
    le_champion = LabelEncoder()
    df["champ_id"] = le_champion.fit_transform(df["orig_champ_id"].values)
    return df


if __name__ == "__main__":
    # python src/get_summoner_data.py
    save_challenger_data()
