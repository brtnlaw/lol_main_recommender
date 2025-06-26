import asyncio
import os
import pickle as pkl
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import aiohttp

from .base_data_loader import BaseDataLoader


def process_match_for_puuid(match: dict, puuid: str) -> tuple[dict, set]:
    if not match:
        return {}, set()
    summoner_dict = defaultdict(int)
    participants = match.get("info", {}).get("participants", [])
    new_puuids = set()
    for participant in participants:
        match_puuid = participant.get("puuid")
        if match_puuid == puuid:
            champion_name = participant.get("championName")
            win = participant.get("win")
            kills = participant.get("kills")
            deaths = participant.get("deaths")
            assists = participant.get("assists")
            for stat_key, value in {
                champion_name: 1,
                f"{champion_name}_kills": kills,
                f"{champion_name}_assists": assists,
                f"{champion_name}_deaths": deaths,
                f"{champion_name}_wins": int(win),
            }.items():
                summoner_dict[stat_key] += value
        else:
            new_puuids.add(match_puuid)
    return summoner_dict, new_puuids


class SummonerMatchLoader(BaseDataLoader):
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

    async def async_dump_data_for_puuid(self, puuid, overwrite=False, ct=25):
        pkl_path = os.path.join(
            self.project_root,
            f"data/summoner_match_pkls/{puuid}.pkl",
        )
        if os.path.exists(pkl_path) and not overwrite:
            print(f"Data already saved for puuid: {puuid}")
            with open(pkl_path, "rb") as file:
                return pkl.load(file)

        summoner_dict = defaultdict(int)
        print(f"Loading data for puuid: {puuid}")
        match_ids = self.riot_api_helper.get_player_matches(puuid, ct)

        # Async load match info given id
        semaphore = asyncio.Semaphore(5)
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.riot_api_helper.async_get_match_info(session, match_id, semaphore)
                for match_id in match_ids
            ]
            matches = await asyncio.gather(*tasks)

        # Parallelize matches
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_match_for_puuid, match, puuid)
                for match in matches
            ]
            results = [future.result() for future in futures]

        match_dicts = [result[0] for result in results]
        match_puuids = [result[1] for result in results]
        for match_puuid in match_puuids:
            if match_puuid not in self.processed_puuids and len(match_puuid) == 78:
                self.pending_puuids.add(match_puuid)
                self.save_puuid_json()

        summoner_dict = defaultdict(int)
        for match_dict in match_dicts:
            for key, value in match_dict.items():
                summoner_dict[key] += value
        with open(pkl_path, "wb") as f:
            pkl.dump(dict(summoner_dict), f)
            print(f"Successfully saved data for puuid: {puuid}")
        return summoner_dict


if __name__ == "__main__":
    """To generate further data"""
    # python -m src.recommender.data_loaders.summoner_match_loader
    instance = SummonerMatchLoader()
    instance.loop_puuid_data()
