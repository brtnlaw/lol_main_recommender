import json
import os

import requests

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")


def get_champ_id_mapping(version="15.7.1"):
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
    file_path = os.path.join(PROJECT_ROOT, "src/champ_id_mapping.json")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return json.load(f)

    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(PROJECT_ROOT, "champ_id_mapping.json"), "wb") as f:
            f.write(response.content)
            print("Champ_id map downloaded successfully.")
            return json.load(f)
    else:
        print("Failed to retrieve the file. Status code:", response.status_code)


def get_champ_id_to_name():
    champ_id_json = get_champ_id_mapping()
    champ_id_to_name = {}
    for champ_id, champ_info in champ_id_json["data"].items():
        champ_id_to_name[champ_info["key"]] = champ_info["id"]
    return champ_id_to_name
