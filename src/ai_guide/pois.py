import json
import os
from typing import Dict, Any

def load_pois(pois_dir: str) -> Dict[str, Dict[str, Any]]:
    pois = {}
    if not os.path.isdir(pois_dir):
        return pois

    for fn in os.listdir(pois_dir):
        if fn.endswith(".json"):
            with open(os.path.join(pois_dir, fn), "r", encoding="utf-8") as f:
                poi = json.load(f)
            pois[poi["id"]] = poi
    return pois
