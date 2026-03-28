#!/usr/bin/env python3
"""
Fetch CDX.NA.HY 5Y and CDX.NA.IG 5Y historical data from Cbonds API.
Saves to data/cdx_hy.json and data/cdx_ig.json for the dashboard to read.

Usage:
    CBONDS_LOGIN=you@example.com CBONDS_PASSWORD=xxx python scripts/fetch_cdx.py
"""

import os
import json
import sys
import requests
from datetime import datetime, timedelta

CBONDS_API = "https://ws.cbonds.info/services/json/get_index_value/?lang=eng"
LOGIN    = os.environ.get("CBONDS_LOGIN", "")
PASSWORD = os.environ.get("CBONDS_PASSWORD", "")

INDICES = [
    {"type_id": 204391, "name": "CDX.NA.HY 5Y", "file": "data/cdx_hy.json"},
    {"type_id": 204395, "name": "CDX.NA.IG 5Y", "file": "data/cdx_ig.json"},
]

YEARS_BACK = 5
PAGE_SIZE  = 1000


def fetch_all(type_id: int) -> list:
    date_from = (datetime.utcnow() - timedelta(days=365 * YEARS_BACK)).strftime("%Y-%m-%d")
    all_items = []
    offset = 0

    while True:
        payload = {
            "auth": {"login": LOGIN, "password": PASSWORD},
            "filters": [
                {"field": "type_id", "operator": "=", "value": type_id},
                {"field": "date",    "operator": ">=", "value": date_from},
            ],
            "quantity": {"limit": PAGE_SIZE, "offset": offset},
            "sorting": [{"field": "date", "order": "asc"}],
            "fields":  [{"field": "date"}, {"field": "value"}],
        }

        resp = requests.post(CBONDS_API, json=payload, timeout=30)
        resp.raise_for_status()
        body = resp.json()

        items = body.get("items", [])
        if not items:
            break

        all_items.extend(items)

        total = body.get("total", 0)
        offset += PAGE_SIZE
        if offset >= total:
            break

    return all_items


def normalize(items: list) -> list:
    out = []
    for it in items:
        d = it.get("date")
        v = it.get("value")
        if d and v is not None:
            try:
                out.append({"date": d, "value": round(float(v), 4)})
            except (ValueError, TypeError):
                pass
    return out


def save(index: dict):
    print(f"Fetching {index['name']} (type_id={index['type_id']}) …")
    raw    = fetch_all(index["type_id"])
    items  = normalize(raw)

    payload = {
        "name":         index["name"],
        "type_id":      index["type_id"],
        "last_updated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items":        items,
    }

    os.makedirs(os.path.dirname(index["file"]) or ".", exist_ok=True)
    with open(index["file"], "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"  → {len(items)} records saved to {index['file']}")


if __name__ == "__main__":
    if not LOGIN or not PASSWORD:
        print("ERROR: set CBONDS_LOGIN and CBONDS_PASSWORD environment variables")
        sys.exit(1)

    errors = []
    for idx in INDICES:
        try:
            save(idx)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append(idx["name"])

    if errors:
        print(f"\nFailed: {', '.join(errors)}")
        sys.exit(1)
    print("\nAll done.")
