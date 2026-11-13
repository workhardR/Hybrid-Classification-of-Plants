"""
Maps a predicted class (plant name) to its group: Medicinal / Endangered / Toxic.
If your dataset's class folders are ALREADY called e.g. 'medicinal', 'endangered', 'poisonous',
nothing needs to be done. Otherwise open category_map.csv (created automatically on the first run)
and fill the 'category' column for each plant. Several groups can be written as  Medicinal; Endangered
"""
import csv
import os

import config


def _guess(cls):
    c = cls.lower()
    if "medic" in c:
        return "Medicinal"
    if "endanger" in c or "threat" in c or "rare" in c:
        return "Endangered"
    if "poison" in c or "toxic" in c:
        return "Toxic (Poisonous)"
    return ""


def ensure_category_map(classes, source_categories=None):
    path = config.CATEGORY_CSV
    existing = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["class"]] = r["category"]
    changed = False
    source_categories = source_categories or {}
    for c in classes:
        supplied = source_categories.get(c, "")
        if supplied and existing.get(c) != supplied:
            existing[c] = supplied
            changed = True
        elif c not in existing:
            existing[c] = _guess(c)
            changed = True
    if changed or not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "category"])
            for c in sorted(existing):
                w.writerow([c, existing[c]])
    return existing


def get_category(cls):
    m = ensure_category_map([cls])
    cat = m.get(cls, "")
    return cat if cat else _guess(cls) or "Unknown - fill category_map.csv"
