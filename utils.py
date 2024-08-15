from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from schema import COLUMNS

_EMPLOYMENT_ORDER = [
    "Employed, full-time",
    "Independent contractor, freelancer, or self-employed",
    "Employed, part-time",
    "Not employed, but looking for work",
    "Not employed, and not looking for work",
    "Student, full-time",
    "Student, part-time",
    "Retired",
]


def _normalize(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s*;\s*", ";", value)
    return value


def canonicalize_employment(selections: Iterable[str]) -> str:
    selections = [s for s in selections if str(s).strip()]
    ordered = [x for x in _EMPLOYMENT_ORDER if x in selections]
    extras = sorted([x for x in selections if x not in _EMPLOYMENT_ORDER])
    return ";".join(ordered + extras)


def preprocess_input(input_dict: dict) -> pd.DataFrame:
    row = {col: 0 for col in COLUMNS}

    # numeric
    try:
        row["YearsCodePro"] = float(input_dict.get("YearsCodePro", 0) or 0)
    except (TypeError, ValueError):
        row["YearsCodePro"] = 0.0

    # Country
    country = input_dict.get("Country")
    if country:
        key = f"Country_{_normalize(country)}"
        if key in row:
            row[key] = 1

    # RemoteWork
    remote = input_dict.get("RemoteWork")
    if remote:
        key = f"RemoteWork_{_normalize(remote)}"
        if key in row:
            row[key] = 1

    # OrgSize (handle straight apostrophe too)
    org = input_dict.get("OrgSize")
    if org:
        org_norm = _normalize(org).replace("I don't know", "I don’t know")
        key = f"OrgSize_{org_norm}"
        if key in row:
            row[key] = 1

    # EdLevel: model schema only has two buckets in this project
    ed = input_dict.get("EdLevel")
    if ed:
        ed_norm = _normalize(ed)
        if ed_norm == "Other" and "EdLevel_Other" in row:
            row["EdLevel_Other"] = 1
        if ed_norm.startswith("Professional degree") and "EdLevel_Professional" in row:
            row["EdLevel_Professional"] = 1

    # Employment: expects a single string with ';' exactly matching schema combos
    employment = input_dict.get("Employment")
    if employment:
        key = f"Employment_{_normalize(employment)}"
        if key in row:
            row[key] = 1

    return pd.DataFrame([row], columns=COLUMNS)
