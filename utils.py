from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from schema import COLUMNS

MODEL_PATH = Path(os.getenv("MODEL_PATH", "Notebooks/Models/reg_model.pkl"))

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

_MODEL = None


def _normalize(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s*;\s*", ";", value)
    return value


def canonicalize_employment(selections: Iterable[str]) -> str:
    selections = [s for s in selections if str(s).strip()]
    ordered = [x for x in _EMPLOYMENT_ORDER if x in selections]
    extras = sorted([x for x in selections if x not in _EMPLOYMENT_ORDER])
    return ";".join(ordered + extras)


def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            f"Place reg_model.pkl there or set MODEL_PATH env var."
        )

    _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


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

    # EdLevel: schema only has two buckets here
    ed = input_dict.get("EdLevel")
    if ed:
        ed_norm = _normalize(ed)
        if ed_norm == "Other" and "EdLevel_Other" in row:
            row["EdLevel_Other"] = 1
        if ed_norm.startswith("Professional degree") and "EdLevel_Professional" in row:
            row["EdLevel_Professional"] = 1

    # Employment (single string, already canonicalized in UI)
    employment = input_dict.get("Employment")
    if employment:
        key = f"Employment_{_normalize(employment)}"
        if key in row:
            row[key] = 1

    return pd.DataFrame([row], columns=COLUMNS)


def predict_salary(input_dict: dict) -> float:
    model = get_model()
    X = preprocess_input(input_dict)
    pred = model.predict(X)[0]
    return round(float(pred), 2)
