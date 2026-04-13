"""
today_games.py
==============
Fetches today's MLB schedule from Baseball-Reference and generates
pre-game win-probability predictions using trained models.

Pre-game prediction strategy
-----------------------------
The paper's models use within-game box-score stats as features.
Since those don't exist before first pitch, we substitute each team's
season-to-date AVERAGE for every feature.  This is the standard approach
for pre-game win-probability models and produces sensible predictions
once teams have played ~10+ games.

Output
------
  data/today_predictions.csv  – one row per game with win probabilities
  Printed table shown in the terminal

Usage
-----
  python today_games.py
  python today_games.py --dataset 2 --date 2026-04-13
"""

import os
import re
import time
import json
import argparse
import logging
from datetime import date, datetime

import requests
import numpy as np
import pandas as pd
import joblib
from bs4 import BeautifulSoup

# Try importing TF quietly
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

BASE_URL      = "https://www.baseball-reference.com"
HEADERS       = {"User-Agent": "Mozilla/5.0 (compatible; MLBResearchBot/1.0)"}
REQUEST_DELAY = 3.0
MODELS_DIR    = "models"
DATA_DIR      = "data"

# Baseball-Reference team abbreviation → our internal abbr
# (BR uses slightly different codes in some places)
BR_ABBR_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CHW", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
    # aliases that appear in schedules
    "SD":  "SDP", "SF":  "SFG", "KC":  "KCR", "TB":  "TBR",
    "WSH": "WSN", "CHW": "CHW", "AZ":  "ARI", "ATH": "OAK",
    "CWS": "CHW",
}

DATASET1_COLS = [f"B{i}" for i in range(1,14)] + [f"SP{i}" for i in range(1,17)] + ["X1"]
DATASET2_COLS = [f"B{i}" for i in range(1,14)] + [f"P{i}"  for i in range(1,19)] + ["X1"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team name / abbreviation lookup (for display)
# ---------------------------------------------------------------------------
TEAM_NAMES = {
    "ARI":"Arizona Diamondbacks","ATL":"Atlanta Braves","BAL":"Baltimore Orioles",
    "BOS":"Boston Red Sox","CHC":"Chicago Cubs","CHW":"Chicago White Sox",
    "CIN":"Cincinnati Reds","CLE":"Cleveland Guardians","COL":"Colorado Rockies",
    "DET":"Detroit Tigers","HOU":"Houston Astros","KCR":"Kansas City Royals",
    "LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers","MIA":"Miami Marlins",
    "MIL":"Milwaukee Brewers","MIN":"Minnesota Twins","NYM":"New York Mets",
    "NYY":"New York Yankees","OAK":"Athletics","PHI":"Philadelphia Phillies",
    "PIT":"Pittsburgh Pirates","SDP":"San Diego Padres","SEA":"Seattle Mariners",
    "SFG":"San Francisco Giants","STL":"St. Louis Cardinals","TBR":"Tampa Bay Rays",
    "TEX":"Texas Rangers","TOR":"Toronto Blue Jays","WSN":"Washington Nationals",
}


# ---------------------------------------------------------------------------
# Schedule fetching
# ---------------------------------------------------------------------------
def fetch_todays_schedule(target_date: str) -> list[dict]:
    """
    Scrape Baseball-Reference's scoreboard page for a given date.
    Returns a list of dicts: {game_id, home_team, away_team, start_time, status}
    """
    dt  = datetime.strptime(target_date, "%Y-%m-%d")
    url = (f"{BASE_URL}/boxes/?month={dt.month}&day={dt.day}&year={dt.year}")
    logger.info("Fetching schedule from: %s", url)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        logger.error("Could not fetch schedule: %s", e)
        return []

    games = []
    # Each game is in a div with class "game_summary"
    for div in soup.find_all("div", class_="game_summary"):
        try:
            # Teams
            teams_table = div.find("table", class_="teams")
            if teams_table is None:
                continue
            rows  = teams_table.find_all("tr")
            away_td = rows[0].find("td", class_="right") if rows else None
            home_td = rows[1].find("td", class_="right") if len(rows) > 1 else None

            away_link = rows[0].find("a") if rows else None
            home_link = rows[1].find("a") if len(rows) > 1 else None
            if not away_link or not home_link:
                continue

            away_name = away_link.get_text(strip=True)
            home_name = home_link.get_text(strip=True)

            # Box score link
            nongame = div.find("td", class_="right gamelink")
            if nongame:
                a = nongame.find("a")
                if a:
                    href    = a["href"]
                    game_id = href.split("/")[-1].replace(".shtml", "")
                else:
                    game_id = None
            else:
                game_id = None

            # Determine status: "preview", "live", or "final"
            if away_td and home_td:
                away_score = away_td.get_text(strip=True)
                home_score = home_td.get_text(strip=True)
                try:
                    int(away_score); int(home_score)
                    status = "final"
                except ValueError:
                    status = "scheduled"
            else:
                status = "scheduled"

            games.append({
                "game_id":   game_id,
                "away_team": resolve_team(away_name),
                "home_team": resolve_team(home_name),
                "status":    status,
                "date":      target_date,
            })
        except Exception:
            continue

    logger.info("Found %d games on %s", len(games), target_date)
    return games


def resolve_team(name_or_abbr: str) -> str:
    """Map any team name or abbreviation to our canonical 3-letter abbr."""
    # Direct abbr lookup
    up = name_or_abbr.strip().upper()
    if up in BR_ABBR_MAP:
        return BR_ABBR_MAP[up]
    # Name lookup
    for abbr, full in TEAM_NAMES.items():
        if name_or_abbr.strip().lower() in full.lower():
            return abbr
    return name_or_abbr[:3].upper()


# ---------------------------------------------------------------------------
# Load season averages
# ---------------------------------------------------------------------------
def load_season_avgs(year: int) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"team_season_avgs_{year}.csv")
    if not os.path.exists(path):
        logger.warning("No season averages file found for %d. "
                       "Run data_collection.py --years %d first.", year, year)
        return None
    return pd.read_csv(path)


def get_team_features(team_abbr: str, avgs_df: pd.DataFrame,
                      feat_cols: list, is_home: bool) -> np.ndarray | None:
    """
    Look up a team's season averages and return a feature vector.
    X1 (home/away flag) is appended based on is_home.
    """
    row = avgs_df[avgs_df["team_abbr"] == team_abbr]
    if len(row) == 0:
        # Try alternate abbreviations
        for alt, canon in BR_ABBR_MAP.items():
            if canon == team_abbr:
                row = avgs_df[avgs_df["team_abbr"] == alt]
                if len(row):
                    break
    if len(row) == 0:
        logger.warning("No season avg data for team: %s", team_abbr)
        return None

    data_cols = [c for c in feat_cols if c != "X1"]
    vec = []
    for c in data_cols:
        vec.append(row.iloc[0].get(c, 0.0) if c in row.columns else 0.0)
    vec.append(1.0 if is_home else 0.0)   # X1
    return np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Normalisation using saved scaler params
# ---------------------------------------------------------------------------
def load_scaler(ds_num: int, use_fs: bool) -> tuple[np.ndarray, np.ndarray] | None:
    suffix = "_selected" if use_fs else ""
    path   = os.path.join(DATA_DIR, f"scaler_ds{ds_num}{suffix}.csv")
    if not os.path.exists(path):
        return None
    sc = pd.read_csv(path)
    return sc["min"].values.astype(np.float32), sc["max"].values.astype(np.float32)


def norm(x: np.ndarray, col_min: np.ndarray, col_max: np.ndarray) -> np.ndarray:
    rng = col_max - col_min
    rng[rng == 0] = 1.0
    return ((x - col_min) / rng).clip(0, 1)


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
def load_all_models(ds_num: int, use_fs: bool) -> dict:
    suffix  = "fs" if use_fs else "no_fs"
    loaded  = {}
    for tag, disp in [("svm", "SVM"), ("ann", "ANN"), ("1dcnn", "1DCNN")]:
        if use_fs and tag == "1dcnn":
            continue
        key      = f"ds{ds_num}_{suffix}_{tag}"
        pkl_path = os.path.join(MODELS_DIR, f"{key}.pkl")
        k_path   = os.path.join(MODELS_DIR, f"{key}.keras")
        if os.path.exists(pkl_path):
            loaded[disp] = ("sklearn", joblib.load(pkl_path))
        elif os.path.exists(k_path) and TF_AVAILABLE:
            loaded[disp] = ("keras", tf.keras.models.load_model(k_path))
    if not loaded:
        logger.warning("No models found for ds%d %s. Run train.py first.", ds_num, suffix)
    return loaded


def predict_win_prob(x_norm: np.ndarray, models: dict) -> dict:
    probs = {}
    for name, (kind, model) in models.items():
        try:
            if kind == "sklearn":
                svm, sc = model
                p = svm.predict_proba(sc.transform(x_norm.reshape(1, -1)))[0][1]
            elif name == "1DCNN":
                x3d = x_norm.reshape(1, x_norm.shape[0], 1)
                p   = float(model.predict(x3d, verbose=0)[0][0])
            else:
                p = float(model.predict(x_norm.reshape(1, -1), verbose=0)[0][0])
            probs[name] = round(float(p), 4)
        except Exception as e:
            logger.debug("%s predict error: %s", name, e)
            probs[name] = None
    return probs


# ---------------------------------------------------------------------------
# Main prediction loop
# ---------------------------------------------------------------------------
def predict_today(target_date: str = None, ds_num: int = 2,
                  use_fs: bool = False, season: int = 2026) -> pd.DataFrame:

    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    logger.info("═══ MLB Predictions for %s ════════════════════════", target_date)

    # Load resources
    avgs_df = load_season_avgs(season)
    models  = load_all_models(ds_num, use_fs)
    scaler  = load_scaler(ds_num, use_fs)
    feat_cols = DATASET1_COLS if ds_num == 1 else DATASET2_COLS

    if avgs_df is None or not models or scaler is None:
        logger.error("Missing data / models / scaler. "
                     "Run data_collection.py and train.py first.")
        return pd.DataFrame()

    col_min, col_max = scaler

    # Fetch today's schedule
    games = fetch_todays_schedule(target_date)
    if not games:
        logger.warning("No games found for %s.", target_date)
        return pd.DataFrame()

    model_names = list(models.keys())
    results = []

    for game in games:
        home = game["home_team"]
        away = game["away_team"]

        x_home = get_team_features(home, avgs_df, feat_cols, is_home=True)
        x_away = get_team_features(away, avgs_df, feat_cols, is_home=False)

        row = {
            "date":      target_date,
            "away_team": away,
            "away_name": TEAM_NAMES.get(away, away),
            "home_team": home,
            "home_name": TEAM_NAMES.get(home, home),
            "status":    game["status"],
        }

        if x_home is None or x_away is None:
            for m in model_names:
                row[f"{m}_home_win_prob"] = None
            row["ensemble_home_win_prob"] = None
            row["prediction"] = "N/A (missing data)"
        else:
            x_home_n = norm(x_home, col_min, col_max)
            x_away_n = norm(x_away, col_min, col_max)

            home_probs = predict_win_prob(x_home_n, models)
            away_probs = predict_win_prob(x_away_n, models)

            # Home win prob = average of home team's win prob and (1 - away team's win prob)
            ensemble_probs = []
            for m in model_names:
                hp = home_probs.get(m)
                ap = away_probs.get(m)
                if hp is not None and ap is not None:
                    combined = (hp + (1 - ap)) / 2
                    row[f"{m}_home_win_prob"] = round(combined, 4)
                    ensemble_probs.append(combined)
                else:
                    row[f"{m}_home_win_prob"] = None

            ens = round(float(np.mean(ensemble_probs)), 4) if ensemble_probs else None
            row["ensemble_home_win_prob"] = ens
            if ens is not None:
                fav  = home if ens >= 0.5 else away
                prob = ens if ens >= 0.5 else 1 - ens
                row["prediction"] = f"{TEAM_NAMES.get(fav, fav)} ({prob*100:.1f}%)"
            else:
                row["prediction"] = "N/A"

        results.append(row)

    df = pd.DataFrame(results)

    # Pretty-print
    print(f"\n{'='*72}")
    print(f"  ⚾  MLB Predictions — {target_date}")
    print(f"  Dataset {ds_num} | {'Feature Selected' if use_fs else 'All Features'} "
          f"| {season} season averages as proxies")
    print(f"{'='*72}")
    print(f"  {'Matchup':<38}  {'Prediction':<35}  Status")
    print(f"  {'-'*38}  {'-'*35}  {'-'*10}")
    for _, r in df.iterrows():
        matchup = f"{r['away_name']} @ {r['home_name']}"
        print(f"  {matchup:<38}  {r['prediction']:<35}  {r['status']}")
    print(f"{'='*72}\n")

    out_path = os.path.join(DATA_DIR, "today_predictions.csv")
    df.to_csv(out_path, index=False)
    logger.info("Predictions saved → %s", out_path)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date",    type=str, default=None,
                   help="Date to predict (YYYY-MM-DD, default: today)")
    p.add_argument("--dataset", type=int, default=2, choices=[1, 2])
    p.add_argument("--fs",      action="store_true",
                   help="Use feature-selected models")
    p.add_argument("--season",  type=int, default=2026,
                   help="Season year for team averages (default: 2026)")
    a = p.parse_args()
    predict_today(target_date=a.date, ds_num=a.dataset,
                  use_fs=a.fs, season=a.season)
