"""
data_collection.py
==================
Scrapes Baseball-Reference for MLB game-level stats (2023–2026).

Default seasons scraped: 2023, 2024, 2025, 2026
  • 2023 + 2024 + 2025          → full training data
  • 2026 (partial season so far) → additional training data (scraped game-by-game)

Also writes per-team season-average stats for the CURRENT season so that
today_games.py can build pre-game feature vectors without live box score data.

Output
------
  data/raw_games_{year}.csv        one row per team-game
  data/raw_games_combined.csv      all years merged
  data/team_season_avgs_{year}.csv rolling team averages (for pre-game prediction)

Usage
-----
  python data_collection.py                         # 2023–2026
  python data_collection.py --years 2026            # current season only
  python data_collection.py --years 2026 --resume   # resume interrupted run
"""

import os
import re
import time
import argparse
import logging
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL      = "https://www.baseball-reference.com"
HEADERS       = {"User-Agent": "Mozilla/5.0 (compatible; MLBResearchBot/1.0)"}
REQUEST_DELAY = 4.0
MAX_RETRIES   = 3
DEFAULT_YEARS = [2023, 2024, 2025, 2026]

MLB_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SEA",
    "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

BATTING_COLS = {
    "AB":"B1","H":"B2","BB":"B3","SO":"B4","PA":"B5","BA":"B6",
    "OBP":"B7","SLG":"B8","OPS":"B9","Pit":"B10","Str":"B11","PO":"B12","A":"B13",
}
SP_COLS = {
    "IP":"SP1","H":"SP2","BB":"SP3","SO":"SP4","HR":"SP5","ERA":"SP6",
    "BF":"SP7","Pit":"SP8","Str":"SP9","Ctct":"SP10","StS":"SP11","StL":"SP12",
    "GB":"SP13","FB":"SP14","LD":"SP15","GSc":"SP16",
}
ALL_PITCH_COLS = {
    "IP":"P1","H":"P2","BB":"P3","SO":"P4","HR":"P5","ERA":"P6",
    "BF":"P7","Pit":"P8","Str":"P9","Ctct":"P10","StS":"P11","StL":"P12",
    "GB":"P13","FB":"P14","LD":"P15","GSc":"P16","IR":"P17","IS":"P18",
}

os.makedirs("data", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("data/scrape.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def safe_float(v) -> float:
    try:
        return float(str(v).replace("%", "").strip())
    except (ValueError, TypeError):
        return float("nan")


def fetch_page(url: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return BeautifulSoup(r.text, "lxml")
            elif r.status_code == 429:
                time.sleep(60 * attempt)
            else:
                time.sleep(REQUEST_DELAY * attempt)
        except requests.RequestException:
            time.sleep(REQUEST_DELAY * attempt)
    return None


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def parse_game_date(text: str, year: int) -> str:
    parts     = text.split(",")
    month_day = parts[-1].strip() if len(parts) > 1 else text.strip()
    month_day = re.sub(r'\s*\(\d\)', '', month_day).strip()
    try:
        return datetime.strptime(f"{month_day} {year}", "%b %d %Y").strftime("%Y-%m-%d")
    except ValueError:
        return f"{year}-01-01"


def get_team_schedule(team: str, year: int) -> list:
    soup = fetch_page(f"{BASE_URL}/teams/{team}/{year}-schedule-scores.shtml")
    if soup is None:
        return []
    table = soup.find("table", {"id": "team_schedule"})
    if table is None:
        return []

    games = []
    for row in table.find("tbody").find_all("tr"):
        if row.get("class") and "thead" in row.get("class"):
            continue

        def cell(stat):
            t = row.find(attrs={"data-stat": stat})
            return t.get_text(strip=True) if t else ""

        result = cell("win_loss_result")
        if result not in ("W", "L", "W-wo", "L-wo"):
            continue
        bs_tag = row.find("td", {"data-stat": "date_game"})
        if not bs_tag:
            continue
        link = bs_tag.find("a", href=re.compile(r"/boxes/"))
        if not link:
            continue
        game_id   = link["href"].split("/")[-1].replace(".shtml", "")
        is_home   = cell("homeORaway").strip() != "@"
        home_won  = (result.startswith("W") and is_home) or \
                    (result.startswith("L") and not is_home)
        opp       = cell("opp")
        game_date = parse_game_date(bs_tag.get_text(strip=True), year)
        games.append({
            "game_id":   game_id,
            "game_date": game_date,
            "season":    year,
            "home_team": team if is_home else opp,
            "away_team": opp  if is_home else team,
            "home_won":  home_won,
        })
    return games


# ---------------------------------------------------------------------------
# Box score
# ---------------------------------------------------------------------------
def _extract(tr, col_map: dict) -> dict:
    return {col: safe_float((tr.find(attrs={"data-stat": stat}) or {}).get_text(""))
            for stat, col in col_map.items()}


def parse_batting_totals(soup, team_id: str) -> dict:
    t = soup.find("table", {"id": f"batting_{team_id}"})
    if not t:
        return {}
    tf = t.find("tfoot")
    tr = tf.find("tr") if tf else None
    return _extract(tr, BATTING_COLS) if tr else {}


def parse_pitching_table(soup, team_id: str) -> tuple:
    t = soup.find("table", {"id": f"pitching_{team_id}"})
    if not t:
        return {}, {}
    tbody  = t.find("tbody")
    sp_row = tbody.find("tr") if tbody else None
    sp     = _extract(sp_row, SP_COLS) if sp_row else {}
    tf     = t.find("tfoot")
    allp   = {}
    if tf:
        tr = tf.find("tr")
        if tr:
            allp = _extract(tr, ALL_PITCH_COLS)
    return sp, allp


def parse_boxscore(game_id, game_date, season, home_team, away_team, home_won) -> list:
    soup = fetch_page(f"{BASE_URL}/boxes/{game_id[:3]}/{game_id}.shtml")
    if soup is None:
        return []
    rows = []
    for is_home, team, won in [(True, home_team, home_won), (False, away_team, not home_won)]:
        bat, (sp, allp) = parse_batting_totals(soup, team), parse_pitching_table(soup, team)
        rows.append({
            "game_id": game_id, "game_date": game_date, "season": season,
            "team": team, "X1": int(is_home), "Y": int(won),
            **bat, **sp, **allp,
        })
    return rows


# ---------------------------------------------------------------------------
# Per-season scrape
# ---------------------------------------------------------------------------
def scrape_season(year: int, resume: bool = False) -> pd.DataFrame:
    raw_path = f"data/raw_games_{year}.csv"
    ids_path = f"data/game_ids_{year}.csv"

    logger.info("▶  Season %d – schedules…", year)
    if resume and os.path.exists(ids_path):
        games_df = pd.read_csv(ids_path)
    else:
        all_games = []
        for team in tqdm(MLB_TEAMS, desc=f"{year} schedules"):
            all_games.extend(get_team_schedule(team, year))
            time.sleep(REQUEST_DELAY)
        games_df = pd.DataFrame(all_games).drop_duplicates(subset="game_id")
        games_df.to_csv(ids_path, index=False)
    logger.info("   %d unique games found.", len(games_df))

    already: set = set()
    rows: list   = []
    if resume and os.path.exists(raw_path):
        ex       = pd.read_csv(raw_path)
        already  = set(ex["game_id"].unique())
        rows     = ex.to_dict("records")
        logger.info("   Resuming – %d games done.", len(already))

    for _, g in tqdm(games_df.iterrows(), total=len(games_df), desc=f"{year} box scores"):
        gid = g["game_id"]
        if gid in already:
            continue
        rows.extend(parse_boxscore(gid, str(g["game_date"]), int(g["season"]),
                                   str(g["home_team"]), str(g["away_team"]),
                                   bool(g["home_won"])))
        time.sleep(REQUEST_DELAY)
        if len(rows) % 100 == 0:
            pd.DataFrame(rows).to_csv(raw_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)
    logger.info("✅  Season %d – %d rows → %s", year, len(df), raw_path)
    return df


# ---------------------------------------------------------------------------
# Build season-average feature vectors (for pre-game prediction)
# ---------------------------------------------------------------------------
def build_team_season_avgs(df_season: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    For each team, compute the average of all feature columns across
    games played so far this season.  This is used as a proxy feature
    vector when predicting today's games before they start.
    """
    feat_cols = (
        list(BATTING_COLS.values()) +
        list(SP_COLS.values()) +
        list(ALL_PITCH_COLS.values())
    )
    available = [c for c in feat_cols if c in df_season.columns]
    avgs = (
        df_season.groupby("team")[available]
        .mean()
        .reset_index()
        .rename(columns={"team": "team_abbr"})
    )
    avgs["season"]     = year
    avgs["games_played"] = df_season.groupby("team").size().values
    path = f"data/team_season_avgs_{year}.csv"
    avgs.to_csv(path, index=False)
    logger.info("Season averages saved → %s  (%d teams)", path, len(avgs))
    return avgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(years: list = DEFAULT_YEARS, resume: bool = False):
    frames = []
    for year in years:
        df = scrape_season(year, resume=resume)
        frames.append(df)
        # Always regenerate season avgs for the most recent year
        if year == max(years):
            build_team_season_avgs(df, year)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv("data/raw_games_combined.csv", index=False)
    logger.info("✅  Combined: %d rows across %s", len(combined), years)
    for yr, grp in combined.groupby("season"):
        logger.info("   %d: %d games, %d team-game rows", yr,
                    grp["game_id"].nunique(), len(grp))
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",  type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(years=args.years, resume=args.resume)
