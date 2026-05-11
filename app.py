# -*- coding: utf-8 -*-
# ============================================================
# DEVIL PICKS — REAL RAILWAY PRO BUILD
# NBA Props Engine for Underdog + Sleeper + Odds API fallback
# Streamlit / GitHub / Railway ready
#
# This is not a demo app:
# - Pulls real prop rows from enabled sources when keys/feeds are available
# - No fake picks are generated
# - Missing/blocked feeds show source errors instead of dummy data
# - Saves official snapshots
# - Supports manual grading + learning loop
# - Bayesian, Markov, Monte Carlo, optional XGBoost/GBM hooks
# ============================================================

import os
import re
import json
import math
import time
import difflib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:
    GradientBoostingClassifier = None

try:
    import xgboost as xgb
except Exception:
    xgb = None


# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="Devil Picks — Real Rail Pro",
    page_icon="😈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

PICK_LOG = os.path.join(DATA_DIR, "official_pick_log.json")
RESULT_LOG = os.path.join(DATA_DIR, "graded_result_log.json")
LEARN_FILE = os.path.join(DATA_DIR, "learning_state.json")
CLV_FILE = os.path.join(DATA_DIR, "clv_tracker.json")
REQUEST_LOG_FILE = os.path.join(DATA_DIR, "source_request_log.json")
SNAPSHOT_FILE = os.path.join(DATA_DIR, "latest_board_snapshot.json")
MODEL_STATE_FILE = os.path.join(DATA_DIR, "model_state.json")

# Environment variables
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "").strip()
OPTICODDS_API_KEY = os.getenv("OPTICODDS_API_KEY", "").strip()

ENABLE_UNDERDOG_DEFAULT = os.getenv("ENABLE_UNDERDOG", "true").lower() == "true"
ENABLE_SLEEPER_DEFAULT = os.getenv("ENABLE_SLEEPER", "true").lower() == "true"
ENABLE_ODDS_API_DEFAULT = os.getenv("ENABLE_ODDS_API", "false").lower() == "true"
ENABLE_OPTICODDS_DEFAULT = os.getenv("ENABLE_OPTICODDS", "false").lower() == "true"

SPORT_KEY = os.getenv("SPORT_KEY", "basketball_nba")
LEAGUE_FILTER = os.getenv("LEAGUE_FILTER", "NBA").upper()

UNDERDOG_API_URL = os.getenv(
    "UNDERDOG_API_URL",
    "https://api.underdogfantasy.com/v1/over_under_lines",
).strip()

# Apify actors can be changed in Railway env vars if you use a different scraper actor.
APIFY_UNDERDOG_ACTOR = os.getenv("APIFY_UNDERDOG_ACTOR", "zen-studio/underdog-player-props").strip()
APIFY_SLEEPER_ACTOR = os.getenv("APIFY_SLEEPER_ACTOR", "zen-studio/sleeper-player-props").strip()

ODDS_BASE = "https://api.the-odds-api.com/v4"

DEFAULT_PRICE = float(os.getenv("DEFAULT_PRICE", "-110"))
DEFAULT_SIM_COUNT = int(os.getenv("PROP_SIMULATION_COUNT", "12000"))
DEFAULT_MIN_PROB = float(os.getenv("MIN_PROB", "0.57"))
DEFAULT_MIN_DATA_SCORE = int(os.getenv("MIN_DATA_SCORE", "68"))
DEFAULT_MAX_KELLY = float(os.getenv("MAX_KELLY", "0.03"))
DEFAULT_BANKROLL = float(os.getenv("BANKROLL", "1000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "180"))

PROP_CONFIG = {
    "PTS": {"label": "Points", "min_edge": 1.8, "std": 5.8, "odds_market": "player_points"},
    "REB": {"label": "Rebounds", "min_edge": 1.5, "std": 3.0, "odds_market": "player_rebounds"},
    "AST": {"label": "Assists", "min_edge": 1.5, "std": 2.8, "odds_market": "player_assists"},
    "PRA": {"label": "Pts + Reb + Ast", "min_edge": 2.5, "std": 7.0, "odds_market": "player_points_rebounds_assists"},
    "PR": {"label": "Pts + Reb", "min_edge": 2.2, "std": 6.2, "odds_market": "player_points_rebounds"},
    "PA": {"label": "Pts + Ast", "min_edge": 2.2, "std": 6.2, "odds_market": "player_points_assists"},
    "RA": {"label": "Reb + Ast", "min_edge": 2.0, "std": 4.2, "odds_market": "player_rebounds_assists"},
    "3PM": {"label": "3PM", "min_edge": 0.65, "std": 1.35, "odds_market": "player_threes"},
}

PROP_ALIASES = {
    "points": "PTS", "point": "PTS", "pts": "PTS", "player points": "PTS",
    "rebounds": "REB", "rebound": "REB", "rebs": "REB", "reb": "REB", "player rebounds": "REB",
    "assists": "AST", "assist": "AST", "asts": "AST", "ast": "AST", "player assists": "AST",
    "points rebounds assists": "PRA", "pts rebs asts": "PRA", "pra": "PRA",
    "points + rebounds + assists": "PRA", "pts + rebs + asts": "PRA",
    "points rebounds": "PR", "points + rebounds": "PR", "pts rebs": "PR", "pr": "PR",
    "points assists": "PA", "points + assists": "PA", "pts asts": "PA", "pa": "PA",
    "rebounds assists": "RA", "rebounds + assists": "RA", "rebs asts": "RA", "ra": "RA",
    "3 pointers made": "3PM", "3-pointers made": "3PM", "three pointers made": "3PM",
    "threes": "3PM", "3pm": "3PM", "fg3m": "3PM", "made threes": "3PM",
}


# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
:root {
  --bg0:#02040a; --bg1:#070b13; --panel:#0b1220; --panel2:#111827;
  --red:#ff344f; --green:#38f063; --orange:#ffb02e; --blue:#67a4ff; --muted:#aeb7c9;
}
.stApp {
  background: radial-gradient(circle at top left,#2b000d 0%,#070b13 42%,#02040a 100%);
  color:#f7f8fb;
}
.block-container {max-width:1720px; padding-top:1rem; padding-bottom:3rem;}
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#050912,#02040a);
  border-right:1px solid rgba(255,52,79,.22);
}
h1,h2,h3 {color:#fff;}
.hero {
  border:1px solid rgba(255,255,255,.15);
  background:linear-gradient(135deg,rgba(12,19,34,.98),rgba(5,8,18,.96));
  border-radius:26px;
  padding:22px;
  margin-bottom:16px;
  box-shadow:0 0 35px rgba(255,52,79,.12);
}
.logo-title {font-size:32px; font-weight:950; letter-spacing:-.5px;}
.sub {color:#aeb7c9; font-size:13px;}
.card {
  border:1px solid rgba(255,255,255,.14);
  background:linear-gradient(145deg,#0a111f,#080d18);
  border-radius:20px;
  padding:18px;
  box-shadow:0 0 22px rgba(0,0,0,.24);
  margin-bottom:14px;
}
.card-good {border-color:rgba(56,240,99,.45); box-shadow:0 0 22px rgba(56,240,99,.10);}
.card-warn {border-color:rgba(255,176,46,.45); box-shadow:0 0 22px rgba(255,176,46,.08);}
.card-bad {border-color:rgba(255,52,79,.45); box-shadow:0 0 22px rgba(255,52,79,.08);}
.metric-grid {display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:12px 0;}
.metric-grid-4 {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:12px 0;}
.metric-box {
  border:1px solid rgba(255,255,255,.14);
  background:linear-gradient(145deg,#0a111f,#080d18);
  border-radius:16px;
  padding:13px;
  min-height:78px;
}
.metric-label {font-size:11px; color:#aeb7c9; text-transform:uppercase; font-weight:900; letter-spacing:.05em;}
.metric-value {font-size:25px; color:#fff; font-weight:950; margin-top:5px;}
.metric-sub {font-size:12px; color:#aeb7c9; margin-top:4px;}
.badge {
  display:inline-block;
  padding:7px 11px;
  border-radius:999px;
  font-weight:900;
  font-size:12px;
  margin:3px 5px 3px 0;
  border:1px solid rgba(255,255,255,.18);
  background:#101827;
  color:#dce4f5;
}
.badge-green {background:#002c16; border-color:rgba(56,240,99,.55); color:#b9ffd0;}
.badge-red {background:#3a0710; border-color:rgba(255,52,79,.55); color:#ffc2cb;}
.badge-orange {background:#362000; border-color:rgba(255,176,46,.55); color:#ffe1a3;}
.badge-blue {background:#071f3d; border-color:rgba(103,164,255,.55); color:#cbe0ff;}
.green {color:#38f063;} .red {color:#ff344f;} .orange {color:#ffb02e;} .blue {color:#67a4ff;} .muted {color:#aeb7c9;}
.big {font-size:24px; font-weight:950;}
.section-title {font-size:22px; font-weight:950; margin:18px 0 10px; border-left:5px solid #ff344f; padding-left:12px;}
.stButton button {border-radius:14px; font-weight:900; border:1px solid rgba(255,255,255,.18);}
.stTabs [data-baseweb="tab"] {color:#b8c3cf; font-weight:900;}
.stTabs [aria-selected="true"] {color:#ff344f!important; border-bottom:3px solid #ff344f;}
[data-testid="stMetric"] {background:#0a111f; border:1px solid rgba(255,255,255,.14); border-radius:16px; padding:14px;}
@media (max-width: 1100px) {
  .metric-grid,.metric-grid-4 {grid-template-columns:repeat(2,minmax(0,1fr));}
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SAFE STORAGE / LOGGING
# ============================================================
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def load_json(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path: str, data: Any) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass

def log_request(source: str, status: str, message: str = "") -> None:
    rows = load_json(REQUEST_LOG_FILE, [])
    rows.append({
        "time": now_iso(),
        "source": str(source)[:200],
        "status": str(status)[:90],
        "message": str(message)[:500],
    })
    save_json(REQUEST_LOG_FILE, rows[-1000:])


# ============================================================
# BASIC HELPERS
# ============================================================
def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normalize_name(name: Any) -> str:
    s = str(name or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())

def name_score(a: Any, b: Any) -> float:
    a, b = normalize_name(a), normalize_name(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.94
    return difflib.SequenceMatcher(None, a, b).ratio()

def clean_prop_type(raw: Any) -> str:
    s = normalize_name(raw)
    if s in PROP_ALIASES:
        return PROP_ALIASES[s]
    for key, val in PROP_ALIASES.items():
        if key in s:
            return val
    up = str(raw or "").upper().strip()
    return up if up in PROP_CONFIG else ""

def american_to_decimal(odds: Any) -> Optional[float]:
    odds = safe_float(odds)
    if odds is None:
        return None
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)

def american_to_implied(odds: Any) -> Optional[float]:
    odds = safe_float(odds)
    if odds is None:
        return None
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def expected_value(prob: Optional[float], odds: Any = DEFAULT_PRICE) -> Optional[float]:
    dec = american_to_decimal(odds)
    if prob is None or dec is None:
        return None
    return float(prob * (dec - 1) - (1 - prob))

def kelly_fraction(prob: Optional[float], odds: Any, max_kelly: float) -> float:
    dec = american_to_decimal(odds)
    if prob is None or dec is None:
        return 0.0
    b = dec - 1
    q = 1 - prob
    if b <= 0:
        return 0.0
    return float(clamp(((b * prob) - q) / b, 0, max_kelly))

def odds_display(o: Any) -> str:
    o = safe_float(o)
    if o is None:
        return "N/A"
    return f"+{int(o)}" if o > 0 else str(int(o))

def pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.1f}%"

def source_key(source: str) -> str:
    return str(source).split("/")[0].strip()


# ============================================================
# HTTP
# ============================================================
def safe_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 25) -> Any:
    try:
        h = {
            "User-Agent": "Mozilla/5.0 DevilPicksRealRail/4.0",
            "Accept": "application/json,text/plain,*/*",
            "Connection": "keep-alive",
        }
        if headers:
            h.update(headers)
        r = requests.get(url, params=params, headers=h, timeout=timeout)
        if r.status_code != 200:
            log_request(url, f"HTTP {r.status_code}", r.text[:400])
            return None
        return r.json()
    except Exception as e:
        log_request(url, "REQUEST_ERROR", str(e))
        return None

def safe_post_json(url: str, payload: Optional[dict] = None, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 60) -> Any:
    try:
        h = {
            "User-Agent": "Mozilla/5.0 DevilPicksRealRail/4.0",
            "Accept": "application/json,text/plain,*/*",
            "Content-Type": "application/json",
        }
        if headers:
            h.update(headers)
        r = requests.post(url, json=payload or {}, params=params, headers=h, timeout=timeout)
        if r.status_code not in [200, 201]:
            log_request(url, f"HTTP {r.status_code}", r.text[:500])
            return None
        return r.json()
    except Exception as e:
        log_request(url, "POST_ERROR", str(e))
        return None


# ============================================================
# SOURCE NORMALIZATION
# ============================================================
def normalize_prop_row(
    source: str,
    player: Any,
    prop: Any,
    line: Any,
    side: Any = "BOTH",
    price: Any = DEFAULT_PRICE,
    team: Any = "",
    opponent: Any = "",
    game: Any = "",
    start_time: Any = "",
    raw: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    prop_key = clean_prop_type(prop)
    if prop_key not in PROP_CONFIG:
        return None
    player = str(player or "").strip()
    if not player:
        return None
    line = safe_float(line)
    if line is None:
        return None

    return {
        "source": str(source),
        "player": player,
        "team": str(team or "").strip(),
        "opponent": str(opponent or "").strip(),
        "game": str(game or "").strip(),
        "start_time": str(start_time or "").strip(),
        "prop": prop_key,
        "prop_label": PROP_CONFIG[prop_key]["label"],
        "line": float(line),
        "book_side": str(side or "BOTH").upper(),
        "price": float(safe_float(price, DEFAULT_PRICE)),
        "raw": raw or {},
    }


# ============================================================
# REAL SOURCE LOADERS
# ============================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_underdog_direct(enabled: bool, url: str) -> List[Dict[str, Any]]:
    if not enabled:
        return []
    data = safe_get_json(url, timeout=25)
    if not data:
        return []

    rows: List[Dict[str, Any]] = []

    included = data.get("included", []) if isinstance(data, dict) else []
    data_items = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    over_under_lines = data.get("over_under_lines", []) if isinstance(data, dict) else []

    appearances: Dict[str, dict] = {}
    players: Dict[str, dict] = {}

    for item in included:
        if not isinstance(item, dict):
            continue
        typ = str(item.get("type", "")).lower()
        item_id = str(item.get("id", ""))
        attrs = item.get("attributes", {}) or {}
        if "appearance" in typ:
            appearances[item_id] = attrs
        if typ in ["player", "players"] or "player" in typ:
            players[item_id] = attrs

    candidates = []
    if isinstance(data_items, list):
        candidates.extend(data_items)
    if isinstance(over_under_lines, list):
        candidates.extend(over_under_lines)

    for item in candidates:
        if not isinstance(item, dict):
            continue

        attrs = item.get("attributes", item) or {}
        rel = item.get("relationships", {}) or {}

        # Underdog shape variants
        over_under = attrs.get("over_under") if isinstance(attrs.get("over_under"), dict) else {}
        option = attrs.get("over_under_option") if isinstance(attrs.get("over_under_option"), dict) else {}

        stat_type = (
            attrs.get("stat_type")
            or attrs.get("stat")
            or attrs.get("display_stat")
            or attrs.get("title")
            or over_under.get("title")
            or option.get("stat_type")
        )

        line = (
            attrs.get("line")
            or attrs.get("stat_value")
            or attrs.get("value")
            or over_under.get("line")
            or option.get("line")
        )

        appearance_id = None
        player_id = None
        try:
            appearance_id = rel.get("appearance", {}).get("data", {}).get("id")
        except Exception:
            appearance_id = None
        try:
            player_id = rel.get("player", {}).get("data", {}).get("id")
        except Exception:
            player_id = None

        app = appearances.get(str(appearance_id), {}) if appearance_id else {}
        pl = players.get(str(player_id), {}) if player_id else {}

        player_name = (
            attrs.get("player_name")
            or attrs.get("player")
            or attrs.get("athlete_name")
            or app.get("player_name")
            or app.get("name")
            or app.get("display_name")
            or pl.get("first_name", "") + " " + pl.get("last_name", "")
        ).strip()

        team = attrs.get("team") or attrs.get("team_abbr") or app.get("team_abbr") or app.get("team") or ""
        opp = attrs.get("opponent") or app.get("opponent_abbr") or app.get("opponent") or ""
        game = attrs.get("match_title") or attrs.get("game") or app.get("match_title") or ""
        start = attrs.get("scheduled_at") or attrs.get("start_time") or app.get("scheduled_at") or ""

        row = normalize_prop_row(
            "Underdog",
            player_name,
            stat_type,
            line,
            side="BOTH",
            price=DEFAULT_PRICE,
            team=team,
            opponent=opp,
            game=game,
            start_time=start,
            raw=attrs,
        )
        if row:
            rows.append(row)

    log_request("Underdog Direct", "OK", f"{len(rows)} rows")
    return rows


def apify_actor_url(actor_id: str) -> str:
    actor = actor_id.replace("/", "~")
    return f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_apify_actor(enabled: bool, actor_id: str, token: str, source_name: str, league: str) -> List[Dict[str, Any]]:
    if not enabled or not token or not actor_id:
        return []

    url = apify_actor_url(actor_id)
    params = {"token": token, "clean": "true"}
    payload = {"league": league, "sport": "NBA", "sports": ["NBA"], "limit": 500}

    data = safe_post_json(url, payload=payload, params=params, timeout=90)
    if data is None:
        # Some actors expose defaults via GET.
        data = safe_get_json(url, params=params, timeout=90)

    if not isinstance(data, list):
        log_request(source_name, "NO_DATA", "Apify actor returned no list dataset")
        return []

    rows: List[Dict[str, Any]] = []

    for it in data:
        if not isinstance(it, dict):
            continue
        player = (
            it.get("playerName")
            or it.get("player_name")
            or it.get("player")
            or it.get("athlete")
            or it.get("athleteName")
            or it.get("name")
        )
        prop = it.get("statType") or it.get("stat_type") or it.get("market") or it.get("prop") or it.get("title")
        line = it.get("line") or it.get("value") or it.get("statValue") or it.get("stat_value")
        team = it.get("team") or it.get("teamAbbreviation") or it.get("team_abbr") or ""
        opponent = it.get("opponent") or it.get("opponentAbbreviation") or it.get("opponent_abbr") or ""
        game = it.get("game") or it.get("matchup") or it.get("matchTitle") or it.get("event") or ""
        start_time = it.get("startTime") or it.get("start_time") or it.get("scheduledAt") or ""
        side = it.get("side") or "BOTH"
        price = it.get("americanOdds") or it.get("american_odds") or it.get("odds") or DEFAULT_PRICE

        row = normalize_prop_row(
            source_name,
            player,
            prop,
            line,
            side=side,
            price=price,
            team=team,
            opponent=opponent,
            game=game,
            start_time=start_time,
            raw=it,
        )
        if row:
            rows.append(row)

    log_request(source_name, "OK", f"{len(rows)} rows")
    return rows


@st.cache_data(ttl=360, show_spinner=False)
def fetch_odds_api_props(enabled: bool, api_key: str, sport_key: str, selected_props: Tuple[str, ...]) -> List[Dict[str, Any]]:
    if not enabled or not api_key:
        return []

    event_url = f"{ODDS_BASE}/sports/{sport_key}/events"
    events = safe_get_json(event_url, params={"apiKey": api_key}, timeout=25)
    if not isinstance(events, list):
        return []

    markets = sorted(set(PROP_CONFIG[p]["odds_market"] for p in selected_props if p in PROP_CONFIG))
    if not markets:
        return []

    market_to_prop = {v["odds_market"]: k for k, v in PROP_CONFIG.items()}
    rows: List[Dict[str, Any]] = []

    for ev in events[:25]:
        event_id = ev.get("id")
        if not event_id:
            continue

        url = f"{ODDS_BASE}/sports/{sport_key}/events/{event_id}/odds"
        data = safe_get_json(
            url,
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": ",".join(markets),
                "oddsFormat": "american",
            },
            timeout=25,
        )
        if not isinstance(data, dict):
            continue

        game = f"{ev.get('away_team','')} @ {ev.get('home_team','')}".strip(" @")
        start_time = ev.get("commence_time") or ""

        for book in data.get("bookmakers", []) or []:
            book_name = book.get("title") or book.get("key") or "Book"
            for market in book.get("markets", []) or []:
                key = market.get("key")
                prop_key = market_to_prop.get(key)
                if not prop_key:
                    continue
                for out in market.get("outcomes", []) or []:
                    player = out.get("description") or out.get("player") or out.get("participant") or ""
                    side = out.get("name") or "BOTH"
                    row = normalize_prop_row(
                        f"OddsAPI/{book_name}",
                        player,
                        prop_key,
                        out.get("point"),
                        side=side,
                        price=out.get("price", DEFAULT_PRICE),
                        game=game,
                        start_time=start_time,
                        raw=out,
                    )
                    if row:
                        rows.append(row)

    log_request("OddsAPI", "OK", f"{len(rows)} rows")
    return rows


@st.cache_data(ttl=240, show_spinner=False)
def get_all_source_props(
    use_underdog: bool,
    use_sleeper: bool,
    use_odds_api: bool,
    use_opticodds: bool,
    underdog_url: str,
    apify_token: str,
    odds_api_key: str,
    selected_props: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    rows.extend(fetch_underdog_direct(use_underdog, underdog_url))
    rows.extend(fetch_apify_actor(use_underdog, APIFY_UNDERDOG_ACTOR, apify_token, "Underdog Apify", LEAGUE_FILTER))
    rows.extend(fetch_apify_actor(use_sleeper, APIFY_SLEEPER_ACTOR, apify_token, "Sleeper", LEAGUE_FILTER))
    rows.extend(fetch_odds_api_props(use_odds_api, odds_api_key, SPORT_KEY, selected_props))

    # OpticOdds placeholder is intentionally not fake. It only logs that no configured parser is enabled.
    if use_opticodds and OPTICODDS_API_KEY:
        log_request("OpticOdds", "NOT_CONFIGURED", "OpticOdds parser not included in this build.")

    # Deduplicate; keep first source row.
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (
            source_key(r["source"]),
            normalize_name(r["player"]),
            r["prop"],
            round(float(r["line"]), 2),
            r.get("book_side", "BOTH"),
            r.get("game", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    save_json(SNAPSHOT_FILE, {"created_at": now_iso(), "rows": out})
    return out


# ============================================================
# LEARNING / CLV
# ============================================================
def load_learning() -> Dict[str, Any]:
    return load_json(LEARN_FILE, {
        "samples": 0,
        "player_bias": {},
        "prop_bias": {},
        "source_bias": {},
        "model_calibration": {},
    })

def save_learning(data: Dict[str, Any]) -> None:
    save_json(LEARN_FILE, data)

def update_clv(row: Dict[str, Any], pick_side: str) -> float:
    data = load_json(CLV_FILE, {})
    key = f"{source_key(row['source'])}_{normalize_name(row['player'])}_{row['prop']}_{pick_side}_{row['game']}"
    latest = float(row["line"])
    if key not in data:
        data[key] = {
            "open": latest,
            "latest": latest,
            "source": row["source"],
            "player": row["player"],
            "prop": row["prop"],
            "side": pick_side,
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        save_json(CLV_FILE, data)
        return 0.0
    open_val = safe_float(data[key].get("open"), latest) or latest
    data[key]["latest"] = latest
    data[key]["updated_at"] = now_iso()
    save_json(CLV_FILE, data)
    return round(latest - open_val, 3)

def rebuild_learning_from_results() -> int:
    results = load_json(RESULT_LOG, [])
    graded = [r for r in results if safe_float(r.get("actual")) is not None and safe_float(r.get("line")) is not None]
    learning = load_learning()

    player_bias = learning.get("player_bias") or {}
    prop_bias = learning.get("prop_bias") or {}
    source_bias = learning.get("source_bias") or {}

    # Reset from scratch each rebuild for stability.
    player_bias = {}
    prop_bias = {}
    source_bias = {}

    for r in graded[-600:]:
        actual = safe_float(r.get("actual"))
        line = safe_float(r.get("line"))
        if actual is None or line is None:
            continue
        err = clamp(actual - line, -10, 10)
        pkey = normalize_name(r.get("player"))
        prop = r.get("prop")
        src = source_key(r.get("source", "Manual"))

        player_bias[pkey] = clamp((safe_float(player_bias.get(pkey), 0.0) or 0.0) + err * 0.004, -2.0, 2.0)
        prop_bias[prop] = clamp((safe_float(prop_bias.get(prop), 0.0) or 0.0) + err * 0.002, -1.25, 1.25)
        source_bias[src] = clamp((safe_float(source_bias.get(src), 0.0) or 0.0) + err * 0.0012, -0.75, 0.75)

    learning["samples"] = len(graded)
    learning["player_bias"] = player_bias
    learning["prop_bias"] = prop_bias
    learning["source_bias"] = source_bias
    learning["updated_at"] = now_iso()
    save_learning(learning)
    return len(graded)


# ============================================================
# MODEL LAYERS
# ============================================================
def bayesian_shrink(prob: float, data_score: int, learning_samples: int, enabled: bool) -> Tuple[float, str]:
    if not enabled:
        return prob, "Bayesian off"
    # Shrinks low-confidence picks toward 50/50.
    data_weight = clamp(data_score / 100.0, 0.30, 0.96)
    sample_weight = clamp(learning_samples / 100.0, 0.10, 1.00)
    confidence = clamp((0.65 * data_weight) + (0.35 * sample_weight), 0.22, 0.94)
    adjusted = (prob * confidence) + (0.50 * (1 - confidence))
    return float(clamp(adjusted, 0.01, 0.99)), f"Bayesian confidence {confidence:.2f}"

def markov_recent_state(player: str, prop: str, line: float, enabled: bool) -> Tuple[str, float]:
    if not enabled:
        return "off", 0.0
    results = load_json(RESULT_LOG, [])
    matches = [
        r for r in results
        if normalize_name(r.get("player")) == normalize_name(player)
        and r.get("prop") == prop
        and safe_float(r.get("actual")) is not None
    ][-10:]

    if len(matches) < 4:
        return "neutral", 0.0

    over_hits = []
    for r in matches:
        actual = safe_float(r.get("actual"), 0.0) or 0.0
        ref_line = safe_float(r.get("line"), line) or line
        over_hits.append(1 if actual > ref_line else 0)

    recent = float(np.mean(over_hits[-4:]))
    full = float(np.mean(over_hits))
    if recent >= full + 0.18:
        return "hot-over", 0.025
    if recent <= full - 0.18:
        return "cold-over", -0.025
    return "stable", 0.0

def projection_from_market_line(row: Dict[str, Any], learning: Dict[str, Any], use_learning: bool, use_markov: bool) -> Tuple[float, float, str, float, str]:
    line = float(row["line"])
    pkey = normalize_name(row["player"])
    prop = row["prop"]
    src = source_key(row["source"])

    player_bias = safe_float((learning.get("player_bias") or {}).get(pkey), 0.0) or 0.0
    prop_bias = safe_float((learning.get("prop_bias") or {}).get(prop), 0.0) or 0.0
    source_bias = safe_float((learning.get("source_bias") or {}).get(src), 0.0) or 0.0

    bias = player_bias + prop_bias + source_bias if use_learning else 0.0
    projection = line + bias
    std = float(PROP_CONFIG[prop].get("std", 3.0))

    markov_state, markov_adj = markov_recent_state(row["player"], prop, line, use_markov)
    note = f"Bias {bias:+.2f} | player {player_bias:+.2f}, prop {prop_bias:+.2f}, source {source_bias:+.2f}"
    return projection, std, markov_state, markov_adj, note

def xgb_gate_score(features: Dict[str, Any], enabled: bool, learning_samples: int) -> Tuple[Optional[float], str]:
    if not enabled:
        return None, "XGBoost off"
    if learning_samples < 75:
        return None, "XGBoost waiting for 75+ graded samples"
    if xgb is None and GradientBoostingClassifier is None:
        return None, "XGBoost/GBM package unavailable"

    # This build keeps a safe hook. It does not fabricate a trained model.
    # After enough graded samples, you can train and persist a real model here.
    state = load_json(MODEL_STATE_FILE, {})
    if not state.get("trained"):
        return None, "XGBoost hook ready, no trained model yet"
    return None, "XGBoost model state not loaded in lightweight build"

def model_one_prop(
    row: Dict[str, Any],
    sim_count: int,
    min_prob: float,
    min_data_score: int,
    min_edge_scale: float,
    max_kelly: float,
    bankroll: float,
    use_bayesian: bool,
    use_markov: bool,
    use_learning: bool,
    use_xgboost: bool,
) -> Dict[str, Any]:
    learning = load_learning()
    learning_samples = int(learning.get("samples", 0) or 0)

    projection, std, markov_state, markov_prob_adj, projection_note = projection_from_market_line(
        row, learning, use_learning, use_markov
    )

    sims = np.random.normal(loc=projection, scale=max(std, 0.40), size=max(1000, sim_count))
    over_raw = float(np.mean(sims > float(row["line"])))
    under_raw = 1.0 - over_raw

    over_raw = clamp(over_raw + markov_prob_adj, 0.01, 0.99)
    under_raw = 1.0 - over_raw

    side = "OVER" if over_raw >= under_raw else "UNDER"
    raw_pick_prob = max(over_raw, under_raw)

    data_score = 50
    if source_key(row["source"]) in ["Underdog", "Sleeper"]:
        data_score += 12
    elif source_key(row["source"]) == "OddsAPI":
        data_score += 8
    if row.get("player"):
        data_score += 7
    if row.get("game") or row.get("team"):
        data_score += 5
    if row.get("price") is not None:
        data_score += 4
    if use_bayesian:
        data_score += 5
    if use_markov:
        data_score += 4
    if use_learning and learning_samples >= 20:
        data_score += 5
    data_score = int(clamp(data_score, 0, 100))

    adj_prob, bayes_note = bayesian_shrink(raw_pick_prob, data_score, learning_samples, use_bayesian)
    if side == "OVER":
        over_prob = adj_prob
        under_prob = 1.0 - adj_prob
    else:
        under_prob = adj_prob
        over_prob = 1.0 - adj_prob

    pick_prob = max(over_prob, under_prob)
    edge = abs(float(projection) - float(row["line"]))
    min_edge = float(PROP_CONFIG[row["prop"]]["min_edge"]) * min_edge_scale

    ev = expected_value(pick_prob, row.get("price", DEFAULT_PRICE))
    kelly = kelly_fraction(pick_prob, row.get("price", DEFAULT_PRICE), max_kelly)
    stake = bankroll * kelly
    clv = update_clv(row, side)

    xgb_prob, xgb_note = xgb_gate_score(
        {
            "pick_prob": pick_prob,
            "edge": edge,
            "data_score": data_score,
            "prop": row["prop"],
            "source": source_key(row["source"]),
        },
        use_xgboost,
        learning_samples,
    )

    reasons = []
    if pick_prob < min_prob:
        reasons.append("probability below gate")
    if edge < min_edge:
        reasons.append("edge below gate")
    if data_score < min_data_score:
        reasons.append("data score below gate")
    if ev is None or ev < 0:
        reasons.append("EV not positive")
    if use_xgboost and xgb_prob is not None and xgb_prob < 0.53:
        reasons.append("XGBoost filter failed")

    qualified = len(reasons) == 0
    if qualified and pick_prob >= 0.64 and edge >= min_edge * 1.25:
        signal = f"😈 STRONG {side}"
    elif qualified:
        signal = f"✅ LEAN {side}"
    else:
        signal = "PASS"

    return {
        **row,
        "projection": float(projection),
        "std": float(std),
        "over_prob": float(over_prob),
        "under_prob": float(under_prob),
        "pick_side": side,
        "pick_prob": float(pick_prob),
        "raw_pick_prob": float(raw_pick_prob),
        "edge": float(edge),
        "min_edge": float(min_edge),
        "ev": None if ev is None else float(ev),
        "kelly": float(kelly),
        "stake": float(stake),
        "clv": float(clv),
        "data_score": int(data_score),
        "qualified": bool(qualified),
        "signal": signal,
        "reasons": reasons,
        "markov_state": markov_state,
        "model_notes": [projection_note, bayes_note, xgb_note],
        "learning_samples": learning_samples,
    }

def build_board(
    rows: List[Dict[str, Any]],
    selected_sources: List[str],
    selected_props: List[str],
    player_search: str,
    min_display_prob: float,
    show_passes: bool,
    **model_kwargs
) -> List[Dict[str, Any]]:
    filtered = []
    source_set = set(selected_sources or [])
    prop_set = set(selected_props or [])

    for r in rows:
        skey = source_key(r["source"])
        if source_set and skey not in source_set and r["source"] not in source_set:
            continue
        if prop_set and r["prop"] not in prop_set:
            continue
        if player_search and player_search.lower() not in r["player"].lower():
            continue
        filtered.append(r)

    board = [model_one_prop(r, **model_kwargs) for r in filtered]

    if not show_passes:
        board = [b for b in board if b["qualified"] or b["pick_prob"] >= min_display_prob]
    else:
        board = [b for b in board if b["pick_prob"] >= min_display_prob or b["qualified"] or b["signal"] == "PASS"]

    board.sort(key=lambda x: (
        not x["qualified"],
        -(x["ev"] if x["ev"] is not None else -9),
        -x["pick_prob"],
        -x["edge"],
        -x["data_score"],
    ))
    return board


# ============================================================
# OFFICIAL SNAPSHOTS / MANUAL GRADING
# ============================================================
def official_pick_id(p: Dict[str, Any]) -> str:
    return (
        f"{datetime.now().date()}_"
        f"{source_key(p['source'])}_"
        f"{normalize_name(p['player'])}_"
        f"{p['prop']}_{p['pick_side']}_{round(float(p['line']), 2)}_"
        f"{normalize_name(p.get('game',''))}"
    )

def save_official_picks(board: List[Dict[str, Any]], save_passes: bool = False) -> int:
    picks = load_json(PICK_LOG, [])
    existing = set(p.get("pick_id") for p in picks)
    saved = 0
    for p in board:
        if p["signal"] == "PASS" and not save_passes:
            continue
        pid = official_pick_id(p)
        if pid in existing:
            continue
        picks.append({
            "pick_id": pid,
            "saved_at": now_iso(),
            "source": p["source"],
            "player": p["player"],
            "team": p.get("team", ""),
            "opponent": p.get("opponent", ""),
            "game": p.get("game", ""),
            "start_time": p.get("start_time", ""),
            "prop": p["prop"],
            "prop_label": p["prop_label"],
            "line": p["line"],
            "side": p["pick_side"],
            "price": p["price"],
            "projection": round(float(p["projection"]), 3),
            "pick_prob": round(float(p["pick_prob"]), 4),
            "edge": round(float(p["edge"]), 3),
            "ev": None if p["ev"] is None else round(float(p["ev"]), 4),
            "kelly": round(float(p["kelly"]), 4),
            "stake": round(float(p["stake"]), 2),
            "signal": p["signal"],
            "data_score": p["data_score"],
            "qualified": p["qualified"],
            "graded": False,
        })
        existing.add(pid)
        saved += 1
    save_json(PICK_LOG, picks)
    return saved

def grade_saved_pick(pick_id: str, actual: float) -> bool:
    picks = load_json(PICK_LOG, [])
    results = load_json(RESULT_LOG, [])
    result_ids = set(r.get("pick_id") for r in results)

    changed = False
    for p in picks:
        if p.get("pick_id") != pick_id:
            continue
        line = safe_float(p.get("line"))
        if line is None:
            continue
        side = p.get("side", "").upper()
        win = actual > line if side == "OVER" else actual < line
        p["graded"] = True
        p["actual"] = actual
        p["win"] = bool(win)
        p["graded_at"] = now_iso()
        if pick_id not in result_ids:
            results.append(dict(p))
        changed = True
        break

    if changed:
        save_json(PICK_LOG, picks)
        save_json(RESULT_LOG, results)
        rebuild_learning_from_results()
    return changed

def performance_summary() -> Dict[str, Any]:
    results = load_json(RESULT_LOG, [])
    graded = [r for r in results if r.get("actual") is not None]
    official = [r for r in graded if r.get("qualified")]
    wins = sum(1 for r in official if r.get("win"))
    return {
        "graded": len(graded),
        "official": len(official),
        "wins": wins,
        "hit_rate": wins / len(official) if official else None,
    }


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 4px 12px 4px;'>
      <div style='font-size:28px;font-weight:950;'>😈 DEVIL PICKS</div>
      <div style='color:#ff344f;font-weight:900;'>REAL RAIL PRO</div>
      <div style='color:#aeb7c9;font-size:12px;margin-top:4px;'>Underdog • Sleeper • Odds API fallback</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Source Toggles")
    use_underdog = st.toggle("Underdog Direct + Apify", value=ENABLE_UNDERDOG_DEFAULT)
    use_sleeper = st.toggle("Sleeper Apify", value=ENABLE_SLEEPER_DEFAULT)
    use_odds_api = st.toggle("Odds API fallback", value=ENABLE_ODDS_API_DEFAULT)
    use_opticodds = st.toggle("OpticOdds placeholder", value=ENABLE_OPTICODDS_DEFAULT)

    st.markdown("### Model Toggles")
    use_bayesian = st.toggle("Bayesian confidence", value=os.getenv("ENABLE_BAYESIAN", "true").lower() == "true")
    use_markov = st.toggle("Markov recent form", value=os.getenv("ENABLE_MARKOV", "true").lower() == "true")
    use_learning = st.toggle("Use saved learning", value=os.getenv("ENABLE_LEARNING", "true").lower() == "true")
    use_xgboost = st.toggle("XGBoost/GBM filter", value=os.getenv("ENABLE_XGBOOST", "false").lower() == "true")

    st.markdown("### Board Filters")
    selected_props = st.multiselect("Props", list(PROP_CONFIG.keys()), default=list(PROP_CONFIG.keys()))
    selected_sources = st.multiselect("Sources", ["Underdog", "Underdog Apify", "Sleeper", "OddsAPI"], default=["Underdog", "Underdog Apify", "Sleeper", "OddsAPI"])
    player_search = st.text_input("Search player", "")

    st.markdown("### Gates")
    bankroll = st.number_input("Bankroll", min_value=10.0, value=DEFAULT_BANKROLL, step=25.0)
    sim_count = st.slider("Monte Carlo sims", 1000, 30000, DEFAULT_SIM_COUNT, 1000)
    min_prob = st.slider("Official min probability", 0.50, 0.75, DEFAULT_MIN_PROB, 0.01)
    min_display_prob = st.slider("Display min probability", 0.45, 0.75, 0.52, 0.01)
    min_data_score = st.slider("Min data score", 40, 95, DEFAULT_MIN_DATA_SCORE, 1)
    min_edge_scale = st.slider("Edge gate scale", 0.30, 1.50, 0.70, 0.05)
    max_kelly = st.slider("Max Kelly stake", 0.0, 0.10, DEFAULT_MAX_KELLY, 0.005)
    show_passes = st.toggle("Show PASS rows", value=True)

    st.markdown("### Key Status")
    st.write("APIFY_TOKEN:", "✅ set" if APIFY_TOKEN else "❌ missing")
    st.write("ODDS_API_KEY:", "✅ set" if ODDS_API_KEY else "not set")
    st.write("Underdog URL:", UNDERDOG_API_URL[:42] + ("..." if len(UNDERDOG_API_URL) > 42 else ""))


# ============================================================
# MAIN APP
# ============================================================
st.markdown("""
<div class='hero'>
  <div class='logo-title'>😈 DEVIL PICKS — Real Rail Pro</div>
  <div class='sub'>Real prop board • Underdog/Sleeper source toggles • Bayesian + Markov + Monte Carlo • official snapshots • manual grading + learning</div>
</div>
""", unsafe_allow_html=True)

perf = performance_summary()
learn = load_learning()

top_cols = st.columns(5)
with top_cols[0]:
    refresh = st.button("🔄 Refresh Live Board", use_container_width=True)
with top_cols[1]:
    save_btn = st.button("💾 Save Official Picks", use_container_width=True)
with top_cols[2]:
    clear_cache = st.button("🧹 Clear Cache", use_container_width=True)
with top_cols[3]:
    rebuild_btn = st.button("🧠 Rebuild Learning", use_container_width=True)
with top_cols[4]:
    export_btn = st.button("📤 Export Snapshot", use_container_width=True)

if clear_cache:
    st.cache_data.clear()
    st.success("Cache cleared. Click Refresh Live Board.")

if rebuild_btn:
    n = rebuild_learning_from_results()
    st.success(f"Learning rebuilt from {n} graded results.")

selected_props_tuple = tuple(selected_props)

if refresh or "source_rows" not in st.session_state:
    with st.spinner("Pulling real source props..."):
        st.session_state["source_rows"] = get_all_source_props(
            use_underdog,
            use_sleeper,
            use_odds_api,
            use_opticodds,
            UNDERDOG_API_URL,
            APIFY_TOKEN,
            ODDS_API_KEY,
            selected_props_tuple,
        )
        st.session_state["last_refresh"] = now_iso()

source_rows = st.session_state.get("source_rows", [])

board = build_board(
    source_rows,
    selected_sources=selected_sources,
    selected_props=selected_props,
    player_search=player_search,
    min_display_prob=min_display_prob,
    show_passes=show_passes,
    sim_count=sim_count,
    min_prob=min_prob,
    min_data_score=min_data_score,
    min_edge_scale=min_edge_scale,
    max_kelly=max_kelly,
    bankroll=bankroll,
    use_bayesian=use_bayesian,
    use_markov=use_markov,
    use_learning=use_learning,
    use_xgboost=use_xgboost,
)

if save_btn:
    saved = save_official_picks(board)
    st.success(f"Saved {saved} official picks.")

if export_btn:
    export_path = os.path.join(DATA_DIR, "export_board.csv")
    if board:
        pd.DataFrame(board).drop(columns=["raw"], errors="ignore").to_csv(export_path, index=False)
        st.success(f"Exported board to {export_path}")
    else:
        st.warning("No board rows to export.")

qualified = [b for b in board if b.get("qualified")]
strong = [b for b in qualified if "STRONG" in b.get("signal", "")]
source_counts = pd.Series([source_key(r["source"]) for r in source_rows]).value_counts().to_dict() if source_rows else {}

st.markdown(f"""
<div class='metric-grid'>
  <div class='metric-box'><div class='metric-label'>Source Rows</div><div class='metric-value'>{len(source_rows)}</div><div class='metric-sub'>{source_counts}</div></div>
  <div class='metric-box'><div class='metric-label'>Board Rows</div><div class='metric-value'>{len(board)}</div><div class='metric-sub'>After filters</div></div>
  <div class='metric-box'><div class='metric-label'>Official Plays</div><div class='metric-value'>{len(qualified)}</div><div class='metric-sub'>Gate passed</div></div>
  <div class='metric-box'><div class='metric-label'>Strong Plays</div><div class='metric-value'>{len(strong)}</div><div class='metric-sub'>High confidence</div></div>
  <div class='metric-box'><div class='metric-label'>Hit Rate</div><div class='metric-value'>{pct(perf['hit_rate'])}</div><div class='metric-sub'>{perf['official']} official graded</div></div>
</div>
""", unsafe_allow_html=True)

if not source_rows:
    st.markdown("""
    <div class='card card-bad'>
      <div class='big'>No real source rows loaded</div>
      <div class='sub'>
        This build does not create fake props. Check source toggles and Railway environment variables.
        Underdog direct may need UNDERDOG_API_URL override if their feed changed.
        Sleeper requires APIFY_TOKEN or another enabled feed because Sleeper does not expose a stable public Picks prop-line API in this app.
      </div>
    </div>
    """, unsafe_allow_html=True)

tabs = st.tabs([
    "😈 Top Plays",
    "📋 Full Board",
    "🔎 Source Rows",
    "💾 Official Picks",
    "✅ Grade + Learn",
    "📈 Performance",
    "🔌 Source Logs",
    "⚙️ Deployment",
])

with tabs[0]:
    st.markdown("<div class='section-title'>Top Real Plays</div>", unsafe_allow_html=True)
    if not board:
        st.info("No board rows after filters.")
    for p in board[:50]:
        card_class = "card card-good" if p["qualified"] else "card card-warn"
        reason_text = ", ".join(p["reasons"]) if p["reasons"] else "gates passed"
        notes = " • ".join(p.get("model_notes", []))
        st.markdown(f"""
        <div class='{card_class}'>
          <div class='big'>{p['player']} — {p['prop_label']}</div>
          <span class='badge badge-blue'>{p['source']}</span>
          <span class='badge {'badge-green' if p['qualified'] else 'badge-orange'}'>{p['signal']}</span>
          <span class='badge'>Pick: {p['pick_side']}</span>
          <span class='badge'>Line: {p['line']}</span>
          <span class='badge'>Price: {odds_display(p['price'])}</span>
          <span class='badge'>State: {p['markov_state']}</span>
          <div class='metric-grid'>
            <div class='metric-box'><div class='metric-label'>Projection</div><div class='metric-value'>{p['projection']:.2f}</div><div class='metric-sub'>Edge {p['edge']:+.2f}</div></div>
            <div class='metric-box'><div class='metric-label'>Over</div><div class='metric-value'>{p['over_prob']*100:.1f}%</div><div class='metric-sub'>Under {p['under_prob']*100:.1f}%</div></div>
            <div class='metric-box'><div class='metric-label'>Pick Prob</div><div class='metric-value'>{p['pick_prob']*100:.1f}%</div><div class='metric-sub'>Raw {p['raw_pick_prob']*100:.1f}%</div></div>
            <div class='metric-box'><div class='metric-label'>EV / Kelly</div><div class='metric-value'>{(p['ev'] or 0)*100:.1f}%</div><div class='metric-sub'>Stake ${p['stake']:.2f}</div></div>
            <div class='metric-box'><div class='metric-label'>Data Score</div><div class='metric-value'>{p['data_score']}</div><div class='metric-sub'>CLV {p['clv']:+.2f}</div></div>
          </div>
          <div class='sub'>Game: {p.get('game','')} | Team: {p.get('team','')} | Notes: {reason_text}</div>
          <div class='sub'>Model: {notes}</div>
        </div>
        """, unsafe_allow_html=True)

with tabs[1]:
    if board:
        display_cols = [
            "signal", "source", "player", "team", "game", "prop", "line", "pick_side",
            "projection", "over_prob", "under_prob", "pick_prob", "edge", "ev",
            "kelly", "stake", "clv", "data_score", "markov_state", "reasons",
        ]
        df = pd.DataFrame(board).drop(columns=["raw"], errors="ignore")
        st.dataframe(df[[c for c in display_cols if c in df.columns]], use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download current board CSV", csv, file_name="devil_picks_board.csv", mime="text/csv")
    else:
        st.info("No board rows.")

with tabs[2]:
    if source_rows:
        raw_df = pd.DataFrame(source_rows).drop(columns=["raw"], errors="ignore")
        st.dataframe(raw_df, use_container_width=True, hide_index=True)
    else:
        st.info("No source rows loaded.")

with tabs[3]:
    picks = load_json(PICK_LOG, [])
    if picks:
        st.dataframe(pd.DataFrame(picks), use_container_width=True, hide_index=True)
    else:
        st.info("No official picks saved yet.")

with tabs[4]:
    st.markdown("<div class='section-title'>Manual Grading</div>", unsafe_allow_html=True)
    picks = load_json(PICK_LOG, [])
    ungraded = [p for p in picks if not p.get("graded")]
    if not ungraded:
        st.info("No ungraded official picks.")
    else:
        pick_labels = [
            f"{p['player']} {p['prop']} {p['side']} {p['line']} | {p['source']} | {p.get('game','')}"
            for p in ungraded
        ]
        idx = st.selectbox("Pick to grade", range(len(ungraded)), format_func=lambda i: pick_labels[i])
        actual = st.number_input("Actual result", value=0.0, step=0.5)
        if st.button("✅ Grade Selected Pick", use_container_width=True):
            ok = grade_saved_pick(ungraded[idx]["pick_id"], actual)
            if ok:
                st.success("Pick graded and learning updated.")
            else:
                st.error("Could not grade selected pick.")

    st.markdown("### Add External/Manual Result")
    with st.form("manual_result_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            m_player = st.text_input("Player")
        with c2:
            m_prop = st.selectbox("Prop", list(PROP_CONFIG.keys()))
        with c3:
            m_line = st.number_input("Line", value=0.0, step=0.5)
        with c4:
            m_actual = st.number_input("Actual", value=0.0, step=0.5)
        m_source = st.text_input("Source", value="Manual")
        submitted = st.form_submit_button("Add Manual Result + Rebuild Learning")
        if submitted and m_player:
            results = load_json(RESULT_LOG, [])
            side = "OVER" if m_actual > m_line else "UNDER"
            results.append({
                "pick_id": f"manual_{now_iso()}_{normalize_name(m_player)}_{m_prop}",
                "saved_at": now_iso(),
                "source": m_source,
                "player": m_player,
                "prop": m_prop,
                "prop_label": PROP_CONFIG[m_prop]["label"],
                "line": m_line,
                "actual": m_actual,
                "side": side,
                "win": True,
                "qualified": False,
                "manual": True,
            })
            save_json(RESULT_LOG, results)
            n = rebuild_learning_from_results()
            st.success(f"Manual result saved. Learning rebuilt from {n} results.")

with tabs[5]:
    summary = performance_summary()
    st.markdown(f"""
    <div class='metric-grid-4'>
      <div class='metric-box'><div class='metric-label'>Total Graded</div><div class='metric-value'>{summary['graded']}</div></div>
      <div class='metric-box'><div class='metric-label'>Official Graded</div><div class='metric-value'>{summary['official']}</div></div>
      <div class='metric-box'><div class='metric-label'>Wins</div><div class='metric-value'>{summary['wins']}</div></div>
      <div class='metric-box'><div class='metric-label'>Hit Rate</div><div class='metric-value'>{pct(summary['hit_rate'])}</div></div>
    </div>
    """, unsafe_allow_html=True)

    results = load_json(RESULT_LOG, [])
    if results:
        rdf = pd.DataFrame(results)
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        if "prop" in rdf.columns and "win" in rdf.columns:
            try:
                by_prop = rdf.dropna(subset=["win"]).groupby("prop")["win"].agg(["count", "mean"]).reset_index()
                by_prop["mean"] = by_prop["mean"].map(lambda x: f"{x*100:.1f}%")
                st.subheader("By Prop")
                st.dataframe(by_prop, use_container_width=True, hide_index=True)
            except Exception:
                pass
    else:
        st.info("No graded results yet.")

    st.subheader("Learning State")
    st.json(load_learning())

with tabs[6]:
    logs = load_json(REQUEST_LOG_FILE, [])
    if logs:
        st.dataframe(pd.DataFrame(logs[-300:]), use_container_width=True, hide_index=True)
    else:
        st.info("No source logs yet.")

with tabs[7]:
    st.markdown("""
    ### Railway Environment Variables

    Minimum for Underdog direct:
    ```text
    ENABLE_UNDERDOG=true
    ENABLE_SLEEPER=false
    ENABLE_ODDS_API=false
    ```

    For Underdog + Sleeper through Apify:
    ```text
    APIFY_TOKEN=your_apify_token
    ENABLE_UNDERDOG=true
    ENABLE_SLEEPER=true
    ```

    Optional Odds API fallback:
    ```text
    ODDS_API_KEY=your_odds_api_key
    ENABLE_ODDS_API=true
    ```

    Optional model controls:
    ```text
    ENABLE_BAYESIAN=true
    ENABLE_MARKOV=true
    ENABLE_LEARNING=true
    ENABLE_XGBOOST=false
    PROP_SIMULATION_COUNT=12000
    MIN_PROB=0.57
    MIN_DATA_SCORE=68
    BANKROLL=1000
    ```

    Railway start command is handled by the included `Procfile`.
    """)
