# -*- coding: utf-8 -*-
# ============================================================
# DEVIL PICKS — UNDERDOG ONLY REAL RAIL BUILD
# Railway / GitHub / Streamlit ready
#
# Real Underdog-only prop engine:
# - Pulls Underdog lines only
# - No Sleeper
# - No Odds API
# - No fake demo rows
# - Clean UI with toggles/buttons
# - Monte Carlo + Bayesian + Markov + learning + CLV
# - Save official picks
# - Manual grading after games
# ============================================================

import os
import re
import json
import math
import time
import difflib
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import pytz
except Exception:
    pytz = None


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Devil Picks — Underdog Only",
    page_icon="😈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

PICK_LOG = os.path.join(DATA_DIR, "underdog_official_picks.json")
RESULT_LOG = os.path.join(DATA_DIR, "underdog_results.json")
LEARN_FILE = os.path.join(DATA_DIR, "underdog_learning.json")
CLV_FILE = os.path.join(DATA_DIR, "underdog_clv.json")
REQUEST_LOG_FILE = os.path.join(DATA_DIR, "underdog_source_logs.json")
SNAPSHOT_FILE = os.path.join(DATA_DIR, "latest_underdog_board.json")

UNDERDOG_API_URL = os.getenv(
    "UNDERDOG_API_URL",
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
).strip()

NBA_PLAYER_INDEX = "https://stats.nba.com/stats/playerindex"
NBA_PLAYER_GAMELOG = "https://stats.nba.com/stats/playergamelog"

DEFAULT_PRICE = float(os.getenv("DEFAULT_PRICE", "-110"))
# Underdog pick'em is not standard sportsbook -110 pricing. This EV uses a configurable
# fair-price proxy so the EV column is useful. Set DEFAULT_PRICE/EV_PRICE in Railway if desired.
EV_PRICE = float(os.getenv("EV_PRICE", os.getenv("DEFAULT_PRICE", "-110")))
DEFAULT_SIM_COUNT = int(os.getenv("PROP_SIMULATION_COUNT", "12000"))
DEFAULT_MIN_PROB = float(os.getenv("MIN_PROB", "0.57"))
DEFAULT_MIN_DATA_SCORE = int(os.getenv("MIN_DATA_SCORE", "68"))
DEFAULT_MAX_KELLY = float(os.getenv("MAX_KELLY", "0.03"))
DEFAULT_BANKROLL = float(os.getenv("BANKROLL", "1000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "120"))
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/Los_Angeles")

PROP_CONFIG = {
    "PTS": {"label": "Points", "min_edge": 1.8, "std": 5.8},
    "REB": {"label": "Rebounds", "min_edge": 1.5, "std": 3.0},
    "AST": {"label": "Assists", "min_edge": 1.5, "std": 2.8},
    "PRA": {"label": "Pts + Reb + Ast", "min_edge": 2.5, "std": 7.0},
    "PR": {"label": "Pts + Reb", "min_edge": 2.2, "std": 6.2},
    "PA": {"label": "Pts + Ast", "min_edge": 2.2, "std": 6.2},
    "RA": {"label": "Reb + Ast", "min_edge": 2.0, "std": 4.2},
    "3PM": {"label": "3PM", "min_edge": 0.65, "std": 1.35},
    "BLK": {"label": "Blocks", "min_edge": 0.45, "std": 0.85},
    "STL": {"label": "Steals", "min_edge": 0.45, "std": 0.85},
    "TO": {"label": "Turnovers", "min_edge": 0.70, "std": 1.25},
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
    "blocks": "BLK", "blocked shots": "BLK", "blk": "BLK",
    "steals": "STL", "stl": "STL",
    "turnovers": "TO", "tos": "TO", "to": "TO",
}

NBA_TEAM_ABBRS = {
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW","HOU","IND",
    "LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK","OKC","ORL","PHI","PHX",
    "POR","SAC","SAS","TOR","UTA","WAS"
}

NBA_TEAM_NAMES = {
    "hawks","celtics","nets","hornets","bulls","cavaliers","mavericks","nuggets",
    "pistons","warriors","rockets","pacers","clippers","lakers","grizzlies","heat",
    "bucks","timberwolves","pelicans","knicks","thunder","magic","76ers","suns",
    "trail blazers","blazers","kings","spurs","raptors","jazz","wizards"
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
# STORAGE
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
        "message": str(message)[:700],
    })
    save_json(REQUEST_LOG_FILE, rows[-1000:])


# ============================================================
# HELPERS
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

def expected_value(prob: Optional[float], odds: Any = EV_PRICE) -> Optional[float]:
    """
    EV per 1 unit risked using an American-odds proxy.
    For Underdog, this is a modeling proxy, not an official sportsbook price.
    """
    dec = american_to_decimal(odds)
    if prob is None or dec is None:
        return None
    return float(prob * (dec - 1) - (1 - prob))

def edge_metrics(projection: float, line: float, side: str, over_prob: float, under_prob: float) -> Dict[str, float]:
    """
    Direction-aware edge:
    - raw_edge = projection - line
    - pick_edge = positive amount supporting the selected side
      OVER: projection - line
      UNDER: line - projection
    - edge_pct = pick_edge / line when possible
    """
    raw_edge = float(projection) - float(line)
    side = str(side).upper()
    pick_edge = raw_edge if side == "OVER" else -raw_edge
    line_abs = abs(float(line)) if abs(float(line)) > 0 else 1.0
    edge_pct = pick_edge / line_abs
    prob_edge = max(float(over_prob), float(under_prob)) - 0.50
    return {
        "raw_edge": float(raw_edge),
        "pick_edge": float(pick_edge),
        "edge_pct": float(edge_pct),
        "prob_edge": float(prob_edge),
    }

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


def current_nba_season() -> str:
    """
    NBA season string used by stats.nba.com.
    Example: before July 2026 returns 2025-26; after July returns 2026-27.
    """
    now = datetime.now(local_tz()) if "local_tz" in globals() else datetime.now()
    y = now.year
    if now.month < 7:
        start = y - 1
    else:
        start = y
    return f"{start}-{str(start + 1)[-2:]}"

def stat_prop_value_from_log_row(row: Dict[str, Any], prop: str) -> float:
    def sf(k):
        return safe_float(row.get(k), 0.0) or 0.0
    if prop == "PTS":
        return sf("PTS")
    if prop == "REB":
        return sf("REB")
    if prop == "AST":
        return sf("AST")
    if prop == "PRA":
        return sf("PTS") + sf("REB") + sf("AST")
    if prop == "PR":
        return sf("PTS") + sf("REB")
    if prop == "PA":
        return sf("PTS") + sf("AST")
    if prop == "RA":
        return sf("REB") + sf("AST")
    if prop == "3PM":
        return sf("FG3M")
    if prop == "BLK":
        return sf("BLK")
    if prop == "STL":
        return sf("STL")
    if prop == "TO":
        return sf("TOV")
    return 0.0



# ============================================================
# HTTP
# ============================================================
def safe_get_json(url: str, timeout: int = 30) -> Any:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://underdogfantasy.com/",
            "Origin": "https://underdogfantasy.com",
            "Connection": "keep-alive",
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            log_request(url, f"HTTP {r.status_code}", r.text[:500])
            return None
        return r.json()
    except Exception as e:
        log_request(url, "REQUEST_ERROR", str(e))
        return None


def safe_get_nba_json(url: str, params: Optional[dict] = None, timeout: int = 25) -> Any:
    try:
        headers = {
            "Host": "stats.nba.com",
            "Connection": "keep-alive",
            "Accept": "application/json, text/plain, */*",
            "x-nba-stats-token": "true",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "x-nba-stats-origin": "stats",
            "Origin": "https://www.nba.com",
            "Referer": "https://www.nba.com/",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            log_request(url, f"NBA_HTTP {r.status_code}", r.text[:400])
            return None
        return r.json()
    except Exception as e:
        log_request(url, "NBA_REQUEST_ERROR", str(e))
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_nba_player_index(season: str) -> List[Dict[str, Any]]:
    params = {
        "College": "",
        "Country": "",
        "DraftPick": "",
        "DraftRound": "",
        "DraftYear": "",
        "Height": "",
        "Historical": "0",
        "LeagueID": "00",
        "Season": season,
        "SeasonType": "Regular Season",
        "TeamID": "0",
        "Weight": "",
    }
    data = safe_get_nba_json(NBA_PLAYER_INDEX, params=params, timeout=25)
    rows = []
    try:
        rs = data.get("resultSets", [])[0]
        headers = rs.get("headers", [])
        for r in rs.get("rowSet", []):
            d = dict(zip(headers, r))
            rows.append(d)
    except Exception as e:
        log_request("NBA_PLAYER_INDEX", "PARSE_ERROR", str(e))
    return rows

def best_nba_player_match(name: str, season: str) -> Tuple[Optional[int], float, str]:
    players = get_nba_player_index(season)
    best_id, best_score, best_name = None, 0.0, ""
    target = normalize_name(name)
    for p in players:
        cand = (
            p.get("PLAYER_NAME")
            or p.get("PLAYER")
            or p.get("DISPLAY_FIRST_LAST")
            or p.get("PLAYER_SLUG")
            or ""
        )
        score = difflib.SequenceMatcher(None, target, normalize_name(cand)).ratio()
        if target and (target in normalize_name(cand) or normalize_name(cand) in target):
            score = max(score, 0.94)
        if score > best_score:
            best_score = score
            best_id = safe_int(p.get("PERSON_ID") or p.get("PLAYER_ID"))
            best_name = str(cand)
    return best_id, float(best_score), best_name

@st.cache_data(ttl=1800, show_spinner=False)
def get_nba_player_gamelog(player_id: int, season: str) -> pd.DataFrame:
    params = {
        "PlayerID": str(player_id),
        "Season": season,
        "SeasonType": "Regular Season",
        "LeagueID": "00",
    }
    data = safe_get_nba_json(NBA_PLAYER_GAMELOG, params=params, timeout=25)
    try:
        rs = data.get("resultSets", [])[0]
        headers = rs.get("headers", [])
        df = pd.DataFrame(rs.get("rowSet", []), columns=headers)
        for c in ["PTS", "REB", "AST", "FG3M", "BLK", "STL", "TOV", "MIN"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        log_request("NBA_PLAYER_GAMELOG", "PARSE_ERROR", str(e))
        return pd.DataFrame()

def true_stat_projection(player_name: str, prop: str, line: float, use_real_stats: bool, season: str) -> Dict[str, Any]:
    """
    Builds an actual player projection from NBA game logs.
    If stats are unavailable, returns a market fallback and flags it clearly.
    """
    fallback = {
        "projection": float(line),
        "std": float(PROP_CONFIG[prop].get("std", 3.0)),
        "source": "market_fallback",
        "note": "No NBA log projection available; using line as fallback",
        "nba_player_id": None,
        "nba_match_score": 0.0,
        "nba_match_name": "",
        "l3": None, "l5": None, "l10": None, "season_avg": None, "minutes": None,
    }
    if not use_real_stats:
        fallback["note"] = "Real NBA stats toggle off; using line fallback"
        return fallback

    pid, score, match_name = best_nba_player_match(player_name, season)
    if not pid or score < 0.78:
        fallback["note"] = f"NBA player match not strong enough ({score:.2f}); using line fallback"
        fallback["nba_match_score"] = score
        fallback["nba_match_name"] = match_name
        return fallback

    logs = get_nba_player_gamelog(pid, season)
    if logs.empty:
        fallback["note"] = "NBA game logs empty; using line fallback"
        fallback["nba_player_id"] = pid
        fallback["nba_match_score"] = score
        fallback["nba_match_name"] = match_name
        return fallback

    values = []
    for _, r in logs.head(15).iterrows():
        values.append(stat_prop_value_from_log_row(r.to_dict(), prop))
    values = [float(v) for v in values if v is not None]
    if not values:
        fallback["note"] = "NBA values empty; using line fallback"
        fallback["nba_player_id"] = pid
        fallback["nba_match_score"] = score
        fallback["nba_match_name"] = match_name
        return fallback

    l3 = float(np.mean(values[:3])) if len(values) >= 3 else float(np.mean(values))
    l5 = float(np.mean(values[:5])) if len(values) >= 5 else float(np.mean(values))
    l10 = float(np.mean(values[:10])) if len(values) >= 10 else float(np.mean(values))
    season_avg = float(np.mean(values))
    minutes = float(pd.to_numeric(logs.head(5)["MIN"], errors="coerce").mean()) if "MIN" in logs.columns else None

    # Weighted real projection, not just the Underdog line.
    projection = (l3 * 0.30) + (l5 * 0.25) + (l10 * 0.25) + (season_avg * 0.20)

    if minutes is not None:
        if minutes < 18:
            projection *= 0.90
        elif minutes > 34:
            projection *= 1.03

    recent_vals = values[:10]
    std = float(np.std(recent_vals)) if len(recent_vals) >= 3 else float(PROP_CONFIG[prop].get("std", 3.0))
    std = max(std, float(PROP_CONFIG[prop].get("std", 3.0)) * 0.55, 0.45)

    return {
        "projection": float(projection),
        "std": float(std),
        "source": "nba_gamelog",
        "note": f"NBA logs: L3 {l3:.2f}, L5 {l5:.2f}, L10 {l10:.2f}, season {season_avg:.2f}",
        "nba_player_id": pid,
        "nba_match_score": score,
        "nba_match_name": match_name,
        "l3": l3, "l5": l5, "l10": l10, "season_avg": season_avg, "minutes": minutes,
    }




# ============================================================
# UNDERDOG DATA-SHAPE HELPERS
# ============================================================
def first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d.get(k) not in [None, ""]:
            return d.get(k)
    return default

def get_nested(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur = d
    try:
        for p in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(p)
        return cur if cur not in [None, ""] else default
    except Exception:
        return default

def local_tz():
    if pytz:
        try:
            return pytz.timezone(APP_TIMEZONE)
        except Exception:
            return pytz.timezone("America/Los_Angeles")
    return timezone(timedelta(hours=-7))

def parse_datetime_any(value: Any) -> Optional[datetime]:
    if value in [None, ""]:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Common ISO/Z formats.
    try:
        ss = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(local_tz())
    except Exception:
        pass

    # Epoch seconds or ms.
    try:
        x = float(s)
        if x > 10_000_000_000:
            x = x / 1000.0
        return datetime.fromtimestamp(x, tz=timezone.utc).astimezone(local_tz())
    except Exception:
        pass

    return None

def date_bucket_for_start(start_value: Any) -> str:
    dt = parse_datetime_any(start_value)
    if not dt:
        return "unknown"
    today = datetime.now(local_tz()).date()
    d = dt.date()
    if d == today:
        return "Today"
    if d == today + timedelta(days=1):
        return "Tomorrow"
    if d < today:
        return "Past"
    return "Future"

def is_start_passed(start_value: Any, grace_hours: float = 4.0) -> bool:
    """
    Treat games as stale only after start time + grace window, so live games don't vanish immediately.
    """
    dt = parse_datetime_any(start_value)
    if not dt:
        return False
    return datetime.now(local_tz()) > dt + timedelta(hours=grace_hours)

def game_matches_day_filter(start_value: Any, day_filter: str, include_live_grace: bool = True) -> bool:
    bucket = date_bucket_for_start(start_value)
    if bucket == "Past":
        return False
    if is_start_passed(start_value, 4.0 if include_live_grace else 0.0):
        return False
    if day_filter == "Today":
        return bucket == "Today"
    if day_filter == "Tomorrow":
        return bucket == "Tomorrow"
    if day_filter == "Today + Tomorrow":
        return bucket in ["Today", "Tomorrow"]
    if day_filter == "All Future":
        return bucket in ["Today", "Tomorrow", "Future", "unknown"]
    return True

def looks_like_nba_row(player: Dict[str, Any], app: Dict[str, Any], game_obj: Dict[str, Any], team: str, opponent: str, game: str) -> bool:
    """
    Strict but not brittle NBA filter.
    """
    fields = []
    for obj in [player, app, game_obj]:
        if isinstance(obj, dict):
            for k in [
                "sport", "sport_id", "sport_name", "league", "league_id", "league_name",
                "organization", "organization_name", "competition", "competition_name",
                "title", "game_type", "match_type"
            ]:
                v = obj.get(k)
                if v is not None:
                    fields.append(str(v).lower())

    joined = " ".join(fields + [str(team).lower(), str(opponent).lower(), str(game).lower()])

    if any(token in joined for token in ["nba", "basketball"]):
        return True

    team_up = str(team or "").upper()
    opp_up = str(opponent or "").upper()
    if team_up in NBA_TEAM_ABBRS or opp_up in NBA_TEAM_ABBRS:
        return True

    game_l = str(game or "").lower()
    if any(name in game_l for name in NBA_TEAM_NAMES):
        return True

    return False

def player_name_from_player_obj(player: Dict[str, Any], app: Dict[str, Any]) -> str:
    if not isinstance(player, dict):
        player = {}
    if not isinstance(app, dict):
        app = {}

    direct = first_present(player, [
        "full_name", "fullName", "display_name", "displayName", "name", "player_name", "title"
    ])
    if direct:
        return str(direct).strip()

    first = first_present(player, ["first_name", "firstName", "first", "given_name"], "")
    last = first_present(player, ["last_name", "lastName", "last", "family_name"], "")
    combo = f"{first} {last}".strip()
    if combo:
        return combo

    app_name = first_present(app, ["player_name", "name", "display_name", "displayName"], "")
    return str(app_name or "").strip()

def stat_name_from_underdog(line_item: Dict[str, Any], ou: Dict[str, Any], app_stat: Dict[str, Any]) -> Any:
    # Correct stat usually lives under over_under.appearance_stat.stat.
    return (
        first_present(app_stat, ["stat", "display_stat", "stat_type", "type"])
        or first_present(ou, ["stat", "stat_type", "display_stat", "title"])
        or first_present(line_item, ["stat", "stat_type", "display_stat", "title"])
    )

def line_value_from_underdog(line_item: Dict[str, Any], ou: Dict[str, Any], app_stat: Dict[str, Any]) -> Any:
    """
    Corrected line priority.

    Underdog's real v5 board usually stores the displayed line at the
    over_under level, often as stat_value. Some old code grabbed line_item fields
    first, which can be stale/alternate/internal values. This function prioritizes:
      1) over_under.stat_value / display_stat_value / line
      2) line_item.stat_value only if over_under is missing
      3) option fields only as last resort
    """
    preferred = first_present(ou, [
        "stat_value", "display_stat_value", "line", "value", "over_under_value", "value_text"
    ])
    if preferred not in [None, ""]:
        return preferred

    second = first_present(line_item, [
        "stat_value", "display_stat_value", "line", "value", "over_under_value"
    ])
    if second not in [None, ""]:
        return second

    # Sometimes options contain display_value/stat_value. Use median to avoid choosing Higher/Lower label.
    opts = line_item.get("options")
    vals = []
    if isinstance(opts, list):
        for opt in opts:
            if not isinstance(opt, dict):
                continue
            v = first_present(opt, ["stat_value", "display_stat_value", "line", "value"])
            fv = safe_float(v)
            if fv is not None:
                vals.append(fv)
    if vals:
        return float(np.median(vals))

    return first_present(app_stat, ["line", "value", "stat_value"])

def status_is_open_line(line_item: Dict[str, Any], ou: Dict[str, Any], app: Dict[str, Any], game_obj: Dict[str, Any]) -> bool:
    """
    Skip removed/suspended/settled rows when the feed marks them that way.
    """
    joined = " ".join(
        str(first_present(obj, [
            "status", "state", "display_status", "live_event_status", "status_text", "result_status"
        ], "")).lower()
        for obj in [line_item, ou, app, game_obj]
        if isinstance(obj, dict)
    )
    bad = ["settled", "closed", "removed", "cancelled", "canceled", "suspended", "graded", "resulted", "inactive"]
    return not any(x in joined for x in bad)



# ============================================================
# UNDERDOG PARSER
# ============================================================
def normalize_prop_row(
    player: Any,
    prop: Any,
    line: Any,
    team: Any = "",
    opponent: Any = "",
    game: Any = "",
    start_time: Any = "",
    price: Any = DEFAULT_PRICE,
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
        "source": "Underdog",
        "player": player,
        "team": str(team or "").strip(),
        "opponent": str(opponent or "").strip(),
        "game": str(game or "").strip(),
        "start_time": str(start_time or "").strip(),
        "prop": prop_key,
        "prop_label": PROP_CONFIG[prop_key]["label"],
        "line": float(line),
        "price": float(safe_float(price, DEFAULT_PRICE)),
        "raw": raw or {},
    }

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_underdog_lines(url: str, strict_nba: bool = True, day_filter: str = 'Today + Tomorrow') -> List[Dict[str, Any]]:
    """
    Pull real Underdog NBA lines only.

    Important fix:
    - Lines are joined strictly through:
      over_under_lines[*].over_under.appearance_stat.appearance_id
      -> appearances[id].player_id
      -> players[id]
    - Cross-sport rows are filtered out when strict_nba=True.
    - Rows with missing appearance/player joins are skipped and logged.
    """
    candidate_urls = []
    if url:
        candidate_urls.append(url)
    for fallback in [
        "https://api.underdogfantasy.com/beta/v5/over_under_lines",
        "https://api.underdogfantasy.com/v5/over_under_lines",
        "https://api.underdogfantasy.com/v1/over_under_lines",
    ]:
        if fallback not in candidate_urls:
            candidate_urls.append(fallback)

    data = None
    used_url = ""
    for u in candidate_urls:
        data = safe_get_json(u, timeout=30)
        if data:
            used_url = u
            break

    if not data:
        log_request("Underdog", "NO_DATA", "All Underdog endpoints returned empty/blocked")
        return []

    rows: List[Dict[str, Any]] = []
    skipped = {"no_join": 0, "not_nba": 0, "bad_prop": 0, "bad_line": 0, "bad_date": 0, "closed": 0}

    # Current v5 shape: players, appearances, games/matches, over_under_lines
    if isinstance(data, dict) and isinstance(data.get("players"), list) and isinstance(data.get("appearances"), list):
        players_raw = data.get("players", []) or []
        appearances_raw = data.get("appearances", []) or []
        lines_raw = data.get("over_under_lines", []) or []
        games_raw = data.get("games", []) or data.get("matches", []) or []

        players = {str(p.get("id")): p for p in players_raw if isinstance(p, dict) and p.get("id") is not None}
        appearances = {str(a.get("id")): a for a in appearances_raw if isinstance(a, dict) and a.get("id") is not None}
        games = {str(g.get("id")): g for g in games_raw if isinstance(g, dict) and g.get("id") is not None}

        for line_item in lines_raw:
            if not isinstance(line_item, dict):
                continue

            ou = line_item.get("over_under") if isinstance(line_item.get("over_under"), dict) else {}
            app_stat = ou.get("appearance_stat") if isinstance(ou.get("appearance_stat"), dict) else {}

            # THE IMPORTANT JOIN: line -> appearance -> player.
            appearance_id = (
                first_present(app_stat, ["appearance_id", "appearanceId"])
                or first_present(ou, ["appearance_id", "appearanceId"])
                or first_present(line_item, ["appearance_id", "appearanceId"])
            )
            app = appearances.get(str(appearance_id), {}) if appearance_id is not None else {}

            player_id = (
                first_present(app, ["player_id", "playerId"])
                or first_present(line_item, ["player_id", "playerId"])
                or first_present(ou, ["player_id", "playerId"])
            )
            player = players.get(str(player_id), {}) if player_id is not None else {}

            if not app or not player:
                skipped["no_join"] += 1
                continue

            stat_name = stat_name_from_underdog(line_item, ou, app_stat)
            prop_key = clean_prop_type(stat_name)
            if prop_key not in PROP_CONFIG:
                skipped["bad_prop"] += 1
                continue

            line_value = line_value_from_underdog(line_item, ou, app_stat)
            if safe_float(line_value) is None:
                skipped["bad_line"] += 1
                continue

            player_name = player_name_from_player_obj(player, app)

            team = (
                first_present(app, ["team_abbr", "teamAbbr", "team"])
                or first_present(player, ["team_abbr", "teamAbbr", "team"])
                or ""
            )
            opponent = first_present(app, ["opponent_abbr", "opponentAbbr", "opponent"], "")

            game_id = (
                first_present(app, ["game_id", "gameId", "match_id", "matchId"])
                or first_present(line_item, ["game_id", "gameId", "match_id", "matchId"])
            )
            game_obj = games.get(str(game_id), {}) if game_id is not None else {}

            game = (
                first_present(app, ["match_title", "matchTitle", "game_title", "gameTitle"])
                or first_present(line_item, ["match_title", "matchTitle", "game"])
                or first_present(game_obj, ["title", "name", "match_title", "matchTitle"])
                or ""
            )
            if not game and isinstance(game_obj, dict):
                away = first_present(game_obj, ["away_team_name", "awayTeamName", "away_team", "awayTeam"], "")
                home = first_present(game_obj, ["home_team_name", "homeTeamName", "home_team", "homeTeam"], "")
                game = f"{away} @ {home}".strip(" @")

            start_time = (
                first_present(app, ["scheduled_at", "scheduledAt", "start_time", "startTime"])
                or first_present(game_obj, ["scheduled_at", "scheduledAt", "start_time", "startTime"])
                or ""
            )

            if strict_nba and not looks_like_nba_row(player, app, game_obj, team, opponent, game):
                skipped["not_nba"] += 1
                continue

            if not status_is_open_line(line_item, ou, app, game_obj):
                skipped["closed"] += 1
                continue

            if not game_matches_day_filter(start_time, day_filter):
                skipped["bad_date"] += 1
                continue

            option_price = DEFAULT_PRICE
            options = line_item.get("options")
            if isinstance(options, list) and options:
                prices = []
                for opt in options:
                    if not isinstance(opt, dict):
                        continue
                    for key in ["american_odds", "americanOdds", "odds", "price"]:
                        val = safe_float(opt.get(key))
                        if val is not None:
                            prices.append(val)
                if prices:
                    option_price = float(np.median(prices))

            row = normalize_prop_row(
                player=player_name,
                prop=stat_name,
                line=line_value,
                team=team,
                opponent=opponent,
                game=game,
                start_time=start_time,
                price=option_price,
                raw={
                    "line_id": line_item.get("id"),
                    "appearance_id": appearance_id,
                    "player_id": player_id,
                    "stat_name": stat_name,
                    "game_id": game_id,
                    "raw_line": line_item,
                },
            )
            if row:
                row["appearance_id"] = str(appearance_id)
                row["player_id"] = str(player_id)
                row["date_bucket"] = date_bucket_for_start(start_time)
                row["local_start"] = str(parse_datetime_any(start_time) or "")
                rows.append(row)

        log_request("Underdog", "OK", f"{len(rows)} NBA rows from {used_url}; skipped={skipped}")
        save_json(SNAPSHOT_FILE, {"created_at": now_iso(), "rows": rows, "skipped": skipped, "url": used_url})
        return rows

    # Older JSON/API fallback shape. This is stricter than before.
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
        attrs["id"] = item_id
        if "appearance" in typ:
            appearances[item_id] = attrs
        if "player" in typ:
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
        over_under = attrs.get("over_under") if isinstance(attrs.get("over_under"), dict) else {}
        app_stat = over_under.get("appearance_stat") if isinstance(over_under.get("appearance_stat"), dict) else {}

        stat_type = stat_name_from_underdog(attrs, over_under, app_stat)
        if clean_prop_type(stat_type) not in PROP_CONFIG:
            skipped["bad_prop"] += 1
            continue

        line = line_value_from_underdog(attrs, over_under, app_stat)
        if safe_float(line) is None:
            skipped["bad_line"] += 1
            continue

        appearance_id = None
        player_id = None
        try:
            appearance_id = rel.get("appearance", {}).get("data", {}).get("id")
        except Exception:
            pass
        try:
            player_id = rel.get("player", {}).get("data", {}).get("id")
        except Exception:
            pass

        if not appearance_id:
            appearance_id = first_present(app_stat, ["appearance_id", "appearanceId"])

        app = appearances.get(str(appearance_id), {}) if appearance_id else {}
        if not player_id:
            player_id = first_present(app, ["player_id", "playerId"])
        pl = players.get(str(player_id), {}) if player_id else {}

        if not pl:
            skipped["no_join"] += 1
            continue

        player_name = player_name_from_player_obj(pl, app)
        team = first_present(attrs, ["team", "team_abbr"], "") or first_present(app, ["team_abbr", "team"], "")
        opponent = first_present(attrs, ["opponent"], "") or first_present(app, ["opponent_abbr", "opponent"], "")
        game = first_present(attrs, ["match_title", "game"], "") or first_present(app, ["match_title"], "")

        if strict_nba and not looks_like_nba_row(pl, app, {}, team, opponent, game):
            skipped["not_nba"] += 1
            continue

        start_time = first_present(attrs, ["scheduled_at", "start_time"], "") or first_present(app, ["scheduled_at"], "")

        if not status_is_open_line(attrs, over_under, app, {}):
            skipped["closed"] += 1
            continue

        if not game_matches_day_filter(start_time, day_filter):
            skipped["bad_date"] += 1
            continue

        row = normalize_prop_row(
            player=player_name,
            prop=stat_type,
            line=line,
            team=team,
            opponent=opponent,
            game=game,
            start_time=start_time,
            price=DEFAULT_PRICE,
            raw={
                "appearance_id": appearance_id,
                "player_id": player_id,
                "stat_name": stat_type,
                "raw_line": attrs,
            },
        )
        if row:
            row["appearance_id"] = str(appearance_id)
            row["player_id"] = str(player_id)
            rows.append(row)

    log_request("Underdog", "OK", f"{len(rows)} NBA rows from {used_url} fallback parser; skipped={skipped}")
    save_json(SNAPSHOT_FILE, {"created_at": now_iso(), "rows": rows, "skipped": skipped, "url": used_url})
    return rows


# ============================================================
# LEARNING / CLV
# ============================================================
def load_learning() -> Dict[str, Any]:
    return load_json(LEARN_FILE, {
        "samples": 0,
        "player_bias": {},
        "prop_bias": {},
    })

def save_learning(data: Dict[str, Any]) -> None:
    save_json(LEARN_FILE, data)

def update_clv(row: Dict[str, Any], pick_side: str) -> float:
    data = load_json(CLV_FILE, {})
    key = f"{normalize_name(row['player'])}_{row['prop']}_{pick_side}_{row.get('game','')}"
    latest = float(row["line"])
    if key not in data:
        data[key] = {
            "open": latest,
            "latest": latest,
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
    graded = [
        r for r in results
        if safe_float(r.get("actual")) is not None and safe_float(r.get("line")) is not None
    ]

    player_bias = {}
    prop_bias = {}

    for r in graded[-600:]:
        actual = safe_float(r.get("actual"))
        line = safe_float(r.get("line"))
        if actual is None or line is None:
            continue

        err = clamp(actual - line, -10, 10)
        pkey = normalize_name(r.get("player"))
        prop = r.get("prop")

        player_bias[pkey] = clamp((safe_float(player_bias.get(pkey), 0.0) or 0.0) + err * 0.004, -2.0, 2.0)
        prop_bias[prop] = clamp((safe_float(prop_bias.get(prop), 0.0) or 0.0) + err * 0.002, -1.25, 1.25)

    learning = {
        "samples": len(graded),
        "player_bias": player_bias,
        "prop_bias": prop_bias,
        "updated_at": now_iso(),
    }
    save_learning(learning)
    return len(graded)


# ============================================================
# MODEL
# ============================================================
def bayesian_shrink(prob: float, data_score: int, learning_samples: int, enabled: bool) -> Tuple[float, str]:
    if not enabled:
        return prob, "Bayesian off"

    data_weight = clamp(data_score / 100.0, 0.30, 0.96)
    sample_weight = clamp(learning_samples / 100.0, 0.10, 1.00)
    confidence = clamp((0.70 * data_weight) + (0.30 * sample_weight), 0.22, 0.94)
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

def projection_from_line(
    row: Dict[str, Any],
    use_learning: bool,
    use_markov: bool,
    use_real_stats: bool,
    nba_season: str,
    market_blend: float,
) -> Tuple[float, float, str, float, str, Dict[str, Any]]:
    learning = load_learning()
    line = float(row["line"])
    pkey = normalize_name(row["player"])
    prop = row["prop"]

    real = true_stat_projection(row["player"], prop, line, use_real_stats, nba_season)

    player_bias = safe_float((learning.get("player_bias") or {}).get(pkey), 0.0) or 0.0
    prop_bias = safe_float((learning.get("prop_bias") or {}).get(prop), 0.0) or 0.0
    bias = player_bias + prop_bias if use_learning else 0.0

    # Blend real stat projection with market line only for stability.
    # 0.0 = pure stats, 1.0 = pure market line.
    market_blend = clamp(float(market_blend), 0.0, 0.85)
    projection = (real["projection"] * (1.0 - market_blend)) + (line * market_blend) + bias
    std = float(real["std"])

    markov_state, markov_adj = markov_recent_state(row["player"], prop, line, use_markov)

    diff = projection - line
    note = (
        f"Projection source: {real['source']} | {real['note']} | "
        f"market blend {market_blend:.2f} | bias {bias:+.2f} | diff vs line {diff:+.2f}"
    )
    return projection, std, markov_state, markov_adj, note, real

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
    use_real_stats: bool,
    nba_season: str,
    market_blend: float,
) -> Dict[str, Any]:
    learning = load_learning()
    learning_samples = int(learning.get("samples", 0) or 0)

    projection, std, markov_state, markov_prob_adj, projection_note, real_projection_meta = projection_from_line(
        row, use_learning, use_markov, use_real_stats, nba_season, market_blend
    )

    sims = np.random.normal(loc=projection, scale=max(std, 0.40), size=max(1000, sim_count))
    over_raw = float(np.mean(sims > float(row["line"])))
    under_raw = 1.0 - over_raw

    over_raw = clamp(over_raw + markov_prob_adj, 0.01, 0.99)
    under_raw = 1.0 - over_raw

    side = "OVER" if over_raw >= under_raw else "UNDER"
    raw_pick_prob = max(over_raw, under_raw)

    data_score = 64
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

    edges = edge_metrics(projection, float(row["line"]), side, over_prob, under_prob)
    raw_edge = edges["raw_edge"]
    pick_edge = edges["pick_edge"]
    edge_pct = edges["edge_pct"]
    prob_edge = edges["prob_edge"]

    # Backward-compatible absolute edge plus direction-aware edge.
    edge = abs(raw_edge)
    min_edge = float(PROP_CONFIG[row["prop"]]["min_edge"]) * min_edge_scale

    # Use EV_PRICE proxy because Underdog lines do not expose standard single-pick odds.
    ev = expected_value(pick_prob, EV_PRICE)
    kelly = kelly_fraction(pick_prob, EV_PRICE, max_kelly)
    stake = bankroll * kelly
    clv = update_clv(row, side)

    reasons = []
    if pick_prob < min_prob:
        reasons.append("probability below gate")
    if pick_edge < min_edge:
        reasons.append("directional edge below gate")
    if data_score < min_data_score:
        reasons.append("data score below gate")
    if ev is None or ev < 0:
        reasons.append("EV not positive")

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
        "raw_edge": float(raw_edge),
        "pick_edge": float(pick_edge),
        "edge_pct": float(edge_pct),
        "prob_edge": float(prob_edge),
        "min_edge": float(min_edge),
        "ev": None if ev is None else float(ev),
        "ev_price": float(EV_PRICE),
        "kelly": float(kelly),
        "stake": float(stake),
        "clv": float(clv),
        "data_score": int(data_score),
        "qualified": bool(qualified),
        "signal": signal,
        "reasons": reasons,
        "markov_state": markov_state,
        "model_notes": [projection_note, bayes_note],
        "learning_samples": learning_samples,
        "projection_source": real_projection_meta.get("source"),
        "nba_player_id": real_projection_meta.get("nba_player_id"),
        "nba_match_score": real_projection_meta.get("nba_match_score"),
        "nba_match_name": real_projection_meta.get("nba_match_name"),
        "l3": real_projection_meta.get("l3"),
        "l5": real_projection_meta.get("l5"),
        "l10": real_projection_meta.get("l10"),
        "season_avg": real_projection_meta.get("season_avg"),
        "minutes": real_projection_meta.get("minutes"),
    }

def build_board(
    rows: List[Dict[str, Any]],
    selected_props: List[str],
    player_search: str,
    game_search: str,
    min_display_prob: float,
    show_passes: bool,
    **model_kwargs
) -> List[Dict[str, Any]]:
    filtered = []
    prop_set = set(selected_props or [])

    for r in rows:
        if prop_set and r["prop"] not in prop_set:
            continue
        if player_search and player_search.lower() not in r["player"].lower():
            continue
        if game_search and game_search.lower() not in str(r.get("game", "")).lower():
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
        -x.get("pick_edge", x.get("edge", 0)),
        -x["data_score"],
    ))
    return board


# ============================================================
# OFFICIAL PICKS / GRADING
# ============================================================
def official_pick_id(p: Dict[str, Any]) -> str:
    return (
        f"{datetime.now().date()}_underdog_"
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
            "source": "Underdog",
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
            "raw_edge": round(float(p.get("raw_edge", 0)), 3),
            "pick_edge": round(float(p.get("pick_edge", 0)), 3),
            "edge_pct": round(float(p.get("edge_pct", 0)), 4),
            "prob_edge": round(float(p.get("prob_edge", 0)), 4),
            "ev": None if p["ev"] is None else round(float(p["ev"]), 4),
            "ev_price": p.get("ev_price", EV_PRICE),
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
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 4px 12px 4px;'>
      <div style='font-size:28px;font-weight:950;'>😈 DEVIL PICKS</div>
      <div style='color:#ff344f;font-weight:900;'>UNDERDOG ONLY</div>
      <div style='color:#aeb7c9;font-size:12px;margin-top:4px;'>Real lines • no mixed feeds</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Toggles")
    use_bayesian = st.toggle("Bayesian confidence", value=os.getenv("ENABLE_BAYESIAN", "true").lower() == "true")
    use_markov = st.toggle("Markov recent form", value=os.getenv("ENABLE_MARKOV", "true").lower() == "true")
    use_learning = st.toggle("Use saved learning", value=os.getenv("ENABLE_LEARNING", "true").lower() == "true")
    use_real_stats = st.toggle("Use real NBA game-log projections", value=os.getenv("ENABLE_REAL_STATS", "true").lower() == "true")
    nba_season = st.text_input("NBA season", value=os.getenv("NBA_SEASON", current_nba_season()))
    market_blend = st.slider("Market-line blend", 0.0, 0.85, float(os.getenv("MARKET_BLEND", "0.25")), 0.05)

    st.markdown("### Board Filters")
    selected_props = st.multiselect("Props", list(PROP_CONFIG.keys()), default=list(PROP_CONFIG.keys()))
    player_search = st.text_input("Search player", "")
    game_search = st.text_input("Search game", "")

    st.markdown("### Gates")
    bankroll = st.number_input("Bankroll", min_value=10.0, value=DEFAULT_BANKROLL, step=25.0)
    sim_count = st.slider("Monte Carlo sims", 1000, 30000, DEFAULT_SIM_COUNT, 1000)
    min_prob = st.slider("Official min probability", 0.50, 0.75, DEFAULT_MIN_PROB, 0.01)
    min_display_prob = st.slider("Display min probability", 0.45, 0.75, 0.52, 0.01)
    min_data_score = st.slider("Min data score", 40, 95, DEFAULT_MIN_DATA_SCORE, 1)
    min_edge_scale = st.slider("Edge gate scale", 0.30, 1.50, 0.70, 0.05)
    max_kelly = st.slider("Max Kelly stake", 0.0, 0.10, DEFAULT_MAX_KELLY, 0.005)
    st.caption(f"EV uses price proxy: {odds_display(EV_PRICE)}. Change EV_PRICE in Railway if needed.")
    show_passes = st.toggle("Show PASS rows", value=True)

    st.markdown("### Source")
    strict_nba_filter = st.toggle("Strict NBA-only filter", value=True)
    day_filter = st.radio("Game date filter", ["Today + Tomorrow", "Today", "Tomorrow", "All Future"], index=0)
    st.write("Source: ✅ Underdog only")
    st.write("Endpoint:", UNDERDOG_API_URL[:42] + ("..." if len(UNDERDOG_API_URL) > 42 else ""))


# ============================================================
# MAIN UI
# ============================================================
st.markdown("""
<div class='hero'>
  <div class='logo-title'>😈 DEVIL PICKS — Underdog Only Rail</div>
  <div class='sub'>Real Underdog lines only • Monte Carlo • Bayesian confidence • Markov tracking • save/grade/learn</div>
</div>
""", unsafe_allow_html=True)

perf = performance_summary()
learn = load_learning()

top_cols = st.columns(5)
with top_cols[0]:
    refresh = st.button("🔄 Refresh Underdog Lines", use_container_width=True)
with top_cols[1]:
    save_btn = st.button("💾 Save Official Picks", use_container_width=True)
with top_cols[2]:
    clear_cache = st.button("🧹 Clear Cache", use_container_width=True)
with top_cols[3]:
    rebuild_btn = st.button("🧠 Rebuild Learning", use_container_width=True)
with top_cols[4]:
    export_btn = st.button("📤 Export CSV", use_container_width=True)

if clear_cache:
    st.cache_data.clear()
    st.success("Cache cleared. Click Refresh Underdog Lines.")

if rebuild_btn:
    n = rebuild_learning_from_results()
    st.success(f"Learning rebuilt from {n} graded results.")

if refresh or "underdog_rows" not in st.session_state:
    with st.spinner("Pulling real Underdog lines..."):
        st.session_state["underdog_rows"] = fetch_underdog_lines(UNDERDOG_API_URL, strict_nba_filter, day_filter)
        st.session_state["last_refresh"] = now_iso()

source_rows = st.session_state.get("underdog_rows", [])

board = build_board(
    source_rows,
    selected_props=selected_props,
    player_search=player_search,
    game_search=game_search,
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
    use_real_stats=use_real_stats,
    nba_season=nba_season,
    market_blend=market_blend,
)

if save_btn:
    saved = save_official_picks(board)
    st.success(f"Saved {saved} official Underdog picks.")

if export_btn:
    if board:
        export_df = pd.DataFrame(board).drop(columns=["raw"], errors="ignore")
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download current board CSV", csv, file_name="underdog_board.csv", mime="text/csv")
    else:
        st.warning("No board rows to export.")

qualified = [b for b in board if b.get("qualified")]
strong = [b for b in qualified if "STRONG" in b.get("signal", "")]
prop_counts = pd.Series([r["prop"] for r in source_rows]).value_counts().to_dict() if source_rows else {}

st.markdown(f"""
<div class='metric-grid'>
  <div class='metric-box'><div class='metric-label'>Underdog Rows</div><div class='metric-value'>{len(source_rows)}</div><div class='metric-sub'>{prop_counts}</div></div>
  <div class='metric-box'><div class='metric-label'>Board Rows</div><div class='metric-value'>{len(board)}</div><div class='metric-sub'>After filters</div></div>
  <div class='metric-box'><div class='metric-label'>Official Plays</div><div class='metric-value'>{len(qualified)}</div><div class='metric-sub'>Gate passed</div></div>
  <div class='metric-box'><div class='metric-label'>Strong Plays</div><div class='metric-value'>{len(strong)}</div><div class='metric-sub'>High confidence</div></div>
  <div class='metric-box'><div class='metric-label'>Hit Rate</div><div class='metric-value'>{pct(perf['hit_rate'])}</div><div class='metric-sub'>{perf['official']} official graded</div></div>
</div>
""", unsafe_allow_html=True)

if not source_rows:
    st.markdown("""
    <div class='card card-bad'>
      <div class='big'>No Underdog lines loaded</div>
      <div class='sub'>
        This app does not create fake props. Open Source Logs to see the request status.
        If Railway is blocked or Underdog changes the feed, set UNDERDOG_API_URL in Railway.
        Then click Clear Cache and Refresh Underdog Lines.
      </div>
    </div>
    """, unsafe_allow_html=True)

tabs = st.tabs([
    "😈 Top Plays",
    "📋 Full Board",
    "🔎 Raw Underdog Lines",
    "💾 Official Picks",
    "✅ Grade + Learn",
    "📈 Performance",
    "🔌 Source Logs",
    "⚙️ Railway Setup",
])

with tabs[0]:
    st.markdown("<div class='section-title'>Top Underdog Plays</div>", unsafe_allow_html=True)
    if not board:
        st.info("No board rows after filters.")
    for p in board[:60]:
        card_class = "card card-good" if p["qualified"] else "card card-warn"
        reason_text = ", ".join(p["reasons"]) if p["reasons"] else "gates passed"
        notes = " • ".join(p.get("model_notes", []))
        st.markdown(f"""
        <div class='{card_class}'>
          <div class='big'>{p['player']} — {p['prop_label']}</div>
          <span class='badge badge-blue'>Underdog</span>
          <span class='badge {'badge-green' if p['qualified'] else 'badge-orange'}'>{p['signal']}</span>
          <span class='badge'>Pick: {p['pick_side']}</span>
          <span class='badge'>Line: {p['line']}</span>
          <span class='badge'>State: {p['markov_state']}</span>
          <span class='badge'>Date: {p.get('date_bucket','')}</span>
          <span class='badge'>Proj Src: {p.get('projection_source','')}</span>
          <div class='metric-grid'>
            <div class='metric-box'><div class='metric-label'>Projection</div><div class='metric-value'>{p['projection']:.2f}</div><div class='metric-sub'>Line {p['line']} | Raw {p['raw_edge']:+.2f}</div></div>
            <div class='metric-box'><div class='metric-label'>Edge</div><div class='metric-value'>{p['pick_edge']:+.2f}</div><div class='metric-sub'>Prob edge {p['prob_edge']*100:.1f}% | {p['edge_pct']*100:.1f}%</div></div>
            <div class='metric-box'><div class='metric-label'>Over / Under</div><div class='metric-value'>{p['over_prob']*100:.1f}%</div><div class='metric-sub'>Under {p['under_prob']*100:.1f}%</div></div>
            <div class='metric-box'><div class='metric-label'>Pick Prob</div><div class='metric-value'>{p['pick_prob']*100:.1f}%</div><div class='metric-sub'>Raw {p['raw_pick_prob']*100:.1f}%</div></div>
            <div class='metric-box'><div class='metric-label'>EV / Kelly</div><div class='metric-value'>{(p['ev'] or 0)*100:.1f}%</div><div class='metric-sub'>Proxy {odds_display(p.get('ev_price', EV_PRICE))} | Stake ${p['stake']:.2f}</div></div>
            <div class='metric-box'><div class='metric-label'>Data Score</div><div class='metric-value'>{p['data_score']}</div><div class='metric-sub'>CLV {p['clv']:+.2f}</div></div>
          </div>
          <div class='sub'>Game: {p.get('game','')} | Team: {p.get('team','')} | Notes: {reason_text}</div>
          <div class='sub'>Model: {notes}</div>
        </div>
        """, unsafe_allow_html=True)

with tabs[1]:
    if board:
        display_cols = [
            "signal", "player", "team", "game", "date_bucket", "local_start", "prop", "line", "pick_side",
            "projection", "projection_source", "l3", "l5", "l10", "season_avg", "minutes", "over_prob", "under_prob", "pick_prob", "raw_edge", "pick_edge", "edge_pct", "prob_edge", "min_edge", "ev", "ev_price",
            "kelly", "stake", "clv", "data_score", "markov_state", "reasons",
        ]
        df = pd.DataFrame(board).drop(columns=["raw"], errors="ignore")
        st.dataframe(df[[c for c in display_cols if c in df.columns]], use_container_width=True, hide_index=True)
    else:
        st.info("No board rows.")

with tabs[2]:
    if source_rows:
        raw_df = pd.DataFrame(source_rows).drop(columns=["raw"], errors="ignore")
        st.dataframe(raw_df, use_container_width=True, hide_index=True)
    else:
        st.info("No raw Underdog lines loaded.")

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
            f"{p['player']} {p['prop']} {p['side']} {p['line']} | {p.get('game','')}"
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
        submitted = st.form_submit_button("Add Manual Result + Rebuild Learning")
        if submitted and m_player:
            results = load_json(RESULT_LOG, [])
            side = "OVER" if m_actual > m_line else "UNDER"
            results.append({
                "pick_id": f"manual_{now_iso()}_{normalize_name(m_player)}_{m_prop}",
                "saved_at": now_iso(),
                "source": "Underdog Manual",
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
    ### Railway setup

    This Underdog-only build does not require `ODDS_API_KEY` or `APIFY_TOKEN`.

    Optional environment variables:

    ```text
    UNDERDOG_API_URL=https://api.underdogfantasy.com/beta/v5/over_under_lines
    ENABLE_BAYESIAN=true
    ENABLE_MARKOV=true
    ENABLE_LEARNING=true
    PROP_SIMULATION_COUNT=12000
    MIN_PROB=0.57
    MIN_DATA_SCORE=68
    BANKROLL=1000
    ```

    Railway start command is handled by the included `Procfile`.

    If props do not show:
    1. Click **Clear Cache**
    2. Click **Refresh Underdog Lines**
    3. Open **Source Logs**
    4. If the endpoint is blocked or changed, update `UNDERDOG_API_URL`
    """)

