# -*- coding: utf-8 -*-
# ============================================================
# DEVIL PICKS — UNDERDOG NBA + WNBA TRUE PROJECTIONS
# Railway / GitHub / Streamlit ready
#
# Real Underdog feed only.
# Separate NBA and WNBA boards.
# Both boards use verified stat-log projections.
# Projection=line fallback rows are NOT eligible as plays.
# ============================================================

import os
import re
import json
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
    page_title="Devil Picks — Underdog NBA/WNBA",
    page_icon="😈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

PICK_LOG = os.path.join(DATA_DIR, "official_picks.json")
RESULT_LOG = os.path.join(DATA_DIR, "graded_results.json")
LEARN_FILE = os.path.join(DATA_DIR, "learning_state.json")
CLV_FILE = os.path.join(DATA_DIR, "clv_tracker.json")
REQUEST_LOG_FILE = os.path.join(DATA_DIR, "source_logs.json")
SNAPSHOT_FILE = os.path.join(DATA_DIR, "latest_underdog_snapshot.json")

UNDERDOG_API_URL = os.getenv(
    "UNDERDOG_API_URL",
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
).strip()

NBA_STATS_BASE = "https://stats.nba.com/stats"
WNBA_STATS_BASE = "https://stats.wnba.com/stats"

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/Los_Angeles")
DEFAULT_PRICE = float(os.getenv("DEFAULT_PRICE", "-110"))
EV_PRICE = float(os.getenv("EV_PRICE", os.getenv("DEFAULT_PRICE", "-110")))
DEFAULT_SIM_COUNT = int(os.getenv("PROP_SIMULATION_COUNT", "12000"))
DEFAULT_MIN_PROB = float(os.getenv("MIN_PROB", "0.57"))
DEFAULT_MIN_DATA_SCORE = int(os.getenv("MIN_DATA_SCORE", "68"))
DEFAULT_MAX_KELLY = float(os.getenv("MAX_KELLY", "0.03"))
DEFAULT_BANKROLL = float(os.getenv("BANKROLL", "1000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "120"))

NBA_SEASON_ENV = os.getenv("NBA_SEASON", "")
WNBA_SEASON_ENV = os.getenv("WNBA_SEASON", "")

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

NBA_TEAM_NAMES = {
    "hawks","celtics","nets","hornets","bulls","cavaliers","mavericks","nuggets",
    "pistons","warriors","rockets","pacers","clippers","lakers","grizzlies","heat",
    "bucks","timberwolves","pelicans","knicks","thunder","magic","76ers","sixers","suns",
    "trail blazers","blazers","kings","spurs","raptors","jazz","wizards"
}
WNBA_TEAM_NAMES = {
    "dream","sky","sun","wings","valkyries","fever","sparks","aces",
    "lynx","liberty","mercury","storm","mystics"
}


# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
.stApp {
  background: radial-gradient(circle at top left,#2b000d 0%,#070b13 42%,#02040a 100%);
  color:#f7f8fb;
}
.block-container {max-width:1720px; padding-top:1rem; padding-bottom:3rem;}
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#050912,#02040a);
  border-right:1px solid rgba(255,52,79,.22);
}
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
.big {font-size:24px; font-weight:950;}
.section-title {font-size:22px; font-weight:950; margin:18px 0 10px; border-left:5px solid #ff344f; padding-left:12px;}
.stButton button {border-radius:14px; font-weight:900; border:1px solid rgba(255,255,255,.18);}
.stTabs [data-baseweb="tab"] {color:#b8c3cf; font-weight:900;}
.stTabs [aria-selected="true"] {color:#ff344f!important; border-bottom:3px solid #ff344f;}
@media (max-width: 1100px) {.metric-grid,.metric-grid-4 {grid-template-columns:repeat(2,minmax(0,1fr));}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# STORAGE / LOGGING
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
    rows.append({"time": now_iso(), "source": str(source)[:220], "status": str(status)[:90], "message": str(message)[:700]})
    save_json(REQUEST_LOG_FILE, rows[-1200:])


# ============================================================
# BASIC HELPERS
# ============================================================
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
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(local_tz())
    except Exception:
        pass
    try:
        x = float(s)
        if x > 10_000_000_000:
            x /= 1000.0
        return datetime.fromtimestamp(x, tz=timezone.utc).astimezone(local_tz())
    except Exception:
        return None

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
    return "N/A" if x is None else f"{x * 100:.1f}%"

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

def game_matches_day_filter(start_value: Any, day_filter: str) -> bool:
    bucket = date_bucket_for_start(start_value)
    if bucket == "Past":
        return False
    dt = parse_datetime_any(start_value)
    if dt and datetime.now(local_tz()) > dt + timedelta(hours=4):
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

def current_nba_season() -> str:
    now = datetime.now(local_tz())
    start = now.year - 1 if now.month < 7 else now.year
    return f"{start}-{str(start + 1)[-2:]}"

def current_wnba_season() -> str:
    now = datetime.now(local_tz())
    return str(now.year)

def first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d.get(k) not in [None, ""]:
            return d.get(k)
    return default


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

def safe_get_stats_json(base_url: str, endpoint: str, params: Optional[dict] = None, timeout: int = 25) -> Any:
    url = f"{base_url}/{endpoint}"
    try:
        host = "stats.wnba.com" if "wnba" in base_url else "stats.nba.com"
        referer = "https://www.wnba.com/" if "wnba" in base_url else "https://www.nba.com/"
        headers = {
            "Host": host,
            "Connection": "keep-alive",
            "Accept": "application/json, text/plain, */*",
            "x-nba-stats-token": "true",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "x-nba-stats-origin": "stats",
            "Origin": referer.rstrip("/"),
            "Referer": referer,
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, params=params or {}, headers=headers, timeout=timeout)
        if r.status_code != 200:
            log_request(url, f"STATS_HTTP {r.status_code}", r.text[:400])
            return None
        return r.json()
    except Exception as e:
        log_request(url, "STATS_REQUEST_ERROR", str(e))
        return None


# ============================================================
# UNDERDOG PARSER
# ============================================================
def classify_underdog_league(player: Dict[str, Any], app: Dict[str, Any], game_obj: Dict[str, Any], team: str, opponent: str, game: str) -> str:
    fields = []
    for obj in [player, app, game_obj]:
        if isinstance(obj, dict):
            for k in [
                "league", "league_id", "league_name", "league_abbreviation", "organization",
                "organization_name", "competition", "competition_name", "sport_league",
                "title", "game_title", "match_title"
            ]:
                v = obj.get(k)
                if v is not None:
                    fields.append(str(v).lower())
    text = " ".join(fields + [str(game).lower(), str(team).lower(), str(opponent).lower()])
    if re.search(r"\bwnba\b", text) or any(name in text for name in WNBA_TEAM_NAMES):
        return "WNBA"
    if re.search(r"\bnba\b", text) or any(name in text for name in NBA_TEAM_NAMES):
        return "NBA"
    return "OTHER"

def player_name_from_player_obj(player: Dict[str, Any], app: Dict[str, Any]) -> str:
    direct = first_present(player, ["full_name", "fullName", "display_name", "displayName", "name", "player_name", "title"])
    if direct:
        return str(direct).strip()
    first = first_present(player, ["first_name", "firstName", "first", "given_name"], "")
    last = first_present(player, ["last_name", "lastName", "last", "family_name"], "")
    combo = f"{first} {last}".strip()
    if combo:
        return combo
    return str(first_present(app, ["player_name", "name", "display_name", "displayName"], "") or "").strip()

def stat_name_from_underdog(line_item: Dict[str, Any], ou: Dict[str, Any], app_stat: Dict[str, Any]) -> Any:
    return (
        first_present(app_stat, ["stat", "display_stat", "stat_type", "type"])
        or first_present(ou, ["stat", "stat_type", "display_stat", "title"])
        or first_present(line_item, ["stat", "stat_type", "display_stat", "title"])
    )

def line_value_from_underdog(line_item: Dict[str, Any], ou: Dict[str, Any], app_stat: Dict[str, Any]) -> Any:
    preferred = first_present(ou, ["stat_value", "display_stat_value", "line", "value", "over_under_value", "value_text"])
    if preferred not in [None, ""]:
        return preferred
    second = first_present(line_item, ["stat_value", "display_stat_value", "line", "value", "over_under_value"])
    if second not in [None, ""]:
        return second
    opts = line_item.get("options")
    vals = []
    if isinstance(opts, list):
        for opt in opts:
            if isinstance(opt, dict):
                fv = safe_float(first_present(opt, ["stat_value", "display_stat_value", "line", "value"]))
                if fv is not None:
                    vals.append(fv)
    if vals:
        return float(np.median(vals))
    return first_present(app_stat, ["line", "value", "stat_value"])

def status_is_open_line(line_item: Dict[str, Any], ou: Dict[str, Any], app: Dict[str, Any], game_obj: Dict[str, Any]) -> bool:
    joined = " ".join(
        str(first_present(obj, ["status", "state", "display_status", "live_event_status", "status_text", "result_status"], "")).lower()
        for obj in [line_item, ou, app, game_obj]
        if isinstance(obj, dict)
    )
    bad = ["settled", "closed", "removed", "cancelled", "canceled", "suspended", "graded", "resulted", "inactive"]
    return not any(x in joined for x in bad)

def normalize_prop_row(
    player: Any, prop: Any, line: Any, league: str, team: Any = "", opponent: Any = "",
    game: Any = "", start_time: Any = "", price: Any = DEFAULT_PRICE, raw: Optional[dict] = None,
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
        "league": league,
        "player": player,
        "team": str(team or "").strip(),
        "opponent": str(opponent or "").strip(),
        "game": str(game or "").strip(),
        "start_time": str(start_time or "").strip(),
        "date_bucket": date_bucket_for_start(start_time),
        "local_start": str(parse_datetime_any(start_time) or ""),
        "prop": prop_key,
        "prop_label": PROP_CONFIG[prop_key]["label"],
        "line": float(line),
        "price": float(safe_float(price, DEFAULT_PRICE)),
        "raw": raw or {},
    }

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_underdog_lines(url: str, day_filter: str = "Today + Tomorrow", sport_filter: str = "NBA + WNBA") -> List[Dict[str, Any]]:
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
    skipped = {"no_join": 0, "not_target_league": 0, "bad_prop": 0, "bad_line": 0, "bad_date": 0, "closed": 0}

    if isinstance(data, dict) and isinstance(data.get("players"), list) and isinstance(data.get("appearances"), list):
        players = {str(p.get("id")): p for p in data.get("players", []) if isinstance(p, dict) and p.get("id") is not None}
        appearances = {str(a.get("id")): a for a in data.get("appearances", []) if isinstance(a, dict) and a.get("id") is not None}
        games_raw = data.get("games", []) or data.get("matches", []) or []
        games = {str(g.get("id")): g for g in games_raw if isinstance(g, dict) and g.get("id") is not None}

        for line_item in data.get("over_under_lines", []) or []:
            if not isinstance(line_item, dict):
                continue
            ou = line_item.get("over_under") if isinstance(line_item.get("over_under"), dict) else {}
            app_stat = ou.get("appearance_stat") if isinstance(ou.get("appearance_stat"), dict) else {}

            appearance_id = (
                first_present(app_stat, ["appearance_id", "appearanceId"])
                or first_present(ou, ["appearance_id", "appearanceId"])
                or first_present(line_item, ["appearance_id", "appearanceId"])
            )
            app = appearances.get(str(appearance_id), {}) if appearance_id is not None else {}
            player_id = first_present(app, ["player_id", "playerId"]) or first_present(line_item, ["player_id", "playerId"])
            player = players.get(str(player_id), {}) if player_id is not None else {}
            if not app or not player:
                skipped["no_join"] += 1
                continue

            stat_name = stat_name_from_underdog(line_item, ou, app_stat)
            if clean_prop_type(stat_name) not in PROP_CONFIG:
                skipped["bad_prop"] += 1
                continue

            line_value = line_value_from_underdog(line_item, ou, app_stat)
            if safe_float(line_value) is None:
                skipped["bad_line"] += 1
                continue

            team = first_present(app, ["team_abbr", "teamAbbr", "team"]) or first_present(player, ["team_abbr", "teamAbbr", "team"], "")
            opponent = first_present(app, ["opponent_abbr", "opponentAbbr", "opponent"], "")
            game_id = first_present(app, ["game_id", "gameId", "match_id", "matchId"]) or first_present(line_item, ["game_id", "gameId", "match_id", "matchId"])
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

            league = classify_underdog_league(player, app, game_obj, team, opponent, game)
            allowed = ["NBA", "WNBA"] if sport_filter == "NBA + WNBA" else [sport_filter]
            if league not in allowed:
                skipped["not_target_league"] += 1
                continue
            if not status_is_open_line(line_item, ou, app, game_obj):
                skipped["closed"] += 1
                continue
            if not game_matches_day_filter(start_time, day_filter):
                skipped["bad_date"] += 1
                continue

            prices = []
            for opt in line_item.get("options", []) or []:
                if isinstance(opt, dict):
                    for key in ["american_odds", "americanOdds", "odds", "price"]:
                        val = safe_float(opt.get(key))
                        if val is not None:
                            prices.append(val)
            option_price = float(np.median(prices)) if prices else DEFAULT_PRICE

            row = normalize_prop_row(
                player=player_name_from_player_obj(player, app),
                prop=stat_name,
                line=line_value,
                league=league,
                team=team,
                opponent=opponent,
                game=game,
                start_time=start_time,
                price=option_price,
                raw={"line_id": line_item.get("id"), "appearance_id": appearance_id, "player_id": player_id, "stat_name": stat_name, "game_id": game_id},
            )
            if row:
                row["appearance_id"] = str(appearance_id)
                row["underdog_player_id"] = str(player_id)
                rows.append(row)

    log_request("Underdog", "OK", f"{len(rows)} rows from {used_url}; skipped={skipped}")
    save_json(SNAPSHOT_FILE, {"created_at": now_iso(), "rows": rows, "skipped": skipped, "url": used_url})
    return rows


# ============================================================
# STATS PROJECTIONS: NBA + WNBA
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_player_index(league: str, season: str) -> List[Dict[str, Any]]:
    base = NBA_STATS_BASE if league == "NBA" else WNBA_STATS_BASE
    league_id = "00" if league == "NBA" else "10"

    # commonallplayers tends to be more stable across NBA/WNBA than playerindex.
    params = {
        "IsOnlyCurrentSeason": "1",
        "LeagueID": league_id,
        "Season": season,
    }
    data = safe_get_stats_json(base, "commonallplayers", params=params, timeout=25)
    rows = []
    try:
        rs = data.get("resultSets", [])[0]
        headers = rs.get("headers", [])
        rows = [dict(zip(headers, r)) for r in rs.get("rowSet", [])]
        if rows:
            return rows
    except Exception as e:
        log_request(f"{league}_COMMONALLPLAYERS", "PARSE_ERROR", str(e))

    # fallback playerindex
    params2 = {
        "College": "", "Country": "", "DraftPick": "", "DraftRound": "", "DraftYear": "",
        "Height": "", "Historical": "0", "LeagueID": league_id, "Season": season,
        "SeasonType": "Regular Season", "TeamID": "0", "Weight": "",
    }
    data2 = safe_get_stats_json(base, "playerindex", params=params2, timeout=25)
    try:
        rs = data2.get("resultSets", [])[0]
        headers = rs.get("headers", [])
        return [dict(zip(headers, r)) for r in rs.get("rowSet", [])]
    except Exception as e:
        log_request(f"{league}_PLAYERINDEX", "PARSE_ERROR", str(e))
        return []

def player_display_name(p: Dict[str, Any]) -> str:
    return str(
        p.get("DISPLAY_FIRST_LAST")
        or p.get("PLAYER_NAME")
        or p.get("PLAYER")
        or p.get("PERSON_NAME")
        or p.get("ROSTER_PLAYER_NAME")
        or p.get("PLAYER_SLUG")
        or ""
    )

def player_id_from_index_row(p: Dict[str, Any]) -> Optional[int]:
    return safe_int(
        p.get("PERSON_ID")
        or p.get("PLAYER_ID")
        or p.get("PlayerID")
        or p.get("person_id")
    )

def best_player_match(name: str, league: str, season: str) -> Tuple[Optional[int], float, str]:
    players = get_player_index(league, season)
    target = normalize_name(name)
    best_id, best_score, best_name = None, 0.0, ""
    for p in players:
        cand = player_display_name(p)
        if not cand:
            continue
        score = difflib.SequenceMatcher(None, target, normalize_name(cand)).ratio()
        if target and (target in normalize_name(cand) or normalize_name(cand) in target):
            score = max(score, 0.94)
        if score > best_score:
            best_score = score
            best_id = player_id_from_index_row(p)
            best_name = cand
    return best_id, float(best_score), best_name

@st.cache_data(ttl=1800, show_spinner=False)
def get_player_gamelog(player_id: int, league: str, season: str) -> pd.DataFrame:
    base = NBA_STATS_BASE if league == "NBA" else WNBA_STATS_BASE
    league_id = "00" if league == "NBA" else "10"
    params = {
        "PlayerID": str(player_id),
        "Season": season,
        "SeasonType": "Regular Season",
        "LeagueID": league_id,
    }
    data = safe_get_stats_json(base, "playergamelog", params=params, timeout=25)
    try:
        rs = data.get("resultSets", [])[0]
        headers = rs.get("headers", [])
        df = pd.DataFrame(rs.get("rowSet", []), columns=headers)
        for c in ["PTS", "REB", "AST", "FG3M", "BLK", "STL", "TOV", "MIN"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        log_request(f"{league}_PLAYERGAMELOG", "PARSE_ERROR", str(e))
        return pd.DataFrame()

def stat_prop_value(row: Dict[str, Any], prop: str) -> float:
    def sf(k): return safe_float(row.get(k), 0.0) or 0.0
    if prop == "PTS": return sf("PTS")
    if prop == "REB": return sf("REB")
    if prop == "AST": return sf("AST")
    if prop == "PRA": return sf("PTS") + sf("REB") + sf("AST")
    if prop == "PR": return sf("PTS") + sf("REB")
    if prop == "PA": return sf("PTS") + sf("AST")
    if prop == "RA": return sf("REB") + sf("AST")
    if prop == "3PM": return sf("FG3M")
    if prop == "BLK": return sf("BLK")
    if prop == "STL": return sf("STL")
    if prop == "TO": return sf("TOV")
    return 0.0

def true_stat_projection(player_name: str, prop: str, line: float, league: str, use_real_stats: bool, season: str) -> Dict[str, Any]:
    fallback = {
        "projection": float(line),
        "std": float(PROP_CONFIG[prop].get("std", 3.0)),
        "source": "market_fallback",
        "note": f"No verified {league} stat projection; using line fallback",
        "stats_player_id": None,
        "match_score": 0.0,
        "match_name": "",
        "l3": None, "l5": None, "l10": None, "season_avg": None, "minutes": None,
    }
    if not use_real_stats:
        fallback["note"] = "Real stats toggle off; using line fallback"
        return fallback

    pid, score, match_name = best_player_match(player_name, league, season)
    fallback["stats_player_id"] = pid
    fallback["match_score"] = score
    fallback["match_name"] = match_name
    if not pid or score < 0.78:
        fallback["note"] = f"{league} player match not strong enough ({score:.2f}); using line fallback"
        return fallback

    logs = get_player_gamelog(pid, league, season)
    if logs.empty:
        fallback["note"] = f"{league} game logs empty; using line fallback"
        return fallback

    values = [stat_prop_value(r.to_dict(), prop) for _, r in logs.head(15).iterrows()]
    values = [float(v) for v in values if v is not None]
    if not values:
        fallback["note"] = f"{league} values empty; using line fallback"
        return fallback

    l3 = float(np.mean(values[:3])) if len(values) >= 3 else float(np.mean(values))
    l5 = float(np.mean(values[:5])) if len(values) >= 5 else float(np.mean(values))
    l10 = float(np.mean(values[:10])) if len(values) >= 10 else float(np.mean(values))
    season_avg = float(np.mean(values))
    minutes = float(pd.to_numeric(logs.head(5)["MIN"], errors="coerce").mean()) if "MIN" in logs.columns else None

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
        "source": f"{league.lower()}_gamelog",
        "note": f"{league} logs: L3 {l3:.2f}, L5 {l5:.2f}, L10 {l10:.2f}, season {season_avg:.2f}",
        "stats_player_id": pid,
        "match_score": score,
        "match_name": match_name,
        "l3": l3, "l5": l5, "l10": l10, "season_avg": season_avg, "minutes": minutes,
    }


# ============================================================
# LEARNING / CLV / MODEL
# ============================================================
def load_learning() -> Dict[str, Any]:
    return load_json(LEARN_FILE, {"samples": 0, "player_bias": {}, "prop_bias": {}})

def rebuild_learning_from_results() -> int:
    results = load_json(RESULT_LOG, [])
    graded = [r for r in results if safe_float(r.get("actual")) is not None and safe_float(r.get("line")) is not None]
    player_bias, prop_bias = {}, {}
    for r in graded[-600:]:
        actual, line = safe_float(r.get("actual")), safe_float(r.get("line"))
        if actual is None or line is None:
            continue
        err = clamp(actual - line, -10, 10)
        pkey = normalize_name(f"{r.get('league','')}_{r.get('player','')}")
        prop = r.get("prop")
        player_bias[pkey] = clamp((safe_float(player_bias.get(pkey), 0.0) or 0.0) + err * 0.004, -2.0, 2.0)
        prop_key = f"{r.get('league','')}_{prop}"
        prop_bias[prop_key] = clamp((safe_float(prop_bias.get(prop_key), 0.0) or 0.0) + err * 0.002, -1.25, 1.25)
    save_json(LEARN_FILE, {"samples": len(graded), "player_bias": player_bias, "prop_bias": prop_bias, "updated_at": now_iso()})
    return len(graded)

def update_clv(row: Dict[str, Any], pick_side: str) -> float:
    data = load_json(CLV_FILE, {})
    key = f"{row.get('league')}_{normalize_name(row['player'])}_{row['prop']}_{pick_side}_{row.get('game','')}"
    latest = float(row["line"])
    if key not in data:
        data[key] = {"open": latest, "latest": latest, "player": row["player"], "prop": row["prop"], "side": pick_side, "created_at": now_iso(), "updated_at": now_iso()}
        save_json(CLV_FILE, data)
        return 0.0
    open_val = safe_float(data[key].get("open"), latest) or latest
    data[key]["latest"] = latest
    data[key]["updated_at"] = now_iso()
    save_json(CLV_FILE, data)
    return round(latest - open_val, 3)

def bayesian_shrink(prob: float, data_score: int, learning_samples: int, enabled: bool) -> Tuple[float, str]:
    if not enabled:
        return prob, "Bayesian off"
    confidence = clamp((0.70 * clamp(data_score / 100.0, 0.30, 0.96)) + (0.30 * clamp(learning_samples / 100.0, 0.10, 1.00)), 0.22, 0.94)
    adjusted = (prob * confidence) + (0.50 * (1 - confidence))
    return float(clamp(adjusted, 0.01, 0.99)), f"Bayesian confidence {confidence:.2f}"

def edge_metrics(projection: float, line: float, side: str, over_prob: float, under_prob: float) -> Dict[str, float]:
    raw_edge = float(projection) - float(line)
    pick_edge = raw_edge if side == "OVER" else -raw_edge
    edge_pct = pick_edge / (abs(float(line)) if abs(float(line)) > 0 else 1.0)
    prob_edge = max(float(over_prob), float(under_prob)) - 0.50
    return {"raw_edge": raw_edge, "pick_edge": pick_edge, "edge_pct": edge_pct, "prob_edge": prob_edge}

def model_one_prop(
    row: Dict[str, Any],
    sim_count: int,
    min_prob: float,
    min_data_score: int,
    min_edge_scale: float,
    max_kelly: float,
    bankroll: float,
    use_bayesian: bool,
    use_learning: bool,
    use_real_stats: bool,
    require_real_projection: bool,
    nba_season: str,
    wnba_season: str,
    market_blend: float,
) -> Dict[str, Any]:
    league = row.get("league", "NBA")
    season = nba_season if league == "NBA" else wnba_season
    learning = load_learning()
    learning_samples = int(learning.get("samples", 0) or 0)
    real = true_stat_projection(row["player"], row["prop"], row["line"], league, use_real_stats, season)
    verified_projection = real["source"] in ["nba_gamelog", "wnba_gamelog"]

    pkey = normalize_name(f"{league}_{row['player']}")
    prop_key = f"{league}_{row['prop']}"
    player_bias = safe_float((learning.get("player_bias") or {}).get(pkey), 0.0) or 0.0
    prop_bias = safe_float((learning.get("prop_bias") or {}).get(prop_key), 0.0) or 0.0
    bias = player_bias + prop_bias if use_learning else 0.0

    blend = clamp(float(market_blend), 0.0, 0.85)
    projection = (real["projection"] * (1 - blend)) + (float(row["line"]) * blend) + bias
    std = float(real["std"])

    sims = np.random.normal(loc=projection, scale=max(std, 0.40), size=max(1000, sim_count))
    over_raw = float(np.mean(sims > float(row["line"])))
    under_raw = 1.0 - over_raw
    side = "OVER" if over_raw >= under_raw else "UNDER"
    raw_pick_prob = max(over_raw, under_raw)

    data_score = 62
    if verified_projection:
        data_score += 16
    if row.get("player"):
        data_score += 5
    if row.get("game") or row.get("team"):
        data_score += 4
    if use_bayesian:
        data_score += 4
    if use_learning and learning_samples >= 20:
        data_score += 4
    data_score = int(clamp(data_score, 0, 100))

    adj_prob, bayes_note = bayesian_shrink(raw_pick_prob, data_score, learning_samples, use_bayesian)
    if side == "OVER":
        over_prob, under_prob = adj_prob, 1 - adj_prob
    else:
        under_prob, over_prob = adj_prob, 1 - adj_prob
    pick_prob = max(over_prob, under_prob)

    edges = edge_metrics(projection, row["line"], side, over_prob, under_prob)
    min_edge = float(PROP_CONFIG[row["prop"]]["min_edge"]) * min_edge_scale
    ev = expected_value(pick_prob, EV_PRICE)
    kelly = kelly_fraction(pick_prob, EV_PRICE, max_kelly)
    stake = bankroll * kelly
    clv = update_clv(row, side)

    reasons = []
    if require_real_projection and not verified_projection:
        reasons.append(f"no verified {league} stat projection")
    if pick_prob < min_prob:
        reasons.append("probability below gate")
    if edges["pick_edge"] < min_edge:
        reasons.append("directional edge below gate")
    if data_score < min_data_score:
        reasons.append("data score below gate")
    if ev is None or ev < 0:
        reasons.append("EV not positive")

    qualified = len(reasons) == 0
    if qualified and pick_prob >= 0.64 and edges["pick_edge"] >= min_edge * 1.25:
        signal = f"😈 STRONG {side}"
    elif qualified:
        signal = f"✅ LEAN {side}"
    else:
        signal = "PASS"

    return {
        **row,
        "verified_projection": bool(verified_projection),
        "projection_source": real["source"],
        "stats_player_id": real.get("stats_player_id"),
        "match_score": real.get("match_score"),
        "match_name": real.get("match_name"),
        "l3": real.get("l3"), "l5": real.get("l5"), "l10": real.get("l10"), "season_avg": real.get("season_avg"), "minutes": real.get("minutes"),
        "projection": float(projection),
        "std": std,
        "over_prob": float(over_prob),
        "under_prob": float(under_prob),
        "pick_side": side,
        "pick_prob": float(pick_prob),
        "raw_pick_prob": float(raw_pick_prob),
        "edge": abs(float(edges["raw_edge"])),
        "raw_edge": float(edges["raw_edge"]),
        "pick_edge": float(edges["pick_edge"]),
        "edge_pct": float(edges["edge_pct"]),
        "prob_edge": float(edges["prob_edge"]),
        "min_edge": float(min_edge),
        "ev": None if ev is None else float(ev),
        "ev_price": float(EV_PRICE),
        "kelly": float(kelly),
        "stake": float(stake),
        "clv": float(clv),
        "data_score": data_score,
        "qualified": qualified,
        "signal": signal,
        "reasons": reasons,
        "model_notes": [
            f"{real['note']}",
            f"market blend {blend:.2f} | bias {bias:+.2f}",
            bayes_note,
        ],
    }

def build_board(rows: List[Dict[str, Any]], selected_props: List[str], player_search: str, game_search: str, show_passes: bool, min_display_prob: float, **model_kwargs) -> List[Dict[str, Any]]:
    filtered = []
    props = set(selected_props or [])
    for r in rows:
        if props and r["prop"] not in props:
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
        board = [b for b in board if b["qualified"] or b["pick_prob"] >= min_display_prob or b["signal"] == "PASS"]
    board.sort(key=lambda x: (not x["qualified"], -x.get("pick_edge", 0), -(x["ev"] if x["ev"] is not None else -9), -x["pick_prob"], -x["data_score"]))
    return board


# ============================================================
# SAVE / GRADE
# ============================================================
def official_pick_id(p: Dict[str, Any]) -> str:
    return f"{datetime.now().date()}_{p.get('league')}_underdog_{normalize_name(p['player'])}_{p['prop']}_{p['pick_side']}_{round(float(p['line']), 2)}_{normalize_name(p.get('game',''))}"

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
            "pick_id": pid, "saved_at": now_iso(), "source": "Underdog",
            "league": p.get("league"), "player": p["player"], "team": p.get("team", ""),
            "game": p.get("game", ""), "prop": p["prop"], "prop_label": p["prop_label"],
            "line": p["line"], "side": p["pick_side"], "price": p["price"],
            "projection": round(float(p["projection"]), 3), "projection_source": p.get("projection_source"),
            "pick_prob": round(float(p["pick_prob"]), 4), "pick_edge": round(float(p.get("pick_edge", 0)), 3),
            "ev": None if p["ev"] is None else round(float(p["ev"]), 4),
            "kelly": round(float(p["kelly"]), 4), "stake": round(float(p["stake"]), 2),
            "signal": p["signal"], "data_score": p["data_score"],
            "qualified": p["qualified"], "graded": False,
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
    return {"graded": len(graded), "official": len(official), "wins": wins, "hit_rate": wins / len(official) if official else None}


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 4px 12px 4px;'>
      <div style='font-size:28px;font-weight:950;'>😈 DEVIL PICKS</div>
      <div style='color:#ff344f;font-weight:900;'>NBA + WNBA</div>
      <div style='color:#aeb7c9;font-size:12px;margin-top:4px;'>Underdog only • verified projections</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### League / Date")
    sport_filter = st.radio("Sport board", ["NBA + WNBA", "NBA", "WNBA"], index=0)
    day_filter = st.radio("Game date filter", ["Today + Tomorrow", "Today", "Tomorrow", "All Future"], index=0)

    st.markdown("### Projection Controls")
    use_real_stats = st.toggle("Use true stat-log projections", value=os.getenv("ENABLE_REAL_STATS", "true").lower() == "true")
    require_real_projection = st.toggle("Require verified projection", value=os.getenv("REQUIRE_REAL_PROJECTION", "true").lower() == "true")
    nba_season = st.text_input("NBA season", value=NBA_SEASON_ENV or current_nba_season())
    wnba_season = st.text_input("WNBA season", value=WNBA_SEASON_ENV or current_wnba_season())
    market_blend = st.slider("Market-line blend", 0.0, 0.85, float(os.getenv("MARKET_BLEND", "0.15")), 0.05)

    st.markdown("### Model")
    use_bayesian = st.toggle("Bayesian confidence", value=os.getenv("ENABLE_BAYESIAN", "true").lower() == "true")
    use_learning = st.toggle("Use saved learning", value=os.getenv("ENABLE_LEARNING", "true").lower() == "true")

    st.markdown("### Filters")
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
    st.caption(f"EV uses price proxy: {odds_display(EV_PRICE)}")
    show_passes = st.toggle("Show PASS rows", value=True)


# ============================================================
# MAIN
# ============================================================
st.markdown("""
<div class='hero'>
  <div class='logo-title'>😈 DEVIL PICKS — Underdog NBA + WNBA True Projections</div>
  <div class='sub'>Separate NBA/WNBA boards • real Underdog lines • verified stat-log projections • edge/EV/Kelly • save/grade/learn</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
refresh = c1.button("🔄 Refresh Lines", use_container_width=True)
save_btn = c2.button("💾 Save Official Picks", use_container_width=True)
clear_cache = c3.button("🧹 Clear Cache", use_container_width=True)
rebuild_btn = c4.button("🧠 Rebuild Learning", use_container_width=True)
export_btn = c5.button("📤 Export CSV", use_container_width=True)

if clear_cache:
    st.cache_data.clear()
    st.success("Cache cleared. Click Refresh Lines.")

if rebuild_btn:
    n = rebuild_learning_from_results()
    st.success(f"Learning rebuilt from {n} graded results.")

if refresh or "underdog_rows" not in st.session_state:
    with st.spinner("Pulling real Underdog NBA/WNBA lines..."):
        st.session_state["underdog_rows"] = fetch_underdog_lines(UNDERDOG_API_URL, day_filter, sport_filter)
        st.session_state["last_refresh"] = now_iso()

source_rows = st.session_state.get("underdog_rows", [])

board = build_board(
    source_rows,
    selected_props=selected_props,
    player_search=player_search,
    game_search=game_search,
    show_passes=show_passes,
    min_display_prob=min_display_prob,
    sim_count=sim_count,
    min_prob=min_prob,
    min_data_score=min_data_score,
    min_edge_scale=min_edge_scale,
    max_kelly=max_kelly,
    bankroll=bankroll,
    use_bayesian=use_bayesian,
    use_learning=use_learning,
    use_real_stats=use_real_stats,
    require_real_projection=require_real_projection,
    nba_season=nba_season,
    wnba_season=wnba_season,
    market_blend=market_blend,
)

if save_btn:
    saved = save_official_picks(board)
    st.success(f"Saved {saved} official picks.")

if export_btn:
    if board:
        df = pd.DataFrame(board).drop(columns=["raw"], errors="ignore")
        st.download_button("Download current board CSV", df.to_csv(index=False).encode("utf-8"), file_name="underdog_nba_wnba_board.csv", mime="text/csv")
    else:
        st.warning("No board rows to export.")

nba_board = [b for b in board if b.get("league") == "NBA"]
wnba_board = [b for b in board if b.get("league") == "WNBA"]
qualified = [b for b in board if b.get("qualified")]
strong = [b for b in qualified if "STRONG" in b.get("signal", "")]
fallback_count = sum(1 for b in board if not b.get("verified_projection"))
perf = performance_summary()

league_counts = pd.Series([r.get("league", "UNK") for r in source_rows]).value_counts().to_dict() if source_rows else {}
st.markdown(f"""
<div class='metric-grid'>
  <div class='metric-box'><div class='metric-label'>Source Rows</div><div class='metric-value'>{len(source_rows)}</div><div class='metric-sub'>{league_counts}</div></div>
  <div class='metric-box'><div class='metric-label'>Board Rows</div><div class='metric-value'>{len(board)}</div><div class='metric-sub'>Fallback rows visible {fallback_count}</div></div>
  <div class='metric-box'><div class='metric-label'>Official Plays</div><div class='metric-value'>{len(qualified)}</div><div class='metric-sub'>Gate passed</div></div>
  <div class='metric-box'><div class='metric-label'>Strong Plays</div><div class='metric-value'>{len(strong)}</div><div class='metric-sub'>High confidence</div></div>
  <div class='metric-box'><div class='metric-label'>Hit Rate</div><div class='metric-value'>{pct(perf['hit_rate'])}</div><div class='metric-sub'>{perf['official']} official graded</div></div>
</div>
""", unsafe_allow_html=True)

if not source_rows:
    st.markdown("""
    <div class='card card-bad'>
      <div class='big'>No Underdog lines loaded</div>
      <div class='sub'>This app does not create fake props. Open Source Logs to see the endpoint status.</div>
    </div>
    """, unsafe_allow_html=True)

def render_cards(rows: List[Dict[str, Any]], title: str):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if not rows:
        st.info("No rows for this board after filters.")
    for p in rows[:70]:
        card_class = "card card-good" if p["qualified"] else "card card-warn"
        reason_text = ", ".join(p["reasons"]) if p["reasons"] else "gates passed"
        notes = " • ".join(p.get("model_notes", []))
        edge_status = "VERIFIED" if p.get("verified_projection") else "LINE FALLBACK"
        edge_badge = "badge-green" if p.get("verified_projection") else "badge-red"
        st.markdown(f"""
        <div class='{card_class}'>
          <div class='big'>{p['player']} — {p['prop_label']}</div>
          <span class='badge badge-blue'>{p.get('league','')} Underdog</span>
          <span class='badge {'badge-green' if p['qualified'] else 'badge-orange'}'>{p['signal']}</span>
          <span class='badge'>Pick: {p['pick_side']}</span>
          <span class='badge'>Line: {p['line']}</span>
          <span class='badge'>Date: {p.get('date_bucket','')}</span>
          <span class='badge'>Proj Src: {p.get('projection_source','')}</span>
          <span class='badge {edge_badge}'>{edge_status}</span>
          <div class='metric-grid'>
            <div class='metric-box'><div class='metric-label'>Projection</div><div class='metric-value'>{p['projection']:.2f}</div><div class='metric-sub'>Line {p['line']} | Raw {p.get('raw_edge',0):+.2f}</div></div>
            <div class='metric-box'><div class='metric-label'>Edge</div><div class='metric-value'>{p.get('pick_edge',0):+.2f}</div><div class='metric-sub'>Prob edge {p.get('prob_edge',0)*100:.1f}% | {p.get('edge_pct',0)*100:.1f}%</div></div>
            <div class='metric-box'><div class='metric-label'>Over / Under</div><div class='metric-value'>{p['over_prob']*100:.1f}%</div><div class='metric-sub'>Under {p['under_prob']*100:.1f}%</div></div>
            <div class='metric-box'><div class='metric-label'>EV / Kelly</div><div class='metric-value'>{(p['ev'] or 0)*100:.1f}%</div><div class='metric-sub'>Proxy {odds_display(p.get('ev_price', EV_PRICE))} | Stake ${p['stake']:.2f}</div></div>
            <div class='metric-box'><div class='metric-label'>Data Score</div><div class='metric-value'>{p['data_score']}</div><div class='metric-sub'>CLV {p['clv']:+.2f}</div></div>
          </div>
          <div class='sub'>Game: {p.get('game','')} | Team: {p.get('team','')} | Match: {p.get('match_name','')} ({p.get('match_score',0):.2f})</div>
          <div class='sub'>Notes: {reason_text}</div>
          <div class='sub'>Model: {notes}</div>
        </div>
        """, unsafe_allow_html=True)

tabs = st.tabs(["🏀 NBA Props", "🏀 WNBA Props", "😈 All Top Plays", "📋 Full Board", "🔎 Raw Lines", "💾 Official Picks", "✅ Grade + Learn", "📈 Performance", "🔌 Source Logs", "⚙️ Setup"])

with tabs[0]:
    render_cards(nba_board, "NBA Props — Verified NBA Game-Log Projections")

with tabs[1]:
    render_cards(wnba_board, "WNBA Props — Verified WNBA Game-Log Projections")

with tabs[2]:
    render_cards(board, "All Top Plays")

with tabs[3]:
    if board:
        df = pd.DataFrame(board).drop(columns=["raw"], errors="ignore")
        cols = ["signal","league","player","team","game","date_bucket","local_start","prop","line","pick_side","projection","projection_source","verified_projection","l3","l5","l10","season_avg","minutes","over_prob","under_prob","pick_prob","raw_edge","pick_edge","edge_pct","prob_edge","min_edge","ev","ev_price","kelly","stake","data_score","reasons"]
        st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True, hide_index=True)
    else:
        st.info("No board rows.")

with tabs[4]:
    if source_rows:
        st.dataframe(pd.DataFrame(source_rows).drop(columns=["raw"], errors="ignore"), use_container_width=True, hide_index=True)
    else:
        st.info("No raw lines loaded.")

with tabs[5]:
    picks = load_json(PICK_LOG, [])
    if picks:
        st.dataframe(pd.DataFrame(picks), use_container_width=True, hide_index=True)
    else:
        st.info("No official picks saved yet.")

with tabs[6]:
    st.markdown("<div class='section-title'>Manual Grading</div>", unsafe_allow_html=True)
    picks = load_json(PICK_LOG, [])
    ungraded = [p for p in picks if not p.get("graded")]
    if ungraded:
        labels = [f"{p.get('league')} {p['player']} {p['prop']} {p['side']} {p['line']} | {p.get('game','')}" for p in ungraded]
        idx = st.selectbox("Pick to grade", range(len(ungraded)), format_func=lambda i: labels[i])
        actual = st.number_input("Actual result", value=0.0, step=0.5)
        if st.button("✅ Grade Selected Pick", use_container_width=True):
            if grade_saved_pick(ungraded[idx]["pick_id"], actual):
                st.success("Pick graded and learning updated.")
            else:
                st.error("Could not grade selected pick.")
    else:
        st.info("No ungraded official picks.")

    st.markdown("### Add Manual Result")
    with st.form("manual_result"):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: m_league = st.selectbox("League", ["NBA", "WNBA"])
        with c2: m_player = st.text_input("Player")
        with c3: m_prop = st.selectbox("Prop", list(PROP_CONFIG.keys()))
        with c4: m_line = st.number_input("Line", value=0.0, step=0.5)
        with c5: m_actual = st.number_input("Actual", value=0.0, step=0.5)
        if st.form_submit_button("Add Result + Rebuild"):
            if m_player:
                results = load_json(RESULT_LOG, [])
                side = "OVER" if m_actual > m_line else "UNDER"
                results.append({"pick_id": f"manual_{now_iso()}_{m_league}_{normalize_name(m_player)}_{m_prop}", "saved_at": now_iso(), "source": "Manual", "league": m_league, "player": m_player, "prop": m_prop, "line": m_line, "actual": m_actual, "side": side, "win": True, "qualified": False, "manual": True})
                save_json(RESULT_LOG, results)
                n = rebuild_learning_from_results()
                st.success(f"Manual result saved. Learning rebuilt from {n} results.")

with tabs[7]:
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
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
    st.subheader("Learning State")
    st.json(load_learning())

with tabs[8]:
    logs = load_json(REQUEST_LOG_FILE, [])
    if logs:
        st.dataframe(pd.DataFrame(logs[-400:]), use_container_width=True, hide_index=True)
    else:
        st.info("No source logs yet.")

with tabs[9]:
    st.markdown("""
    ### Railway env vars

    ```text
    ENABLE_REAL_STATS=true
    REQUIRE_REAL_PROJECTION=true
    MARKET_BLEND=0.15
    NBA_SEASON=2025-26
    WNBA_SEASON=2026
    EV_PRICE=-110
    ```

    Use **Clear Cache** then **Refresh Lines** after changing env vars.

    This build does not generate fake props. If a player cannot be matched to NBA/WNBA stats logs, that row is marked `LINE FALLBACK` and will PASS while verified projections are required.
    """)
