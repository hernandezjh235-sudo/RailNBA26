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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


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

DEFAULT_PRICE = float(os.getenv("DEFAULT_PRICE", "-110"))
DEFAULT_SIM_COUNT = int(os.getenv("PROP_SIMULATION_COUNT", "12000"))
DEFAULT_MIN_PROB = float(os.getenv("MIN_PROB", "0.57"))
DEFAULT_MIN_DATA_SCORE = int(os.getenv("MIN_DATA_SCORE", "68"))
DEFAULT_MAX_KELLY = float(os.getenv("MAX_KELLY", "0.03"))
DEFAULT_BANKROLL = float(os.getenv("BANKROLL", "1000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "120"))

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
def fetch_underdog_lines(url: str) -> List[Dict[str, Any]]:
    """
    Pull real Underdog lines only.

    Primary endpoint:
      https://api.underdogfantasy.com/beta/v5/over_under_lines

    This function supports both current v5 top-level structure and older included/data shapes.
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

    # Current v5 shape: players, appearances, games, over_under_lines
    if isinstance(data, dict) and isinstance(data.get("players"), list) and isinstance(data.get("appearances"), list):
        players_raw = data.get("players", []) or []
        appearances_raw = data.get("appearances", []) or []
        lines_raw = data.get("over_under_lines", []) or []
        games_raw = data.get("games", []) or []

        players = {str(p.get("id")): p for p in players_raw if isinstance(p, dict)}
        appearances = {str(a.get("id")): a for a in appearances_raw if isinstance(a, dict)}
        games = {str(g.get("id")): g for g in games_raw if isinstance(g, dict)}

        for line_item in lines_raw:
            if not isinstance(line_item, dict):
                continue

            ou = line_item.get("over_under") if isinstance(line_item.get("over_under"), dict) else {}
            app_stat = ou.get("appearance_stat") if isinstance(ou.get("appearance_stat"), dict) else {}

            appearance_id = (
                app_stat.get("appearance_id")
                or line_item.get("appearance_id")
                or line_item.get("appearanceId")
            )
            stat_name = (
                app_stat.get("stat")
                or app_stat.get("display_stat")
                or line_item.get("stat")
                or line_item.get("stat_type")
                or line_item.get("display_stat")
                or ou.get("title")
            )
            line_value = (
                line_item.get("stat_value")
                or line_item.get("line")
                or line_item.get("value")
                or ou.get("line")
                or ou.get("stat_value")
                or app_stat.get("value")
            )

            app = appearances.get(str(appearance_id), {}) if appearance_id is not None else {}
            player_id = app.get("player_id") or app.get("playerId") or line_item.get("player_id")
            player = players.get(str(player_id), {}) if player_id is not None else {}

            first = player.get("first_name") or player.get("firstName") or ""
            last = player.get("last_name") or player.get("lastName") or ""
            player_name = (
                player.get("full_name")
                or player.get("fullName")
                or player.get("display_name")
                or app.get("player_name")
                or f"{first} {last}".strip()
            )

            team = app.get("team_abbr") or app.get("team") or player.get("team") or player.get("team_abbr") or ""
            opponent = app.get("opponent_abbr") or app.get("opponent") or ""
            game_id = app.get("game_id") or app.get("match_id") or line_item.get("game_id") or ""
            game_obj = games.get(str(game_id), {}) if game_id else {}
            game = (
                app.get("match_title")
                or line_item.get("match_title")
                or game_obj.get("title")
                or (str(game_obj.get("away_team_name", "")) + (" @ " if game_obj else "") + str(game_obj.get("home_team_name", ""))).strip()
            )
            start_time = (
                app.get("scheduled_at")
                or app.get("start_time")
                or game_obj.get("scheduled_at")
                or game_obj.get("start_time")
                or ""
            )

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
                raw=line_item,
            )
            if row:
                rows.append(row)

        log_request("Underdog", "OK", f"{len(rows)} rows from {used_url}")
        save_json(SNAPSHOT_FILE, {"created_at": now_iso(), "rows": rows})
        return rows

    # Older JSON/API fallback shape
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

        stat_type = (
            attrs.get("stat_type")
            or attrs.get("stat")
            or attrs.get("display_stat")
            or app_stat.get("stat")
            or over_under.get("title")
        )
        line = (
            attrs.get("line")
            or attrs.get("stat_value")
            or attrs.get("value")
            or over_under.get("line")
            or over_under.get("stat_value")
        )

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

        app = appearances.get(str(appearance_id), {}) if appearance_id else {}
        pl = players.get(str(player_id), {}) if player_id else {}

        player_name = (
            attrs.get("player_name")
            or attrs.get("player")
            or attrs.get("athlete_name")
            or app.get("player_name")
            or app.get("name")
            or app.get("display_name")
            or (str(pl.get("first_name", "")) + " " + str(pl.get("last_name", ""))).strip()
        )

        row = normalize_prop_row(
            player=player_name,
            prop=stat_type,
            line=line,
            team=attrs.get("team") or attrs.get("team_abbr") or app.get("team_abbr") or app.get("team") or "",
            opponent=attrs.get("opponent") or app.get("opponent_abbr") or app.get("opponent") or "",
            game=attrs.get("match_title") or attrs.get("game") or app.get("match_title") or "",
            start_time=attrs.get("scheduled_at") or attrs.get("start_time") or app.get("scheduled_at") or "",
            price=DEFAULT_PRICE,
            raw=attrs,
        )
        if row:
            rows.append(row)

    log_request("Underdog", "OK", f"{len(rows)} rows from {used_url} fallback parser")
    save_json(SNAPSHOT_FILE, {"created_at": now_iso(), "rows": rows})
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

def projection_from_line(row: Dict[str, Any], use_learning: bool, use_markov: bool) -> Tuple[float, float, str, float, str]:
    learning = load_learning()
    line = float(row["line"])
    pkey = normalize_name(row["player"])
    prop = row["prop"]

    player_bias = safe_float((learning.get("player_bias") or {}).get(pkey), 0.0) or 0.0
    prop_bias = safe_float((learning.get("prop_bias") or {}).get(prop), 0.0) or 0.0
    bias = player_bias + prop_bias if use_learning else 0.0

    projection = line + bias
    std = float(PROP_CONFIG[prop].get("std", 3.0))
    markov_state, markov_adj = markov_recent_state(row["player"], prop, line, use_markov)
    note = f"Bias {bias:+.2f} | player {player_bias:+.2f}, prop {prop_bias:+.2f}"
    return projection, std, markov_state, markov_adj, note

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
) -> Dict[str, Any]:
    learning = load_learning()
    learning_samples = int(learning.get("samples", 0) or 0)

    projection, std, markov_state, markov_prob_adj, projection_note = projection_from_line(
        row, use_learning, use_markov
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
    edge = abs(float(projection) - float(row["line"]))
    min_edge = float(PROP_CONFIG[row["prop"]]["min_edge"]) * min_edge_scale

    ev = expected_value(pick_prob, row.get("price", DEFAULT_PRICE))
    kelly = kelly_fraction(pick_prob, row.get("price", DEFAULT_PRICE), max_kelly)
    stake = bankroll * kelly
    clv = update_clv(row, side)

    reasons = []
    if pick_prob < min_prob:
        reasons.append("probability below gate")
    if edge < min_edge:
        reasons.append("edge below gate")
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
        "model_notes": [projection_note, bayes_note],
        "learning_samples": learning_samples,
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
        -x["edge"],
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
    show_passes = st.toggle("Show PASS rows", value=True)

    st.markdown("### Source")
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
        st.session_state["underdog_rows"] = fetch_underdog_lines(UNDERDOG_API_URL)
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
            "signal", "player", "team", "game", "prop", "line", "pick_side",
            "projection", "over_prob", "under_prob", "pick_prob", "edge", "ev",
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

