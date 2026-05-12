
# -*- coding: utf-8 -*-
# ============================================================
# DEVIL PICKS 9.5 — NBA FULL MARKET + UNDERDOG PROP ENGINE
# GitHub / Streamlit / Railway ready single-file app.py
# Moneyline + Spread + Total kept separate from Player Props
# NBA schedule: NBA CDN first, ESPN fallback
# Props: The Odds API props + Underdog lines
# Models: XGBoost optional training/fallback, Bayesian, Markov trend,
# Monte Carlo, EV, Kelly, CLV, injury/minutes weighting
# ============================================================

import os
import re
import json
import math
import time
import difflib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Devil Picks 9.5 NBA Engine",
    page_icon="😈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "9.5.1"
LOCAL_DIR = "devil_picks_data"
os.makedirs(LOCAL_DIR, exist_ok=True)

PICK_LOG = os.path.join(LOCAL_DIR, "market_pick_log.json")
PROP_PICK_LOG = os.path.join(LOCAL_DIR, "prop_pick_log.json")
CLV_FILE = os.path.join(LOCAL_DIR, "market_clv.json")
PROP_CLV_FILE = os.path.join(LOCAL_DIR, "prop_clv.json")
LEARN_FILE = os.path.join(LOCAL_DIR, "learning.json")
INJURY_FILE = os.path.join(LOCAL_DIR, "injury_adjustments.json")
REQUEST_LOG_FILE = os.path.join(LOCAL_DIR, "request_log.json")

NBA_CDN_TODAY = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
NBA_CDN_BY_DATE = "https://cdn.nba.com/static/json/liveData/scoreboard/scoreboard_00.json"
NBA_BOXSCORE = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"

UNDERDOG_API = "https://api.underdogfantasy.com/v1/over_under_lines"

DEFAULT_ODDS = -110
MAX_KELLY = 0.03
SIMS = 12000
PROP_SIMS = 9000

TEAM_ALIASES = {
    "ATL":"Atlanta Hawks","BOS":"Boston Celtics","BKN":"Brooklyn Nets","CHA":"Charlotte Hornets","CHI":"Chicago Bulls","CLE":"Cleveland Cavaliers",
    "DAL":"Dallas Mavericks","DEN":"Denver Nuggets","DET":"Detroit Pistons","GSW":"Golden State Warriors","HOU":"Houston Rockets","IND":"Indiana Pacers",
    "LAC":"LA Clippers","LAL":"Los Angeles Lakers","MEM":"Memphis Grizzlies","MIA":"Miami Heat","MIL":"Milwaukee Bucks","MIN":"Minnesota Timberwolves",
    "NOP":"New Orleans Pelicans","NYK":"New York Knicks","OKC":"Oklahoma City Thunder","ORL":"Orlando Magic","PHI":"Philadelphia 76ers","PHX":"Phoenix Suns",
    "POR":"Portland Trail Blazers","SAC":"Sacramento Kings","SAS":"San Antonio Spurs","TOR":"Toronto Raptors","UTA":"Utah Jazz","WAS":"Washington Wizards",
}
NAME_TO_ABBR = {v.lower(): k for k, v in TEAM_ALIASES.items()}
EXTRA_NAME_TO_ABBR = {
    "la clippers": "LAC", "los angeles clippers": "LAC",
    "brooklyn nets": "BKN", "new york knicks": "NYK",
    "phoenix suns": "PHX", "golden state warriors": "GSW",
    "new orleans pelicans": "NOP", "san antonio spurs": "SAS",
    "oklahoma city thunder": "OKC",
}

PROP_TYPES = {
    "PTS": {"label": "Points", "odds_market": "player_points", "ud_terms": ["points"], "min_edge": 1.4},
    "REB": {"label": "Rebounds", "odds_market": "player_rebounds", "ud_terms": ["rebounds"], "min_edge": 1.2},
    "AST": {"label": "Assists", "odds_market": "player_assists", "ud_terms": ["assists"], "min_edge": 1.2},
    "PRA": {"label": "Pts+Rebs+Asts", "odds_market": "player_points_rebounds_assists", "ud_terms": ["points + rebounds + assists", "pts + rebs + asts", "pra"], "min_edge": 2.2},
    "PR": {"label": "Pts+Rebs", "odds_market": "player_points_rebounds", "ud_terms": ["points + rebounds", "pts + rebs"], "min_edge": 1.8},
    "PA": {"label": "Pts+Asts", "odds_market": "player_points_assists", "ud_terms": ["points + assists", "pts + asts"], "min_edge": 1.8},
    "RA": {"label": "Rebs+Asts", "odds_market": "player_rebounds_assists", "ud_terms": ["rebounds + assists", "rebs + asts"], "min_edge": 1.7},
    "3PM": {"label": "3-Pointers Made", "odds_market": "player_threes", "ud_terms": ["3-pointers made", "three pointers made", "3pt made"], "min_edge": 0.55},
}

# ============================================================
# STYLES
# ============================================================

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at top left,#21000a 0%,#070b13 40%,#02040a 100%); color:#f7f8fb;}
.block-container {padding-top:1rem; max-width:1650px;}
section[data-testid="stSidebar"] {background:#050912; border-right:1px solid rgba(255,52,79,.28);}
.hero {border:1px solid rgba(255,255,255,.16); background:linear-gradient(135deg,rgba(12,19,34,.96),rgba(5,8,18,.96)); border-radius:24px; padding:22px; box-shadow:0 0 34px rgba(255,52,79,.12); margin-bottom:16px;}
.title {font-size:32px; font-weight:950;}
.sub {color:#aeb7c9; font-size:13px;}
.card {border:1px solid rgba(255,255,255,.14); background:linear-gradient(145deg,#0a111f,#080d18); border-radius:18px; padding:16px; margin-bottom:14px;}
.good {border-color:rgba(56,240,99,.50); box-shadow:0 0 20px rgba(56,240,99,.10);}
.warn {border-color:rgba(255,176,46,.50);}
.badge {display:inline-block; padding:6px 10px; border-radius:999px; font-weight:900; font-size:12px; margin:3px 5px 3px 0; border:1px solid rgba(255,255,255,.18); background:#101827; color:#dce4f5;}
.green {color:#38f063;} .red {color:#ff344f;} .orange {color:#ffb02e;} .muted {color:#aeb7c9;}
.metric-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:10px 0;}
.metric-box {border:1px solid rgba(255,255,255,.14); background:#0a111f; border-radius:16px; padding:14px;}
.metric-label {font-size:12px; color:#aeb7c9; text-transform:uppercase; font-weight:800;}
.metric-value {font-size:24px; color:#fff; font-weight:950; margin-top:4px;}
.section-title {font-size:22px; font-weight:950; margin:18px 0 10px; border-left:5px solid #ff344f; padding-left:12px;}
@media (max-width: 1000px) {.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================

def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

ODDS_API_KEY = get_secret("ODDS_API_KEY", "")

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def eastern_date(offset_days=0):
    if ZoneInfo:
        dt = datetime.now(ZoneInfo("America/New_York")) + timedelta(days=offset_days)
    else:
        dt = datetime.utcnow() - timedelta(hours=5) + timedelta(days=offset_days)
    return dt.strftime("%Y-%m-%d")

def yyyymmdd(date_str):
    return str(date_str).replace("-", "")

def safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass

def log_request(source, status, message=""):
    rows = load_json(REQUEST_LOG_FILE, [])
    rows.append({"time": now_iso(), "source": str(source)[:180], "status": str(status), "message": str(message)[:500]})
    save_json(REQUEST_LOG_FILE, rows[-500:])

def normalize_name(name):
    s = str(name or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def name_score(a, b):
    a = normalize_name(a)
    b = normalize_name(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.94
    return difflib.SequenceMatcher(None, a, b).ratio()

def team_abbr_from_name(name):
    n = normalize_name(name)
    if n in EXTRA_NAME_TO_ABBR:
        return EXTRA_NAME_TO_ABBR[n]
    if n in NAME_TO_ABBR:
        return NAME_TO_ABBR[n]
    best, score = None, 0
    for abbr, full in TEAM_ALIASES.items():
        sc = name_score(n, full)
        if sc > score:
            best, score = abbr, sc
    return best if score >= 0.72 else (name or "")[:3].upper()

@st.cache_data(ttl=60, show_spinner=False)
def safe_get_json(url, params=None, headers=None, timeout=20):
    h = {
        "User-Agent": "Mozilla/5.0 DevilPicks/9.5",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nba.com/",
    }
    if headers:
        h.update(headers)
    try:
        r = requests.get(url, params=params, headers=h, timeout=timeout)
        if r.status_code != 200:
            log_request(url, f"HTTP {r.status_code}", r.text[:250])
            return None
        return r.json()
    except Exception as e:
        log_request(url, "REQUEST_ERROR", str(e))
        return None

def decimal_odds(odds):
    odds = safe_float(odds)
    if odds is None:
        return None
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)

def american_to_implied(odds):
    odds = safe_float(odds)
    if odds is None:
        return None
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def expected_value(prob, odds):
    dec = decimal_odds(odds)
    if prob is None or dec is None:
        return None
    return (prob * (dec - 1)) - (1 - prob)

def kelly_fraction(prob, odds):
    dec = decimal_odds(odds)
    if prob is None or dec is None:
        return 0.0
    b = dec - 1
    q = 1 - prob
    if b <= 0:
        return 0.0
    return float(clamp(((b * prob) - q) / b, 0.0, MAX_KELLY))

def odds_display(o):
    o = safe_float(o)
    if o is None:
        return "N/A"
    return f"+{int(o)}" if o > 0 else str(int(o))

def update_clv(path, key, value):
    if value is None:
        return 0.0
    data = load_json(path, {})
    v = float(value)
    if key not in data:
        data[key] = {"open": v, "latest": v, "created_at": now_iso(), "updated_at": now_iso()}
        save_json(path, data)
        return 0.0
    open_v = safe_float(data[key].get("open"), v)
    data[key]["latest"] = v
    data[key]["updated_at"] = now_iso()
    save_json(path, data)
    return round(v - open_v, 3)

# ============================================================
# NBA SCHEDULE LOADER: NBA CDN FIRST, ESPN FALLBACK
# ============================================================

def parse_nba_cdn_games(data, date_str):
    out = []
    games = (data or {}).get("scoreboard", {}).get("games", []) or []
    for g in games:
        home = g.get("homeTeam", {}) or {}
        away = g.get("awayTeam", {}) or {}
        home_abbr = home.get("teamTricode") or team_abbr_from_name(home.get("teamName"))
        away_abbr = away.get("teamTricode") or team_abbr_from_name(away.get("teamName"))
        gid = str(g.get("gameId") or f"{date_str}_{away_abbr}_{home_abbr}")
        out.append({
            "date": date_str,
            "game_id": gid,
            "home": home_abbr,
            "away": away_abbr,
            "home_name": TEAM_ALIASES.get(home_abbr, home.get("teamName") or home_abbr),
            "away_name": TEAM_ALIASES.get(away_abbr, away.get("teamName") or away_abbr),
            "home_score": safe_int(home.get("score")),
            "away_score": safe_int(away.get("score")),
            "status": g.get("gameStatusText") or "Scheduled",
            "game_time": g.get("gameTimeUTC") or g.get("gameEt") or "",
            "source": "NBA CDN",
        })
    return out

def parse_espn_games(data, date_str):
    out = []
    events = (data or {}).get("events", []) or []
    for ev in events:
        comp = (ev.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", []) or []
        home_c = None
        away_c = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_c = c
            elif c.get("homeAway") == "away":
                away_c = c
        if not home_c or not away_c:
            continue
        home_team = home_c.get("team", {}) or {}
        away_team = away_c.get("team", {}) or {}
        home_abbr = home_team.get("abbreviation") or team_abbr_from_name(home_team.get("displayName"))
        away_abbr = away_team.get("abbreviation") or team_abbr_from_name(away_team.get("displayName"))
        status = ((comp.get("status") or {}).get("type") or {}).get("shortDetail") or ((comp.get("status") or {}).get("type") or {}).get("description") or "Scheduled"
        out.append({
            "date": date_str,
            "game_id": str(ev.get("id") or f"{date_str}_{away_abbr}_{home_abbr}"),
            "home": home_abbr,
            "away": away_abbr,
            "home_name": TEAM_ALIASES.get(home_abbr, home_team.get("displayName") or home_abbr),
            "away_name": TEAM_ALIASES.get(away_abbr, away_team.get("displayName") or away_abbr),
            "home_score": safe_int(home_c.get("score")),
            "away_score": safe_int(away_c.get("score")),
            "status": status,
            "game_time": ev.get("date", ""),
            "source": "ESPN fallback",
        })
    return out

@st.cache_data(ttl=90, show_spinner=False)
def get_nba_games_for_date(date_str):
    # 1) NBA CDN by date
    data = safe_get_json(NBA_CDN_BY_DATE, params={"GameDate": date_str}, timeout=15)
    games = parse_nba_cdn_games(data, date_str)
    if games:
        return games

    # 2) NBA CDN today endpoint
    if date_str == eastern_date(0):
        data = safe_get_json(NBA_CDN_TODAY, timeout=15)
        games = parse_nba_cdn_games(data, date_str)
        if games:
            return games

    # 3) ESPN fallback
    data = safe_get_json(ESPN_SCOREBOARD, params={"dates": yyyymmdd(date_str), "limit": 30}, timeout=15)
    games = parse_espn_games(data, date_str)
    return games

def get_dates(day_mode):
    if day_mode == "Today":
        return [eastern_date(0)]
    if day_mode == "Tomorrow":
        return [eastern_date(1)]
    return [eastern_date(0), eastern_date(1)]

# ============================================================
# ODDS API: MARKETS + SPORTSBOOK PROPS
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def get_odds_market_events():
    if not ODDS_API_KEY:
        return []
    url = f"{ODDS_BASE}/sports/{SPORT_KEY}/odds"
    data = safe_get_json(
        url,
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        },
        timeout=20,
    )
    return data if isinstance(data, list) else []

@st.cache_data(ttl=300, show_spinner=False)
def get_odds_event_list():
    if not ODDS_API_KEY:
        return []
    url = f"{ODDS_BASE}/sports/{SPORT_KEY}/events"
    data = safe_get_json(url, params={"apiKey": ODDS_API_KEY}, timeout=20)
    return data if isinstance(data, list) else []

def match_odds_event(game, events):
    best, best_score = None, 0.0
    for ev in events or []:
        h, a = ev.get("home_team", ""), ev.get("away_team", "")
        score = max(
            (name_score(game["home_name"], h) + name_score(game["away_name"], a)) / 2,
            (name_score(game["home_name"], a) + name_score(game["away_name"], h)) / 2,
        )
        if score > best_score:
            best, best_score = ev, score
    return best if best_score >= 0.70 else None

def extract_consensus_market(game, odds_events):
    matched = match_odds_event(game, odds_events)
    result = {
        "home_price": None, "away_price": None, "home_spread": None, "spread_price": DEFAULT_ODDS,
        "total": None, "over_price": DEFAULT_ODDS, "under_price": DEFAULT_ODDS,
        "quality": "NO ODDS", "rows": [],
    }
    if not matched:
        return result

    home_prices, away_prices, spreads, totals, over_prices, under_prices, rows = [], [], [], [], [], [], []
    for book in matched.get("bookmakers", []) or []:
        row = {"Book": book.get("title") or book.get("key"), "Home ML": None, "Away ML": None, "Spread": None, "Total": None}
        for market in book.get("markets", []) or []:
            key = market.get("key")
            for out in market.get("outcomes", []) or []:
                nm = out.get("name", "")
                price = safe_float(out.get("price"))
                point = safe_float(out.get("point"))
                if key == "h2h":
                    if name_score(nm, game["home_name"]) >= .70:
                        home_prices.append(price); row["Home ML"] = price
                    elif name_score(nm, game["away_name"]) >= .70:
                        away_prices.append(price); row["Away ML"] = price
                elif key == "spreads" and point is not None:
                    if name_score(nm, game["home_name"]) >= .70:
                        spreads.append(point); row["Spread"] = point
                elif key == "totals" and point is not None:
                    totals.append(point); row["Total"] = point
                    if normalize_name(nm) == "over":
                        over_prices.append(price)
                    elif normalize_name(nm) == "under":
                        under_prices.append(price)
        rows.append(row)

    if home_prices: result["home_price"] = float(np.nanmedian(home_prices))
    if away_prices: result["away_price"] = float(np.nanmedian(away_prices))
    if spreads: result["home_spread"] = float(np.nanmedian(spreads))
    if totals: result["total"] = float(np.nanmedian(totals))
    if over_prices: result["over_price"] = float(np.nanmedian(over_prices))
    if under_prices: result["under_price"] = float(np.nanmedian(under_prices))
    result["rows"] = rows
    count = sum([result["home_price"] is not None, result["away_price"] is not None, result["home_spread"] is not None, result["total"] is not None])
    result["quality"] = "STRONG" if count >= 4 and len(rows) >= 2 else "OK" if count >= 2 else "THIN"
    return result

@st.cache_data(ttl=180, show_spinner=False)
def get_sportsbook_props_for_event(event_id, markets_csv):
    if not ODDS_API_KEY or not event_id:
        return []
    url = f"{ODDS_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds"
    data = safe_get_json(
        url,
        params={"apiKey": ODDS_API_KEY, "regions": "us", "markets": markets_csv, "oddsFormat": "american"},
        timeout=25,
    )
    rows = []
    if not isinstance(data, dict):
        return rows
    for book in data.get("bookmakers", []) or []:
        book_name = book.get("title") or book.get("key") or "Book"
        for market in book.get("markets", []) or []:
            mkey = market.get("key")
            for out in market.get("outcomes", []) or []:
                player = out.get("description") or out.get("player") or out.get("participant") or ""
                rows.append({
                    "Source": "Sportsbook",
                    "Book": book_name,
                    "Market": mkey,
                    "Player": player,
                    "Prop": odds_market_to_prop(mkey),
                    "Side": str(out.get("name", "")).upper(),
                    "Line": safe_float(out.get("point")),
                    "Price": safe_float(out.get("price"), DEFAULT_ODDS),
                    "Updated": market.get("last_update") or book.get("last_update"),
                })
    return rows

def odds_market_to_prop(market):
    for p, cfg in PROP_TYPES.items():
        if cfg["odds_market"] == market:
            return p
    return market

# ============================================================
# UNDERDOG LINES
# ============================================================

def underdog_stat_to_prop(stat_name):
    s = normalize_name(stat_name)
    for prop, cfg in PROP_TYPES.items():
        for term in cfg["ud_terms"]:
            if normalize_name(term) in s or s in normalize_name(term):
                return prop
    if "points" in s and "rebounds" in s and "assists" in s:
        return "PRA"
    if "points" in s and "rebounds" in s:
        return "PR"
    if "points" in s and "assists" in s:
        return "PA"
    if "rebounds" in s and "assists" in s:
        return "RA"
    if "points" in s:
        return "PTS"
    if "rebounds" in s:
        return "REB"
    if "assists" in s:
        return "AST"
    if "three" in s or "3" in s:
        return "3PM"
    return None

def parse_underdog_response(data):
    rows = []
    if not isinstance(data, dict):
        return rows

    included = data.get("included", []) or []
    inc_by_type = {}
    for item in included:
        typ = item.get("type")
        iid = str(item.get("id"))
        inc_by_type.setdefault(typ, {})[iid] = item

    # Flexible helper for attributes
    def attrs(obj):
        return obj.get("attributes", {}) or {}

    appearances = inc_by_type.get("appearance", {}) or inc_by_type.get("appearances", {})
    players = inc_by_type.get("player", {}) or inc_by_type.get("players", {})
    stat_types = inc_by_type.get("stat_type", {}) or inc_by_type.get("stat_types", {})

    raw_lines = []
    if isinstance(data.get("data"), list):
        raw_lines = data.get("data")
    elif isinstance(data.get("over_under_lines"), list):
        raw_lines = data.get("over_under_lines")
    elif isinstance(data.get("lines"), list):
        raw_lines = data.get("lines")

    # Underdog sometimes keeps lines in included
    if not raw_lines:
        for typ, bucket in inc_by_type.items():
            if "over_under" in str(typ):
                raw_lines.extend(bucket.values())

    for line in raw_lines:
        a = attrs(line) if "attributes" in line else line
        rel = line.get("relationships", {}) or {}

        line_value = (
            safe_float(a.get("stat_value")) or
            safe_float(a.get("line")) or
            safe_float(a.get("value")) or
            safe_float(a.get("over_under"))
        )
        if line_value is None:
            continue

        appearance_id = None
        stat_type_id = None
        try:
            appearance_id = str(rel.get("appearance", {}).get("data", {}).get("id"))
        except Exception:
            pass
        try:
            stat_type_id = str(rel.get("over_under", {}).get("data", {}).get("id"))
        except Exception:
            pass
        if not stat_type_id:
            try:
                stat_type_id = str(rel.get("stat_type", {}).get("data", {}).get("id"))
            except Exception:
                pass

        app = appearances.get(appearance_id, {}) if appearance_id else {}
        app_a = attrs(app)
        player_name = a.get("player_name") or app_a.get("display_name") or app_a.get("player_name") or app_a.get("name")

        # Try player relationship through appearance
        if not player_name:
            try:
                pid = str((app.get("relationships", {}) or {}).get("player", {}).get("data", {}).get("id"))
                player_name = attrs(players.get(pid, {})).get("display_name") or attrs(players.get(pid, {})).get("first_name", "") + " " + attrs(players.get(pid, {})).get("last_name", "")
            except Exception:
                pass

        stat_name = a.get("stat_type") or a.get("title") or a.get("display_stat") or a.get("over_under_title")
        if not stat_name and stat_type_id:
            stat_name = attrs(stat_types.get(stat_type_id, {})).get("display_name") or attrs(stat_types.get(stat_type_id, {})).get("name")

        # Some payloads include title like "LeBron James Points"
        if not player_name:
            title = str(a.get("title") or a.get("display_title") or "")
            player_name = title

        prop = underdog_stat_to_prop(stat_name or "")
        if not prop:
            continue

        sport = str(a.get("sport_id") or a.get("sport") or app_a.get("sport_id") or app_a.get("sport") or "").lower()
        league = str(a.get("league") or app_a.get("league") or app_a.get("sport_name") or "").lower()
        if sport and "nba" not in sport and "basketball" not in sport and league and "nba" not in league:
            # Don't hard filter if missing, only if clearly another sport
            continue

        game_text = a.get("game") or app_a.get("matchup") or app_a.get("game_title") or ""
        team = a.get("team") or app_a.get("team_abbr") or app_a.get("team") or ""

        rows.append({
            "Source": "Underdog",
            "Book": "Underdog",
            "Market": f"underdog_{prop}",
            "Player": str(player_name or "").strip(),
            "Prop": prop,
            "Side": "OVER/UNDER",
            "Line": float(line_value),
            "Price": DEFAULT_ODDS,
            "Updated": now_iso(),
            "Team": str(team or "").upper(),
            "Game": game_text,
        })

    # Clean names
    clean = []
    for r in rows:
        if not r["Player"] or len(r["Player"]) < 2:
            continue
        clean.append(r)
    return clean

@st.cache_data(ttl=90, show_spinner=False)
def fetch_underdog_props():
    data = safe_get_json(UNDERDOG_API, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=20)
    return parse_underdog_response(data)

def underdog_props_for_game(game, selected_props):
    rows = fetch_underdog_props()
    out = []
    home = game["home"]
    away = game["away"]
    for r in rows:
        if r.get("Prop") not in selected_props:
            continue
        team = str(r.get("Team") or "").upper()
        game_text = normalize_name(r.get("Game"))
        keep = False
        if team in [home, away]:
            keep = True
        elif normalize_name(home) in game_text or normalize_name(away) in game_text:
            keep = True
        elif not team and not game_text:
            # If Underdog payload lacks team/game, keep it globally but still show as Underdog line.
            keep = True
        if keep:
            rr = dict(r)
            rr["game_id"] = game["game_id"]
            rr["date"] = game["date"]
            rr["home"] = home
            rr["away"] = away
            out.append(rr)
    return out

# ============================================================
# MODEL LAYERS
# ============================================================

def load_learning():
    return load_json(LEARN_FILE, {"market_games": [], "prop_results": [], "team_bias": {}, "player_bias": {}})

def save_learning(data):
    save_json(LEARN_FILE, data)

def bayesian_update(prior, evidence_prob, strength=0.35):
    prior = clamp(float(prior), 0.01, 0.99)
    evidence_prob = clamp(float(evidence_prob), 0.01, 0.99)
    return clamp((prior * (1 - strength)) + (evidence_prob * strength), 0.01, 0.99)

def markov_trend(values):
    vals = [safe_float(v) for v in values if safe_float(v) is not None]
    if len(vals) < 3:
        return 0.0
    diffs = np.diff(vals[:8])
    if len(diffs) == 0:
        return 0.0
    up_rate = float(np.mean(diffs > 0))
    avg_move = float(np.mean(diffs))
    return clamp((up_rate - 0.5) * 2.0 + avg_move * 0.06, -1.25, 1.25)

def xgb_market_probability(features, label_history=None):
    # True XGBoost if enough local history exists; otherwise use logistic fallback.
    x = np.array(features, dtype=float).reshape(1, -1)
    hist = label_history or []
    if XGBOOST_AVAILABLE and len(hist) >= 30:
        try:
            X = np.array([h["features"] for h in hist], dtype=float)
            y = np.array([h["label"] for h in hist], dtype=int)
            model = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, eval_metric="logloss")
            model.fit(X, y)
            return float(model.predict_proba(x)[0][1]), "XGBoost trained"
        except Exception as e:
            log_request("xgb_market_probability", "XGB_ERROR", str(e))

    # Fallback: rating diff and implied features
    rating_diff = features[0]
    market_implied = features[1] if len(features) > 1 and features[1] > 0 else 0.50
    logistic = 1 / (1 + math.exp(-(rating_diff / 7.5)))
    return float(clamp(logistic * 0.70 + market_implied * 0.30, 0.03, 0.97)), "XGBoost fallback logistic"

def xgb_prop_projection(base_projection, features, history=None):
    hist = history or []
    if XGBOOST_AVAILABLE and len(hist) >= 40:
        try:
            X = np.array([h["features"] for h in hist], dtype=float)
            y = np.array([h["actual"] for h in hist], dtype=float)
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
            model.fit(X, y)
            pred = float(model.predict(np.array(features, dtype=float).reshape(1, -1))[0])
            return pred, "XGBoost prop trained"
        except Exception as e:
            log_request("xgb_prop_projection", "XGB_ERROR", str(e))
    return float(base_projection), "XGBoost prop fallback"

def injury_adjustment_for_team(team):
    data = load_json(INJURY_FILE, {})
    return safe_float(data.get(team, 0.0), 0.0) or 0.0

def minutes_projection(player_name, prop_line, prop):
    # Production-safe fallback. If official minutes source is not connected,
    # estimate role from line size and prop type instead of random data.
    line = safe_float(prop_line, 0) or 0
    if prop == "PTS":
        mins = 20 + min(line, 32) * 0.55
    elif prop in ["PRA", "PR", "PA", "RA"]:
        mins = 22 + min(line, 42) * 0.35
    else:
        mins = 20 + min(line, 16) * 0.75
    return float(clamp(mins, 10, 38))

def base_prop_projection_from_line(line, prop, team_adj=0.0):
    # Market-anchored projection is intentional when player historical API is unavailable.
    # It prevents fake random stats while still allowing model layers to evaluate edge.
    line = safe_float(line, 0) or 0
    prop_scale = {"PTS": 0.38, "REB": 0.25, "AST": 0.22, "PRA": 0.55, "PR": 0.45, "PA": 0.42, "RA": 0.36, "3PM": 0.18}.get(prop, 0.30)
    return float(line + team_adj * prop_scale)

# ============================================================
# MARKET MODEL
# ============================================================

def model_market_game(game, bankroll, use_market=True, manual_home_adj=0.0, manual_away_adj=0.0):
    odds = extract_consensus_market(game, get_odds_market_events()) if use_market else {
        "home_price": None, "away_price": None, "home_spread": None, "total": None, "quality": "OFF", "rows": []
    }

    # Rating inputs: scoreboard + market anchored. No fake team stats.
    home_injury = injury_adjustment_for_team(game["home"]) + manual_home_adj
    away_injury = injury_adjustment_for_team(game["away"]) + manual_away_adj
    spread_line = odds.get("home_spread")
    market_rating = -float(spread_line) if spread_line is not None else 0.0
    rating_diff = market_rating + 2.0 + home_injury - away_injury

    home_price = odds.get("home_price") if odds.get("home_price") is not None else DEFAULT_ODDS
    away_price = odds.get("away_price") if odds.get("away_price") is not None else DEFAULT_ODDS
    home_implied = american_to_implied(home_price) or 0.50
    hist = load_learning().get("market_games", [])

    xgb_prob, xgb_note = xgb_market_probability([rating_diff, home_implied, home_injury, away_injury], hist)
    sim_margins = np.random.normal(rating_diff, 12.0, SIMS)
    sim_prob = float(np.mean(sim_margins > 0))
    home_prob = bayesian_update(xgb_prob, sim_prob, strength=0.40)
    away_prob = 1 - home_prob

    pick = game["home"] if home_prob >= away_prob else game["away"]
    pick_prob = max(home_prob, away_prob)
    pick_price = home_price if pick == game["home"] else away_price
    implied = american_to_implied(pick_price) or 0.50
    ev = expected_value(pick_prob, pick_price)
    kelly = kelly_fraction(pick_prob, pick_price)
    edge_prob = pick_prob - implied

    projected_margin = rating_diff
    total_line = odds.get("total")
    projected_total = float(total_line) if total_line is not None else None

    spread_pick = "PASS"
    spread_edge = None
    spread_prob = None
    if spread_line is not None:
        spread_prob = float(np.mean(sim_margins > float(spread_line)))
        if spread_prob >= 0.53:
            spread_pick = f"{game['home']} spread"
            spread_edge = projected_margin - float(spread_line)
        elif (1 - spread_prob) >= 0.53:
            spread_pick = f"{game['away']} spread"
            spread_edge = float(spread_line) - projected_margin

    total_pick = "PASS"
    total_edge = None
    total_prob = None
    if total_line is not None:
        total_sims = np.random.normal(float(total_line), 14.0, SIMS)
        over_prob = float(np.mean(total_sims > float(total_line)))
        under_prob = 1 - over_prob
        total_prob = max(over_prob, under_prob)
        total_pick = "OVER" if over_prob >= under_prob else "UNDER"
        total_edge = 0.0

    data_score = 55
    if odds.get("quality") == "STRONG": data_score += 25
    elif odds.get("quality") == "OK": data_score += 15
    elif odds.get("quality") == "THIN": data_score += 8
    if ODDS_API_KEY: data_score += 10
    data_score = int(clamp(data_score, 0, 100))

    qualified = bool(pick_prob >= 0.55 and edge_prob >= 0.025 and ev is not None and ev > 0.00 and data_score >= 70)
    signal = "PASS"
    if qualified and pick_prob >= 0.61 and ev >= 0.04:
        signal = f"😈 STRONG {pick} ML"
    elif qualified:
        signal = f"✅ LEAN {pick} ML"

    clv = update_clv(CLV_FILE, f"{game['game_id']}_{pick}_ML", pick_price)

    return {
        **game,
        "home_prob": home_prob, "away_prob": away_prob, "model_pick": pick, "pick_prob": pick_prob,
        "home_price": home_price, "away_price": away_price, "pick_price": pick_price,
        "implied": implied, "ev": ev, "kelly": kelly, "edge_prob": edge_prob,
        "rating_diff": rating_diff, "spread_line": spread_line, "spread_pick": spread_pick,
        "spread_edge": spread_edge, "spread_prob": spread_prob, "total_line": total_line,
        "total_pick": total_pick, "total_edge": total_edge, "total_prob": total_prob,
        "projected_total": projected_total, "data_score": data_score, "qualified": qualified,
        "signal": signal, "odds_quality": odds.get("quality"), "odds_rows": odds.get("rows", []),
        "clv": clv, "xgb_note": xgb_note,
    }

# ============================================================
# PLAYER PROP ENGINE: SPORTSBOOK + UNDERDOG, SEPARATE FROM MARKET
# ============================================================

def match_prop_line_to_game(line_row, game):
    team = str(line_row.get("Team") or "").upper()
    if team in [game["home"], game["away"]]:
        return True
    game_text = normalize_name(line_row.get("Game"))
    if normalize_name(game["home"]) in game_text or normalize_name(game["away"]) in game_text:
        return True
    # If sportsbook matched from event endpoint, it belongs to game.
    if line_row.get("Source") == "Sportsbook":
        return True
    # Underdog often omits team in parse; keep globally in Underdog tab, but not top game-specific unless no team data.
    return False

def find_best_prop_lines(game, selected_props, include_underdog=True, include_sportsbook=True):
    lines = []
    if include_sportsbook and ODDS_API_KEY:
        events = get_odds_event_list()
        ev = match_odds_event(game, events)
        if ev:
            markets = ",".join(sorted(set(PROP_TYPES[p]["odds_market"] for p in selected_props)))
            lines.extend(get_sportsbook_props_for_event(ev.get("id"), markets))
    if include_underdog:
        lines.extend(underdog_props_for_game(game, selected_props))

    # Clean rows
    cleaned = []
    for r in lines:
        if r.get("Prop") not in selected_props:
            continue
        if r.get("Line") is None:
            continue
        if not r.get("Player"):
            continue
        rr = dict(r)
        rr["PlayerKey"] = normalize_name(rr["Player"])
        cleaned.append(rr)
    return cleaned

def model_prop_line(row, game, bankroll):
    prop = row["Prop"]
    line = safe_float(row.get("Line"))
    team = row.get("Team") if row.get("Team") in [game["home"], game["away"]] else ""
    team_adj = injury_adjustment_for_team(team) if team else 0.0

    base_proj = base_prop_projection_from_line(line, prop, team_adj=team_adj)
    mins = minutes_projection(row["Player"], line, prop)
    minute_adj = (mins - 28.0) * {"PTS": 0.10, "REB": 0.05, "AST": 0.045, "PRA": 0.18, "PR": 0.14, "PA": 0.13, "RA": 0.10, "3PM": 0.018}.get(prop, 0.06)

    # Markov trend is neutral until actual player history is collected in logs.
    learning = load_learning()
    player_hist = [
        r.get("actual") for r in learning.get("prop_results", [])
        if normalize_name(r.get("Player")) == normalize_name(row["Player"]) and r.get("Prop") == prop
    ][-10:]
    trend_adj = markov_trend(player_hist)
    bayes_prior = 0.50 + clamp((base_proj + minute_adj + trend_adj - line) / max(8.0, line * 0.35), -0.18, 0.18)

    features = [line, mins, team_adj, trend_adj, 1 if row.get("Source") == "Underdog" else 0]
    projection, xgb_note = xgb_prop_projection(base_proj + minute_adj + trend_adj, features, learning.get("prop_results", []))

    sd = max(0.75, abs(line) * {"PTS": 0.22, "REB": 0.28, "AST": 0.30, "PRA": 0.20, "PR": 0.21, "PA": 0.22, "RA": 0.24, "3PM": 0.35}.get(prop, 0.25))
    sims = np.random.normal(projection, sd, PROP_SIMS)
    over_prob_raw = float(np.mean(sims > line))
    over_prob = bayesian_update(bayes_prior, over_prob_raw, strength=0.55)
    under_prob = 1 - over_prob

    side = "OVER" if over_prob >= under_prob else "UNDER"
    pick_prob = max(over_prob, under_prob)
    edge = abs(projection - line)
    price = safe_float(row.get("Price"), DEFAULT_ODDS) or DEFAULT_ODDS
    ev = expected_value(pick_prob, price)
    kelly = kelly_fraction(pick_prob, price)

    threshold = PROP_TYPES[prop]["min_edge"]
    data_score = 60
    if row.get("Source") == "Sportsbook": data_score += 15
    if row.get("Source") == "Underdog": data_score += 12
    if mins >= 24: data_score += 8
    if edge >= threshold: data_score += 10
    data_score = int(clamp(data_score, 0, 100))

    qualified = bool(pick_prob >= 0.56 and edge >= threshold and ev is not None and ev > 0 and data_score >= 72)
    signal = "PASS"
    if qualified and pick_prob >= 0.62:
        signal = f"😈 STRONG {side}"
    elif qualified:
        signal = f"✅ LEAN {side}"

    clv = update_clv(PROP_CLV_FILE, f"{row.get('Source')}_{game['game_id']}_{normalize_name(row['Player'])}_{prop}_{side}", line)

    return {
        **game,
        "Source": row.get("Source"),
        "Book": row.get("Book"),
        "Player": row.get("Player"),
        "Team": team,
        "Prop": prop,
        "Prop Label": PROP_TYPES[prop]["label"],
        "Line": line,
        "Price": price,
        "Projection": projection,
        "Minutes": mins,
        "Side": side,
        "Over Prob": over_prob,
        "Under Prob": under_prob,
        "Pick Prob": pick_prob,
        "Edge": edge,
        "EV": ev,
        "Kelly": kelly,
        "Data Score": data_score,
        "Qualified": qualified,
        "Signal": signal,
        "CLV": clv,
        "XGB": xgb_note,
        "Updated": row.get("Updated"),
        "Game Text": row.get("Game", ""),
    }

def build_prop_board(games, selected_props, bankroll, include_underdog=True, include_sportsbook=True):
    board = []
    raw_underdog_all = fetch_underdog_props() if include_underdog else []
    for g in games:
        rows = find_best_prop_lines(g, selected_props, include_underdog, include_sportsbook)
        for r in rows:
            try:
                board.append(model_prop_line(r, g, bankroll))
            except Exception as e:
                log_request("model_prop_line", "ERROR", str(e))
    # Add a global Underdog table for visibility even if team/game did not parse
    return sorted(board, key=lambda x: (not x["Qualified"], -safe_float(x.get("EV"), -9), -x.get("Pick Prob", 0))), raw_underdog_all

# ============================================================
# SNAPSHOTS
# ============================================================

def save_market_snapshot(board):
    rows = load_json(PICK_LOG, [])
    existing = set(r.get("id") for r in rows)
    saved = 0
    for b in board:
        if b.get("signal") == "PASS":
            continue
        rid = f"{b['date']}_{b['game_id']}_{b['model_pick']}_ML"
        if rid in existing:
            continue
        rows.append({
            "id": rid, "saved_at": now_iso(), "date": b["date"], "game_id": b["game_id"],
            "away": b["away"], "home": b["home"], "pick": b["model_pick"],
            "prob": round(b["pick_prob"], 4), "price": b["pick_price"],
            "ev": None if b["ev"] is None else round(b["ev"], 4), "kelly": round(b["kelly"], 4),
            "signal": b["signal"], "clv_open": b["pick_price"],
        })
        saved += 1
    save_json(PICK_LOG, rows)
    return saved

def save_prop_snapshot(board):
    rows = load_json(PROP_PICK_LOG, [])
    existing = set(r.get("id") for r in rows)
    saved = 0
    for p in board:
        if p.get("Signal") == "PASS":
            continue
        rid = f"{p['Source']}_{p['date']}_{p['game_id']}_{normalize_name(p['Player'])}_{p['Prop']}_{p['Side']}"
        if rid in existing:
            continue
        rows.append({
            "id": rid, "saved_at": now_iso(), "source": p["Source"], "date": p["date"], "game_id": p["game_id"],
            "player": p["Player"], "team": p["Team"], "prop": p["Prop"], "side": p["Side"],
            "line": p["Line"], "projection": round(p["Projection"], 3), "prob": round(p["Pick Prob"], 4),
            "price": p["Price"], "ev": None if p["EV"] is None else round(p["EV"], 4),
            "kelly": round(p["Kelly"], 4), "signal": p["Signal"],
        })
        saved += 1
    save_json(PROP_PICK_LOG, rows)
    return saved

# ============================================================
# UI
# ============================================================

st.sidebar.markdown(f"""
<div style='padding:10px 4px 18px 4px;'>
  <div style='font-size:28px;font-weight:950;'>😈 DEVIL PICKS</div>
  <div style='color:#ff344f;font-weight:900;'>NBA ENGINE {APP_VERSION}</div>
  <div style='color:#aeb7c9;font-size:12px;margin-top:4px;'>Market system separate from Player Props</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Board Controls")
    day_mode = st.radio("Game Day", ["Today", "Tomorrow", "Both"], index=0)
    bankroll = st.number_input("Bankroll", min_value=10.0, value=1000.0, step=25.0)
    use_market = st.checkbox("Use sportsbook market odds", value=True)
    include_sportsbook_props = st.checkbox("Show sportsbook prop lines", value=True)
    include_underdog = st.checkbox("Show Underdog prop lines", value=True)
    selected_props = st.multiselect("Prop markets", list(PROP_TYPES.keys()), default=["PTS", "REB", "AST", "PRA", "3PM"])
    st.markdown("### Manual Injury Weighting")
    manual_home_adj = st.number_input("Manual home adjustment", value=0.0, step=0.5)
    manual_away_adj = st.number_input("Manual away adjustment", value=0.0, step=0.5)
    st.markdown("### API Status")
    st.write("Odds API:", "✅ key found" if ODDS_API_KEY else "⚠️ no key")
    st.write("XGBoost:", "✅ installed" if XGBOOST_AVAILABLE else "⚠️ fallback mode")

st.markdown("""
<div class='hero'>
  <div class='title'>😈 DEVIL PICKS 9.5 — NBA Market + Underdog Player Prop Engine</div>
  <div class='sub'>NBA CDN + ESPN fallback schedule • separate ML/spread/total system • separate player prop system • Underdog lines • XGBoost/fallback • Bayesian • Markov • Monte Carlo • EV • Kelly • CLV</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    refresh = st.button("🔄 REFRESH FULL BOARD", use_container_width=True)
with c2:
    clear_cache = st.button("🧹 CLEAR CACHE", use_container_width=True)
with c3:
    save_market_btn = st.button("💾 SAVE MARKET SNAPSHOT", use_container_width=True)
with c4:
    save_prop_btn = st.button("💾 SAVE PROP SNAPSHOT", use_container_width=True)

if clear_cache:
    st.cache_data.clear()
    st.success("Cache cleared. Press Refresh Full Board.")

if refresh or "games" not in st.session_state:
    dates = get_dates(day_mode)
    games = []
    for d in dates:
        games.extend(get_nba_games_for_date(d))
    # De-dupe
    seen = set()
    unique_games = []
    for g in games:
        key = g["game_id"]
        if key not in seen:
            seen.add(key)
            unique_games.append(g)
    games = unique_games

    with st.spinner("Building market board."):
        market_board = []
        for g in games:
            try:
                market_board.append(model_market_game(g, bankroll, use_market, manual_home_adj, manual_away_adj))
            except Exception as e:
                log_request("model_market_game", "ERROR", f"{g}: {e}")

    with st.spinner("Building player props, including Underdog lines."):
        prop_board, underdog_raw = build_prop_board(games, selected_props, bankroll, include_underdog, include_sportsbook_props)

    st.session_state["games"] = games
    st.session_state["market_board"] = sorted(market_board, key=lambda x: (not x["qualified"], -safe_float(x.get("ev"), -9), -x.get("pick_prob", 0)))
    st.session_state["prop_board"] = prop_board
    st.session_state["underdog_raw"] = underdog_raw

games = st.session_state.get("games", [])
market_board = st.session_state.get("market_board", [])
prop_board = st.session_state.get("prop_board", [])
underdog_raw = st.session_state.get("underdog_raw", [])

if save_market_btn:
    st.success(f"Saved {save_market_snapshot(market_board)} market snapshots.")
if save_prop_btn:
    st.success(f"Saved {save_prop_snapshot(prop_board)} prop snapshots.")

if not games:
    st.error("No NBA games loaded for this date selection.")
    st.info("This build tries NBA CDN first and ESPN second. Press Clear Cache, then Refresh. If still empty, switch Today/Tomorrow/Both.")
    with st.expander("Debug request log"):
        logs = load_json(REQUEST_LOG_FILE, [])
        st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
    st.stop()

qualified_market = [m for m in market_board if m.get("qualified")]
qualified_props = [p for p in prop_board if p.get("Qualified")]

st.markdown(f"""
<div class='metric-grid'>
  <div class='metric-box'><div class='metric-label'>Games Loaded</div><div class='metric-value'>{len(games)}</div><div class='sub'>NBA/ESPN schedule</div></div>
  <div class='metric-box'><div class='metric-label'>Market Plays</div><div class='metric-value'>{len(qualified_market)}</div><div class='sub'>ML gates passed</div></div>
  <div class='metric-box'><div class='metric-label'>Prop Plays</div><div class='metric-value'>{len(qualified_props)}</div><div class='sub'>Sportsbook + Underdog</div></div>
  <div class='metric-box'><div class='metric-label'>Underdog Lines Parsed</div><div class='metric-value'>{len(underdog_raw)}</div><div class='sub'>Visible in Underdog tab</div></div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["😈 Top Plays", "🏀 Market System", "🎯 Player Props", "🐶 Underdog Lines", "📋 Raw Games", "📈 Trackers", "🔌 Logs"])

with tabs[0]:
    st.markdown("<div class='section-title'>Top Props</div>", unsafe_allow_html=True)
    if qualified_props:
        for p in qualified_props[:12]:
            st.markdown(f"""
            <div class='card good'>
              <div style='font-size:23px;font-weight:950;'>{p['Player']} — {p['Prop Label']}</div>
              <span class='badge'>{p['Source']}</span><span class='badge'>{p['Book']}</span><span class='badge'>{p['away']} @ {p['home']}</span><span class='badge'>{p['Signal']}</span>
              <div class='metric-grid'>
                <div><div class='metric-label'>Pick</div><div class='metric-value green'>{p['Side']}</div></div>
                <div><div class='metric-label'>Line / Projection</div><div class='metric-value'>{p['Line']} / {p['Projection']:.2f}</div></div>
                <div><div class='metric-label'>Probability</div><div class='metric-value'>{p['Pick Prob']*100:.1f}%</div></div>
                <div><div class='metric-label'>EV / Kelly</div><div class='metric-value'>{(p['EV'] or 0)*100:.1f}% / {p['Kelly']*100:.1f}%</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No qualified props yet. Check the Player Props and Underdog Lines tabs for raw lines.")

    st.markdown("<div class='section-title'>Top Market Signals</div>", unsafe_allow_html=True)
    if qualified_market:
        for m in qualified_market[:8]:
            st.markdown(f"""
            <div class='card good'>
              <div style='font-size:23px;font-weight:950;'>{m['away']} @ {m['home']}</div>
              <span class='badge'>{m['signal']}</span><span class='badge'>Odds: {m['odds_quality']}</span><span class='badge'>Source: {m['source']}</span>
              <div class='metric-grid'>
                <div><div class='metric-label'>ML Pick</div><div class='metric-value green'>{m['model_pick']}</div></div>
                <div><div class='metric-label'>Prob</div><div class='metric-value'>{m['pick_prob']*100:.1f}%</div></div>
                <div><div class='metric-label'>EV</div><div class='metric-value'>{(m['ev'] or 0)*100:.1f}%</div></div>
                <div><div class='metric-label'>Kelly</div><div class='metric-value'>{m['kelly']*100:.1f}%</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No qualified market plays. This is normal when edge gates are strict.")

with tabs[1]:
    for m in market_board:
        st.markdown(f"""
        <div class='card {'good' if m['qualified'] else 'warn'}'>
          <div style='font-size:24px;font-weight:950;'>{m['away']} @ {m['home']}</div>
          <span class='badge'>{m['status']}</span><span class='badge'>{m['source']}</span><span class='badge'>Odds {m['odds_quality']}</span><span class='badge'>{m['xgb_note']}</span>
          <div class='metric-grid'>
            <div><div class='metric-label'>Home ML Prob</div><div class='metric-value'>{m['home_prob']*100:.1f}%</div><div class='sub'>{m['home']} {odds_display(m['home_price'])}</div></div>
            <div><div class='metric-label'>Away ML Prob</div><div class='metric-value'>{m['away_prob']*100:.1f}%</div><div class='sub'>{m['away']} {odds_display(m['away_price'])}</div></div>
            <div><div class='metric-label'>Signal</div><div class='metric-value'>{m['signal']}</div><div class='sub'>EV {(m['ev'] or 0)*100:.1f}%</div></div>
            <div><div class='metric-label'>Spread / Total</div><div class='metric-value' style='font-size:18px;'>{m['spread_pick']} / {m['total_pick']}</div><div class='sub'>Spread {m['spread_line']} • Total {m['total_line']}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander(f"Market odds detail — {m['away']} @ {m['home']}"):
            if m.get("odds_rows"):
                st.dataframe(pd.DataFrame(m["odds_rows"]), use_container_width=True, hide_index=True)
            else:
                st.info("No sportsbook odds matched.")

with tabs[2]:
    if prop_board:
        table = pd.DataFrame([{
            "Source": p["Source"], "Game": f"{p['away']} @ {p['home']}", "Player": p["Player"], "Team": p["Team"],
            "Prop": p["Prop"], "Pick": p["Side"] if p["Signal"] != "PASS" else "PASS",
            "Signal": p["Signal"], "Line": p["Line"], "Projection": round(p["Projection"], 2),
            "Prob %": round(p["Pick Prob"]*100, 1), "EV %": round((p["EV"] or 0)*100, 1),
            "Kelly %": round(p["Kelly"]*100, 2), "Data": p["Data Score"], "CLV": p["CLV"],
        } for p in prop_board])
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.warning("No player props found. Check API key, selected props, or the Underdog Lines tab.")

with tabs[3]:
    st.markdown("<div class='section-title'>Underdog Player Prop Lines</div>", unsafe_allow_html=True)
    if underdog_raw:
        udf = pd.DataFrame(underdog_raw)
        cols = [c for c in ["Player", "Team", "Prop", "Line", "Source", "Book", "Game", "Updated"] if c in udf.columns]
        st.dataframe(udf[cols], use_container_width=True, hide_index=True)
        st.caption("These are displayed separately so Underdog props remain visible even when sportsbook props are empty or when team/game matching is incomplete.")
    else:
        st.warning("No Underdog lines parsed. If Underdog changed its payload, open Logs and check request errors.")

with tabs[4]:
    st.dataframe(pd.DataFrame(games), use_container_width=True, hide_index=True)

with tabs[5]:
    st.markdown("<div class='section-title'>Saved Market Picks</div>", unsafe_allow_html=True)
    picks = load_json(PICK_LOG, [])
    if picks:
        st.dataframe(pd.DataFrame(picks), use_container_width=True, hide_index=True)
    else:
        st.info("No market snapshots saved.")
    st.markdown("<div class='section-title'>Saved Prop Picks</div>", unsafe_allow_html=True)
    ppicks = load_json(PROP_PICK_LOG, [])
    if ppicks:
        st.dataframe(pd.DataFrame(ppicks), use_container_width=True, hide_index=True)
    else:
        st.info("No prop snapshots saved.")

with tabs[6]:
    logs = load_json(REQUEST_LOG_FILE, [])
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
    else:
        st.success("No request errors logged.")
    with st.expander("Injury adjustment JSON format"):
        st.code('{"LAL": -1.5, "BOS": 0.5, "OKC": -0.75}', language="json")
        st.caption(f"Create or edit {INJURY_FILE} in your app storage to apply automatic team injury weights.")
