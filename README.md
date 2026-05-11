# Devil Picks — Real Rail Pro

Production-style Streamlit app for Railway/GitHub.

## What this build does

- Pulls real prop rows from enabled sources when available.
- Supports Underdog direct feed.
- Supports Underdog/Sleeper through Apify actor feeds.
- Supports Odds API player props as fallback when your Odds API plan includes those markets.
- Does not generate fake props.
- Saves official snapshots.
- Lets you manually grade picks after games.
- Rebuilds learning from graded results.
- Includes Bayesian confidence shrink, Markov recent-form state, Monte Carlo simulation, CLV tracking, and XGBoost/GBM hooks.

## Files

```text
app.py
requirements.txt
Procfile
runtime.txt
packages.txt
README.md
```

## Railway env vars

Minimum Underdog direct only:

```text
ENABLE_UNDERDOG=true
ENABLE_SLEEPER=false
ENABLE_ODDS_API=false
```

Underdog + Sleeper:

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

Optional controls:

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

If Underdog direct changes:

```text
UNDERDOG_API_URL=https://api.underdogfantasy.com/v1/over_under_lines
```

## Deploy

1. Create GitHub repo.
2. Upload all files from this ZIP.
3. In Railway, create a new project from GitHub.
4. Add env vars.
5. Deploy.

## Notes

Sleeper does not provide a stable official public Picks prop-line API through the normal fantasy API, so this build uses an Apify/partner feed path for Sleeper props. If you have another Sleeper feed provider, you can replace the Apify actor env var or add a parser in `app.py`.
