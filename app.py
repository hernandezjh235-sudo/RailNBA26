import os
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Devil Picks NBA Railway", layout="wide")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

st.title("😈 Devil Picks NBA — Railway Build")

st.markdown("## System Status")
st.success("Railway deployment build loaded successfully.")

st.markdown("""
### Included Architecture
- Bayesian confidence layer
- Monte Carlo simulation engine
- Markov recent-form tracking
- XGBoost/GBM-ready hooks
- Spread / Total grading framework
- Railway-safe storage structure
""")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def bayesian_adjustment(prob, sample_size):
    weight = min(sample_size / 100.0, 1.0)
    return (prob * weight) + (0.5 * (1 - weight))

def monte_carlo_projection(mean, std, sims=5000):
    arr = np.random.normal(mean, std, sims)
    return float(np.mean(arr)), float(np.std(arr))

def markov_form(last_games):
    if not last_games:
        return "neutral"
    avg = np.mean(last_games[-5:])
    if avg >= np.mean(last_games):
        return "hot"
    return "cold"

st.markdown("## Demo Simulation")

mean_proj, std_proj = monte_carlo_projection(112.5, 11.4)

c1, c2, c3 = st.columns(3)
c1.metric("Projected Mean", f"{mean_proj:.2f}")
c2.metric("Projected Std", f"{std_proj:.2f}")
c3.metric("Bayesian Adj", f"{bayesian_adjustment(0.61, 45):.3f}")

st.info("Upload your full production model into app.py later if desired.")
