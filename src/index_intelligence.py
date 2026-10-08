from __future__ import annotations

"""Index intelligence layer.

Builds leakage-safe, market-level regime and forecast features from the existing
OHLCV universe. Index signals are advisory: they adjust downstream stock
selection risk rather than replacing the V1 production champion.
"""

from pathlib import Path
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "ohlcv.csv"
SUMMARY = DATA / "index_intelligence_summary.csv"
CONFIG = DATA / "index_intelligence_config.json"

INDEXES = {
    "NIFTY50": ["^NSEI", "NIFTY50", "NIFTY 50"],
    "BANKNIFTY": ["^NSEBANK", "BANKNIFTY", "NIFTY BANK"],
    "NIFTYIT": ["^CNXIT", "NIFTYIT", "NIFTY IT"],
    "NIFTYAUTO": ["NIFTYAUTO", "NIFTY AUTO"],
    "NIFTYFIN": ["NIFTYFIN", "NIFTY FINANCIAL SERVICES"],
}

def _series(hist: pd.DataFrame, names: list[str]) -> pd.Series:
    if hist.empty or not {"date", "symbol", "close"}.issubset(hist.columns):
        return pd.Series(dtype=float)
    x = hist.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    y = x[x["symbol"].isin([n.upper() for n in names])].dropna(subset=["date", "close"])
    if y.empty:
        return pd.Series(dtype=float)
    return y.groupby("date")["close"].last().sort_index()

def _signal(s: pd.Series) -> dict:
    if len(s) < 50:
        return {"regime": "NEUTRAL", "direction": "HOLD", "confidence": 0.50,
                "expected_return_pct": 0.0, "return_5d_pct": np.nan, "return_20d_pct": np.nan}
    s20, s50 = s.rolling(20).mean().iloc[-1], s.rolling(50).mean().iloc[-1]
    r5 = float((s.iloc[-1] / s.iloc[-6] - 1) * 100)
    r20 = float((s.iloc[-1] / s.iloc[-21] - 1) * 100)
    vol = float(s.pct_change().rolling(20).std().iloc[-1] * 100)
    bull = s.iloc[-1] > s20 > s50 and r20 >= 1.5
    bear = s.iloc[-1] < s20 < s50 and r20 <= -1.5
    regime = "BULL" if bull else "BEAR" if bear else "NEUTRAL"
    direction = "BUY" if r5 > 0.5 and r20 > 0 else "AVOID" if r5 < -0.5 and r20 < 0 else "HOLD"
    alignment = 1.0 if regime != "NEUTRAL" and ((direction == "BUY" and regime == "BULL") or (direction == "AVOID" and regime == "BEAR")) else 0.0
    confidence = float(np.clip(0.50 + min(abs(r20) / 10, 0.25) + alignment * 0.15, 0.50, 0.90))
    return {"regime": regime, "direction": direction, "confidence": confidence,
            "expected_return_pct": float(np.clip(r20 * 0.25 + r5 * 0.15, -5, 5)),
            "return_5d_pct": r5, "return_20d_pct": r20, "volatility20_pct": vol}

def build_index_intelligence(hist: pd.DataFrame | None = None) -> pd.DataFrame:
    hist = hist if hist is not None else (pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame())
    rows = []
    for name, aliases in INDEXES.items():
        s = _series(hist, aliases)
        if s.empty:
            continue
        r = _signal(s)
        r.update({"index": name, "as_of": str(s.index[-1].date())})
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["index", "regime", "direction", "confidence", "expected_return_pct", "return_5d_pct", "return_20d_pct", "volatility20_pct", "as_of"])
    out.to_csv(SUMMARY, index=False)
    market = "NEUTRAL"
    if not out.empty:
        bull = (out["regime"] == "BULL").mean()
        bear = (out["regime"] == "BEAR").mean()
        market = "BULL" if bull >= 0.60 else "BEAR" if bear >= 0.60 else "NEUTRAL"
    news = fetch_market_news()
    news_score, headline, news_sentiment, news_url = score_news(news)
    news_reason = headline
    config = {"version": "index_intelligence_v2", "market_regime": market,
              "risk_multiplier": 1.0 if market == "BULL" else 0.80 if market == "NEUTRAL" else 0.55,
              "news_sentiment": news_sentiment, "news_score": news_score,
              "market_reason": news_reason, "market_reason_url": news_url,
              "news_headline": headline}
    CONFIG.write_text(json.dumps(config, indent=2))
    out["news_sentiment"] = news_sentiment
    out["market_reason"] = news_reason
    out["news_headline"] = headline
    out["news_source_url"] = news_url
    out.to_csv(SUMMARY, index=False)
    print(json.dumps(config, indent=2))
    return out

def apply_index_overlay(candidates: pd.DataFrame) -> pd.DataFrame:
    x = candidates.copy()
    if x.empty:
        return x
    try:
        cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    except Exception:
        cfg = {}
    regime = cfg.get("market_regime", "NEUTRAL")
    risk = float(cfg.get("risk_multiplier", 0.80))
    x["index_market_regime"] = regime
    x["index_risk_multiplier"] = risk
    x["market_reason"] = cfg.get("market_reason", "No fresh market headline available")
    x["news_sentiment"] = cfg.get("news_sentiment", "NEUTRAL")
    x["news_headline"] = cfg.get("news_headline", "No fresh market headline available")
    if "phase4_score" in x.columns:
        x["phase4_score"] = pd.to_numeric(x["phase4_score"], errors="coerce").fillna(0.0) * risk
    if "score" in x.columns:
        x["index_adjusted_score"] = pd.to_numeric(x["score"], errors="coerce").fillna(0.0) * risk
    return x

if __name__ == "__main__":
    build_index_intelligence()


NEWS_FEEDS = [
    ("Google News", "https://news.google.com/rss/search?q=" + urllib.parse.quote("India Nifty Sensex stock market RBI crude oil FII OR geopolitics when:1d") + "&hl=en-IN&gl=IN&ceid=IN:en"),
    ("Google News", "https://news.google.com/rss/search?q=" + urllib.parse.quote("Indian stock market Nifty today when:1d") + "&hl=en-IN&gl=IN&ceid=IN:en"),
]
BULL_WORDS = {"surge", "gain", "gains", "rally", "rise", "rises", "bullish", "upgrade", "inflow", "growth", "strong", "easing", "cut", "cuts", "record high"}
BEAR_WORDS = {"fall", "falls", "drop", "drops", "decline", "declines", "bearish", "outflow", "inflation", "hike", "hikes", "war", "crude", "oil", "weak", "selling", "tightening", "yield", "risk"}

def fetch_market_news(limit: int = 12) -> list[dict]:
    items = []
    for source, url in NEWS_FEEDS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "stock-picker/1.0"})
            root = ET.fromstring(urllib.request.urlopen(req, timeout=10).read())
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                if title:
                    items.append({"source": source, "headline": re.sub(r"\\s+", " ", title), "url": link, "published": pub})
        except Exception as exc:
            print(f"News feed unavailable: {exc}")
    unique = []
    seen = set()
    for item in items:
        key = item["headline"].lower()
        if key not in seen:
            seen.add(key); unique.append(item)
    return unique[:limit]

def score_news(items: list[dict]) -> tuple[float, str, str, str]:
    scored = []
    for item in items:
        text = item["headline"].lower()
        bull = sum(1 for w in BULL_WORDS if w in text)
        bear = sum(1 for w in BEAR_WORDS if w in text)
        scored.append((bull - bear, item))
    if not scored:
        return 0.0, "No fresh market headline available", "NEUTRAL", ""
    scored.sort(key=lambda z: abs(z[0]), reverse=True)
    score, item = scored[0]
    sentiment = "BULLISH" if score > 0 else "BEARISH" if score < 0 else "NEUTRAL"
    return float(score), item["headline"], sentiment, item.get("url", "")
