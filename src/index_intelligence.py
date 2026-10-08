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
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "ohlcv.csv"
SUMMARY = DATA / "index_intelligence_summary.csv"
CONFIG = DATA / "index_intelligence_config.json"

INDEXES = {
    "NIFTY50": ["^NSEI", "NIFTY50", "NIFTY 50"],
    "BANKNIFTY": ["^NSEBANK", "BANKNIFTY", "NIFTY BANK"],
    "NIFTYIT": ["^CNXIT", "NIFTYIT", "NIFTY IT"],
    "NIFTYAUTO": ["^CNXAUTO", "NIFTYAUTO", "NIFTY AUTO"],
    "NIFTYFIN": ["^CNXFIN", "NIFTYFIN", "NIFTY FINANCIAL SERVICES"],
}
INDEX_TICKERS = {"NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "NIFTYIT": "^CNXIT", "NIFTYAUTO": "^CNXAUTO", "NIFTYFIN": "^CNXFIN"}

def _download_index_series(name: str) -> pd.Series:
    ticker = INDEX_TICKERS.get(name)
    if not ticker:
        return pd.Series(dtype=float)
    try:
        raw = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=False)
        if raw.empty or "Close" not in raw.columns:
            return pd.Series(dtype=float)
        s = pd.to_numeric(raw["Close"], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index, errors="coerce").tz_localize(None).normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        if (now.hour, now.minute, now.second) < (15, 30, 0):
            s = s[s.index < pd.Timestamp(now.date())]
        return s
    except Exception as exc:
        print(f"Index data unavailable for {name}: {exc}")
        return pd.Series(dtype=float)

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
            s = _download_index_series(name)
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
    news_score, headline, news_sentiment, news_url, news_reasons = score_news(news)
    breadth_pct, breadth_ratio = _breadth(hist)
    market_score, risk_level, market_confidence, data_quality, index_data_available = _market_score(out, news_score, breadth_pct)
    news_reason = headline
    risk_multiplier = 1.0 if risk_level == "RISK_ON" else 0.55 if risk_level == "RISK_OFF" else 0.80
    config = {"version": "index_intelligence_v3", "market_regime": market,
              "risk_multiplier": risk_multiplier,
              "news_sentiment": news_sentiment, "news_score": news_score,
              "market_reason": news_reason, "market_reason_url": news_url,
              "news_headline": headline, "news_reasons": news_reasons, "market_intelligence_score": market_score, "risk_level": risk_level, "market_confidence": market_confidence, "data_quality": data_quality, "index_data_available": index_data_available, "breadth_pct_above_sma20": breadth_pct}
    CONFIG.write_text(json.dumps(config, indent=2))
    out["news_sentiment"] = news_sentiment
    out["market_reason"] = news_reason
    out["news_headline"] = headline
    out["news_source_url"] = news_url
    out["news_reasons"] = json.dumps(news_reasons)
    out["market_intelligence_score"] = market_score
    out["market_confidence"] = market_confidence
    out["data_quality"] = data_quality
    out["index_data_available"] = index_data_available
    out["risk_level"] = risk_level
    out["breadth_pct_above_sma20"] = breadth_pct
    out.to_csv(SUMMARY, index=False)
    print(json.dumps(config, indent=2))
    return out


def _breadth(hist: pd.DataFrame) -> tuple[float, float]:
    if hist.empty or not {"date", "symbol", "close"}.issubset(hist.columns):
        return 50.0, 0.50
    x = hist.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = x.dropna(subset=["date", "symbol", "close"]).sort_values(["symbol", "date"])
    x["sma20"] = x.groupby("symbol")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    latest = x.groupby("symbol").tail(1).dropna(subset=["sma20"])
    if latest.empty:
        return 50.0, 0.50
    pct = float((latest["close"] > latest["sma20"]).mean() * 100)
    return pct, pct / 100.0

def _market_score(out: pd.DataFrame, news_score: float, breadth_pct: float) -> tuple[float, str, float, str, bool]:
    weights = {"NIFTY50": .40, "BANKNIFTY": .25, "NIFTYFIN": .15, "NIFTYIT": .10, "NIFTYAUTO": .10}
    valid = []
    if not out.empty and {"index", "direction", "confidence"}.issubset(out.columns):
        for _, row in out.iterrows():
            idx = str(row["index"])
            if idx not in weights:
                continue
            conf = pd.to_numeric(row["confidence"], errors="coerce")
            if pd.isna(conf):
                continue
            d = {"BUY": 1.0, "HOLD": 0.0, "AVOID": -1.0}.get(str(row["direction"]), 0.0)
            valid.append((weights[idx], d, float(conf)))
    index_available = bool(valid)
    if index_available:
        denom = sum(w for w, _, _ in valid)
        price_score = 50.0 + 45.0 * sum(w * d * conf for w, d, conf in valid) / max(denom, .01)
    else:
        price_score = None

    components = []
    if price_score is not None:
        components.append((.60, price_score))
    if breadth_pct is not None and np.isfinite(breadth_pct):
        components.append((.20, float(np.clip(breadth_pct, 0, 100))))
    news_component = float(np.clip(50 + news_score * 8, 0, 100))
    components.append((.20, news_component))

    total_w = sum(w for w, _ in components)
    score = float(np.clip(sum(w * v for w, v in components) / max(total_w, .01), 0, 100))
    coverage = total_w
    confidence = float(np.clip(coverage * (.55 + .45 * min(1.0, abs(score - 50) / 50)), 0, 1))
    if coverage < .80:
        confidence *= .85

    if score >= 65:
        level = "RISK_ON"
    elif score <= 35:
        level = "RISK_OFF"
    else:
        level = "NEUTRAL"
    quality = "HIGH" if coverage >= .95 else "MEDIUM" if coverage >= .75 else "LOW"
    return score, level, confidence, quality, index_available

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
    x["news_reasons"] = json.dumps(cfg.get("news_reasons", []))
    x["market_intelligence_score"] = float(cfg.get("market_intelligence_score", 50.0))
    x["market_confidence"] = float(cfg.get("market_confidence", 0.0))
    x["data_quality"] = cfg.get("data_quality", "LOW")
    x["index_data_available"] = bool(cfg.get("index_data_available", False))
    x["risk_level"] = cfg.get("risk_level", "NEUTRAL")
    x["breadth_pct_above_sma20"] = float(cfg.get("breadth_pct_above_sma20", 50.0))
    if "phase4_score" in x.columns:
        x["phase4_score"] = pd.to_numeric(x["phase4_score"], errors="coerce").fillna(0.0) * risk
    if "score" in x.columns:
        x["index_adjusted_score"] = pd.to_numeric(x["score"], errors="coerce").fillna(0.0) * risk
    return x


NEWS_FEEDS = [
    ("Google News", "https://news.google.com/rss/search?q=" + urllib.parse.quote("India Nifty Sensex stock market RBI crude oil FII OR geopolitics when:1d") + "&hl=en-IN&gl=IN&ceid=IN:en"),
    ("Google News", "https://news.google.com/rss/search?q=" + urllib.parse.quote("Indian stock market Nifty today when:1d") + "&hl=en-IN&gl=IN&ceid=IN:en"),
]
EVENTS = {
    "RBI / rates": ({"rbi", "repo", "rate", "rates", "policy", "monetary"}, -1.4),
    "Crude oil": ({"crude", "oil", "brent", "opec"}, -1.3),
    "FII / DII flows": ({"fii", "fpi", "dii", "inflow", "outflow", "foreign"}, 1.0),
    "Geopolitics": ({"war", "iran", "israel", "geopolitical", "conflict", "tariff"}, -1.4),
    "Inflation": ({"inflation", "cpi", "wpi"}, -1.2),
    "US / global markets": ({"fed", "nasdaq", "dow", "s&p", "wall street", "us stocks"}, -0.8),
    "Earnings / growth": ({"earnings", "profit", "revenue", "growth", "guidance", "results"}, 0.9),
    "Currency": ({"rupee", "rupee", "inr", "dollar", "currency"}, -0.7),
}
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

def score_news(items: list[dict]) -> tuple[float, str, str, str, list[dict]]:
    scored = []
    for item in items:
        text = item["headline"].lower()
        bull = sum(1 for w in BULL_WORDS if w in text)
        bear = sum(1 for w in BEAR_WORDS if w in text)
        event_scores = []
        for event, (terms, weight) in EVENTS.items():
            hits = sum(1 for term in terms if term in text)
            if hits:
                event_scores.append((event, weight * min(hits, 2)))
        event = max(event_scores, key=lambda z: abs(z[1]))[0] if event_scores else "General market"
        event_impact = next((v for k, v in event_scores if k == event), 0.0)
        score = (bull - bear) + event_impact
        scored.append((score, item, event, event_impact))
    if not scored:
        return 0.0, "No fresh market headline available", "NEUTRAL", "", []
    scored.sort(key=lambda z: abs(z[0]), reverse=True)
    top = scored[:5]
    score, item, event, _ = top[0]
    sentiment = "BULLISH" if score > 0 else "BEARISH" if score < 0 else "NEUTRAL"
    reasons = [{"event": e, "impact": round(float(s), 2), "headline": i["headline"], "url": i.get("url", "")} for s, i, e, _ in top]
    return float(score), f"{event}: {item['headline']}", sentiment, item.get("url", ""), reasons


if __name__ == "__main__":
    build_index_intelligence()
