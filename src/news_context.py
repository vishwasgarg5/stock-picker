from __future__ import annotations

"""Context-aware market-news interpretation."""

import re

POSITIVE = {"rise","rises","rose","surge","surges","gain","gains","rally","rallies","jump","jumps","higher","up","inflow","inflows","easing","cut","cuts","lower","falls","fall","decline","declines"}
NEGATIVE = {"drop","drops","fell","fall","falls","decline","declines","lower","down","outflow","outflows","hike","hikes","higher","war","weak","weakens","selling","tightening","inflation","surge"}

EVENTS = {
    "Crude oil": {"terms":{"crude","oil","brent","opec"},"falling":1.0,"rising":-1.0,"sectors":{"airlines":1.0,"paints":0.8,"chemicals":0.6,"tyres":0.6,"oil & gas":-0.8}},
    "RBI / rates": {"terms":{"rbi","repo","rate","rates","policy","monetary"},"falling":0.9,"rising":-0.9,"sectors":{"bank":0.8,"financial":0.8,"real estate":0.7,"auto":0.4}},
    "FII / DII flows": {"terms":{"fii","fpi","dii","inflow","outflow","foreign"},"falling":-0.8,"rising":0.8,"sectors":{}},
    "Geopolitics": {"terms":{"war","iran","israel","geopolitical","conflict","tariff"},"falling":0.0,"rising":-1.0,"sectors":{"defence":0.7}},
    "Inflation": {"terms":{"inflation","cpi","wpi"},"falling":0.7,"rising":-0.9,"sectors":{}},
    "US / global markets": {"terms":{"fed","nasdaq","dow","wall street","us stocks"},"falling":0.7,"rising":-0.7,"sectors":{}},
    "Earnings / growth": {"terms":{"earnings","profit","revenue","growth","guidance","results"},"falling":-0.8,"rising":0.8,"sectors":{}},
    "Currency": {"terms":{"rupee","inr","dollar","currency"},"falling":-0.5,"rising":0.5,"sectors":{"information technology":0.6,"it":0.6,"pharma":0.3}},
}

def _direction(text: str) -> int:
    words=set(re.sub(r"[^a-z0-9\s-]"," ",text.lower()).split())
    pos=sum(w in POSITIVE for w in words); neg=sum(w in NEGATIVE for w in words)
    return 1 if pos>neg else -1 if neg>pos else 0

def classify_headline(headline: str) -> dict:
    text=headline.lower(); direction=_direction(text)
    matches=[]
    for event,spec in EVENTS.items():
        hits=sum(term in text for term in spec["terms"])
        if hits: matches.append((hits,event,spec))
    if not matches:
        return {"event":"General market","market_impact":float(direction),"sector_impacts":{},"direction":direction}
    _,event,spec=max(matches,key=lambda x:x[0])
    if event=="Crude oil":
        impact=1.0 if any(w in text for w in ("falls","fall","drops","drop","lower","declines","decline")) else -1.0 if any(w in text for w in ("rises","rise","surges","surge","higher","jumps","jump")) else 0.0
    elif event=="RBI / rates":
        impact=0.9 if any(w in text for w in ("cut","cuts","easing","lower","lowered")) else -0.9 if any(w in text for w in ("hike","hikes","raise","raises","raised","tightening")) else 0.0
    elif event=="FII / DII flows":
        impact=0.8 if "inflow" in text else -0.8 if "outflow" in text or "selling" in text else 0.0
    elif event=="Inflation":
        impact=0.7 if any(w in text for w in ("falls","fall","lower","eases","easing")) else -0.9 if any(w in text for w in ("rises","rise","higher","surges","surge","accelerates")) else 0.0
    elif event=="Earnings / growth":
        impact=0.8 if direction>0 else -0.8 if direction<0 else 0.0
    elif event=="US / global markets":
        impact=0.7 if direction>0 else -0.7 if direction<0 else 0.0
    elif event=="Currency":
        # Broad-market interpretation: a stronger INR is supportive; a weaker INR is a drag.
        impact=0.5 if any(w in text for w in ("strengthens","stronger","gains","rises")) else -0.5 if any(w in text for w in ("weakens","weaker","falls","drops")) else 0.0
    else:
        impact=spec["rising"] if direction>0 else spec["falling"] if direction<0 else 0.0
    return {"event":event,"market_impact":float(impact),"sector_impacts":dict(spec["sectors"]),"direction":direction}

def sector_impacts(headlines: list[dict]) -> dict[str,float]:
    totals={}
    for item in headlines:
        for sector,impact in classify_headline(item.get("headline",""))["sector_impacts"].items():
            totals[sector]=totals.get(sector,0.0)+float(impact)
    return totals
