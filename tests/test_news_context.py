from src.news_context import classify_headline

def test_crude_falls_is_contextually_positive_for_oil_users():
    r=classify_headline("Crude oil falls sharply as supply concerns ease")
    assert r["event"]=="Crude oil"
    assert r["market_impact"]>0
    assert r["sector_impacts"]["airlines"]>0

def test_rbi_rate_cut_is_contextually_supportive():
    r=classify_headline("RBI cuts repo rate to support growth")
    assert r["event"]=="RBI / rates"
    assert r["market_impact"]>0
    assert r["sector_impacts"]["bank"]>0
