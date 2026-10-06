    ipo_rows = latest.loc[~latest["symbol"].isin(core_latest["symbol"])].copy()
    spreads = []
    for name in TARGETS:
        bundle = joblib.load(MODELS / f"{name}.joblib")
        pred, spread = _ensemble_predict(bundle, core_latest[FEATURE_COLUMNS])
        core_latest[f"predicted_{name}"] = core_latest["close"] * (1 + pred)
        spreads.append(spread)
    core_latest["prediction_spread"] = np.mean(np.column_stack(spreads), axis=1) if spreads else 0.0
    latest = pd.concat([core_latest, ipo_rows], ignore_index=True, sort=False)
    latest["predicted_high"] = latest[["predicted_high", "predicted_open", "predicted_close"]].max(axis=1)
    latest["predicted_low"] = latest[["predicted_low", "predicted_open", "predicted_close"]].min(axis=1)
    out = latest[["date", "symbol", "close", "predicted_open", "predicted_high", "predicted_low", "predicted_close", "prediction_spread"]].copy().rename(columns={"date": "prediction_date", "close": "base_close"})
    lookup = ranked.set_index("symbol")
    out["rank"] = out["symbol"].map(lookup["rank"]).astype(int)
    out["score"] = out["symbol"].map(lookup["total_score"])
    out["technical_score"] = out["symbol"].map(lookup["technical_score"])
    out["fundamental_score"] = out["symbol"].map(lookup["fundamental_score"])
    # Confidence v2 is based on ensemble forecast uncertainty rather than
    # predicted-move magnitude. Lower spread means stronger agreement between
    # independent ensemble models and therefore higher empirical confidence.
    out["prediction_spread"] = pd.to_numeric(out["prediction_spread"], errors="coerce").abs()
    spread_rank = out["prediction_spread"].rank(method="average", pct=True)
    out["confidence_score"] = ((1.0 - spread_rank).clip(0.0, 1.0) * 100.0).fillna(0.0)
    out["confidence_calibration_version"] = "uncertainty_rank_v2"

    # New listings get a separate maturity-aware confidence score. This does not
    # change the core ML prediction; it prevents short post-IPO histories from
    # being interpreted as equally reliable as established stocks.
    out["listing_age_days"] = np.nan
    out["listing_status"] = "CORE"
    out["listing_confidence_factor"] = 1.0
    try:
        if NEW_LISTINGS_FILE.exists():
            listing = pd.read_csv(NEW_LISTINGS_FILE)
            if {"symbol", "calendar_age", "status"}.issubset(listing.columns):
                listing["symbol"] = listing["symbol"].astype(str).str.upper().str.strip()
                listing["calendar_age"] = pd.to_numeric(listing["calendar_age"], errors="coerce")
                meta = listing.drop_duplicates("symbol").set_index("symbol")
                out["listing_age_days"] = out["symbol"].map(meta["calendar_age"])
                out["listing_status"] = out["symbol"].map(meta["status"]).fillna("CORE")
                out["listing_confidence_factor"] = out["listing_age_days"].map(
                    lambda age: 0.70 if pd.notna(age) and age < 20
                    else 0.85 if pd.notna(age) and age < 60
                    else 1.0
                )
    except Exception as exc:
        print(f"New-listing confidence adjustment unavailable: {exc}")
    out["adjusted_confidence_score"] = (
        out["confidence_score"] * out["listing_confidence_factor"]
    ).clip(0, 100)

    out["target_date"] = pd.Timestamp(target_date).normalize()
    out["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    candidates = out.sort_values("rank").copy()
    selection_method = "ranking_top10"
    selected_symbols = set(candidates.head(10)["symbol"])

    # Apply the recent-error selector only after its own out-of-sample
    # A/B validation gate has demonstrated improvement over pure rank selection.
    # Until then, production stays on the safer rank-based Top-10.
    try:
        validation = pd.read_csv(SELECTION_VALIDATION_FILE) if SELECTION_VALIDATION_FILE.exists() else pd.DataFrame()
        selector_validated = (
            not validation.empty
            and "recent_error_promotion_evidence" in validation.columns
            and bool(validation["recent_error_promotion_evidence"].fillna(False).astype(bool).any())
        )
        if selector_validated and EVALUATIONS_FILE.exists():
            ev = pd.read_csv(EVALUATIONS_FILE)
            ev["target_date"] = pd.to_datetime(ev["target_date"], errors="coerce").dt.normalize()
            ev["close_abs_pct_error"] = pd.to_numeric(ev["close_abs_pct_error"], errors="coerce")
            ev = ev.dropna(subset=["target_date", "symbol", "close_abs_pct_error"])
            if not ev.empty:
                cutoff = ev["target_date"].max() - pd.Timedelta(days=21)
                recent = ev[ev["target_date"] >= cutoff]
                if len(recent) >= 30:
                    recent_error = recent.groupby("symbol")["close_abs_pct_error"].mean()
                    candidates["recent_close_error"] = candidates["symbol"].map(recent_error)
                    median_error = float(recent_error.median())
                    excess = ((candidates["recent_close_error"] / max(median_error, 1e-6)) - 1.0).clip(lower=0, upper=2)
                    candidates["selection_penalty"] = (excess * 0.75).fillna(0.0)
                    # No recent history means no evidence of persistent error; do not penalize it.
                    candidates["selection_priority"] = candidates["rank"] + candidates["selection_penalty"]
                    selected_symbols = set(candidates.sort_values(["selection_priority", "rank"]).head(10)["symbol"])
                    selection_method = "ranking_recent_error_adjusted"