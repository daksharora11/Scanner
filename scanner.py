#!/usr/bin/env python3
"""
Bull & Bear Ledger - scanner.

PAPER TRADING / EDUCATION ONLY. Not financial advice.

What it does each run:
  1. Pulls price history + latest intraday price for every ticker in config.json (Yahoo Finance via yfinance).
  2. Computes 50/200-day MAs, RSI(14), ATR(14), relative volume, support/resistance.
  3. Scores every ticker for a long setup and a short setup (0-100, transparent point system).
  4. Builds a paper-trade plan (entry zone, stop, two targets, R/R, position size).
  5. For the top ideas: pulls fundamentals from SEC EDGAR (10-K XBRL data), earnings date and headlines.
  6. Writes docs/data/latest.json, which docs/index.html displays.

Anything it cannot verify is written as "NOT FOUND" - it never estimates.

Usage:
  python scanner.py            # live run
  python scanner.py --demo     # synthetic data, to preview the dashboard
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT_FILE = ROOT / "docs" / "data" / "latest.json"
FUND_CACHE = ROOT / "docs" / "data" / "fundamentals_cache.json"
ET = ZoneInfo("America/New_York")
NF = "NOT FOUND"

MACRO = {
    "S&P 500": "^GSPC",
    "VIX": "^VIX",
    "US 10Y yield %": "^TNX",
    "WTI crude": "CL=F",
    "Brent crude": "BZ=F",
    "Gold": "GC=F",
    "USD/CAD": "CAD=X",
}


# --------------------------------------------------------------------------- helpers
def _f(x, nd=2):
    """float or None (NaN/inf -> None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, nd)


def clean(o):
    """Make an object JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load_config(path=None):
    with open(path or ROOT / "config.json") as fh:
        return json.load(fh)


def market_state(now_et):
    """(is_open, fraction_of_session_elapsed). Ignores exchange holidays."""
    if now_et.weekday() >= 5:
        return False, 0.0
    o = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    c = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    if o <= now_et <= c:
        return True, (now_et - o).total_seconds() / 60.0 / 390.0
    return False, 0.0


# --------------------------------------------------------------------------- indicators
def rsi_wilder(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    r = 100 - 100 / (1 + rs)
    return r.where(ad != 0, 100.0)


def atr_wilder(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_indicators(df, live_price=None, now_et=None, market_open=False, frac=0.0):
    """df: daily OHLCV (index = dates). Returns dict of indicators or None."""
    df = df.dropna(subset=["Close", "High", "Low"]).copy()
    if len(df) < 60:
        return None
    now_et = now_et or datetime.now(ET)
    today = now_et.date()
    last_date = pd.Timestamp(df.index[-1]).date()
    today_in_df = last_date == today

    close = df["Close"].astype(float)
    price = float(live_price) if live_price and live_price > 0 else float(close.iloc[-1])

    # previous close for day-change
    if market_open and not today_in_df:
        prev_close = float(close.iloc[-1])
    else:
        prev_close = float(close.iloc[-2])

    # close series with live price applied (so RSI / MAs are current)
    c2 = close.copy()
    if live_price and live_price > 0:
        if today_in_df:
            c2.iloc[-1] = price
        else:
            c2.loc[pd.Timestamp(today)] = price

    sma50 = c2.rolling(50).mean().iloc[-1] if len(c2) >= 50 else np.nan
    sma200 = c2.rolling(200).mean().iloc[-1] if len(c2) >= 200 else np.nan
    rsi = rsi_wilder(c2).iloc[-1]
    atr = atr_wilder(df).iloc[-1]

    # completed sessions only (exclude today's partial bar)
    completed = df.iloc[:-1] if (market_open and today_in_df) else df
    avg_vol20 = completed["Volume"].tail(20).mean()
    prev_rel = (completed["Volume"].iloc[-1] / avg_vol20) if avg_vol20 else np.nan
    rel_vol, rel_src = prev_rel, "last completed session"
    if market_open and today_in_df and frac >= 0.15 and avg_vol20:
        rel_vol = df["Volume"].iloc[-1] / (avg_vol20 * frac)
        rel_src = "today, projected from elapsed session time"

    prior = completed.tail(20)
    hi20, lo20 = prior["High"].max(), prior["Low"].min()
    yr = df.tail(252)
    hi52 = max(yr["High"].max(), price)
    lo52 = min(yr["Low"].min(), price)

    def ret(n):
        return price / c2.iloc[-(n + 1)] - 1 if len(c2) > n else np.nan

    rng = hi52 - lo52
    return {
        "price": price,
        "prev_close": prev_close,
        "change_pct": price / prev_close - 1,
        "sma50": _f(sma50, 4),
        "sma200": _f(sma200, 4),
        "rsi": _f(rsi, 1),
        "atr": _f(atr, 4),
        "atr_pct": _f(atr / price * 100, 2) if atr else None,
        "avg_vol20": _f(avg_vol20, 0),
        "avg_dollar_vol": _f(avg_vol20 * price, 0) if avg_vol20 else None,
        "rel_vol": _f(rel_vol, 2),
        "rel_vol_src": rel_src,
        "hi20": _f(hi20, 4),
        "lo20": _f(lo20, 4),
        "hi52": _f(hi52, 4),
        "lo52": _f(lo52, 4),
        "pct_from_hi52": _f(price / hi52 - 1, 4),
        "range_pos": _f((price - lo52) / rng, 3) if rng > 0 else None,
        "ret5": _f(ret(5), 4),
        "ret20": _f(ret(20), 4),
        "ret63": _f(ret(63), 4),
        "day_up": price > prev_close,
    }


# --------------------------------------------------------------------------- trade plan
def build_plan(ind, direction, cfg):
    """Paper-trade plan. Every level carries its reasoning."""
    p, a = ind["price"], ind["atr"]
    if not a or a <= 0:
        return None
    sign = 1 if direction == "long" else -1
    mult = cfg.get("atr_stop_mult", 1.5)

    # ---- stop
    if direction == "long":
        below = [x for x in (ind["lo20"], ind["sma50"], ind["sma200"]) if x and x < p]
        support = max(below) if below else None
        struct_stop = support - 0.25 * a if support else None
        dist = p - struct_stop if struct_stop else None
        near_level = support
    else:
        above = [x for x in (ind["hi20"], ind["sma50"], ind["sma200"]) if x and x > p]
        resistance = min(above) if above else None
        struct_stop = resistance + 0.25 * a if resistance else None
        dist = struct_stop - p if struct_stop else None
        near_level = resistance
    if struct_stop and 0.75 * a <= dist <= 2.5 * a:
        stop = struct_stop
        stop_why = (
            f"Just beyond the nearest {'support' if direction == 'long' else 'resistance'} "
            f"at {near_level:.2f} with a 0.25 ATR buffer"
        )
    else:
        stop = p - sign * mult * a
        stop_why = f"{mult} x ATR(14) = {mult * a:.2f} from entry (no clean level within 0.75-2.5 ATR)"
    risk = abs(p - stop)
    if risk <= 0:
        return None

    # ---- targets
    if direction == "long":
        cands = [x for x in (ind["hi20"], ind["hi52"], ind["sma50"], ind["sma200"]) if x and x > p]
        struct_t = min(cands) if cands else None
    else:
        cands = [x for x in (ind["lo20"], ind["lo52"], ind["sma50"], ind["sma200"]) if x and x < p]
        struct_t = max(cands) if cands else None
    t1_struct = bool(struct_t and abs(struct_t - p) >= 1.25 * risk)
    if t1_struct:
        t1 = struct_t
        t1_why = f"Nearest {'resistance' if direction == 'long' else 'support'} at {struct_t:.2f} (at least 1.25R away)"
    else:
        t1 = p + sign * 1.5 * risk
        t1_why = "1.5R - no chart level at least 1.25R away, so a measured move"
    t2 = p + sign * 3.0 * risk
    t2_why = "3R measured move from the risk taken"
    if (t2 - t1) * sign <= 0:
        t2 = t1 + sign * risk
        t2_why = "Target 1 plus one more R (3R would sit inside Target 1)"
    rr1, rr2 = abs(t1 - p) / risk, abs(t2 - p) / risk

    # ---- size
    risk_usd = cfg["account_size"] * cfg["risk_pct"] / 100.0
    raw_sh = math.floor(risk_usd / risk)
    max_sh = math.floor(cfg["account_size"] * cfg.get("max_position_pct", 25) / 100.0 / p)
    shares = max(0, min(raw_sh, max_sh))
    capped = raw_sh > max_sh
    math_txt = f"${risk_usd:,.0f} max risk / ${risk:.2f} risk per share = {raw_sh:,} shares"
    if capped:
        math_txt += (
            f"; capped at {max_sh:,} shares so the position stays under "
            f"{cfg.get('max_position_pct', 25)}% of the account"
        )
    zone = (p - 0.25 * a, p) if direction == "long" else (p, p + 0.25 * a)
    return {
        "direction": direction,
        "entry": _f(p, 4),
        "entry_zone": [_f(zone[0], 4), _f(zone[1], 4)],
        "stop": _f(stop, 4),
        "stop_why": stop_why,
        "risk_per_share": _f(risk, 4),
        "t1": _f(t1, 4),
        "t1_why": t1_why,
        "t1_structural": t1_struct,
        "t2": _f(t2, 4),
        "t2_why": t2_why,
        "rr1": _f(rr1, 2),
        "rr2": _f(rr2, 2),
        "rr_math": f"T1: |{t1:.2f} - {p:.2f}| / |{p:.2f} - {stop:.2f}| = {rr1:.2f}R;  T2: {rr2:.2f}R",
        "shares": int(shares),
        "position_value": _f(shares * p, 0),
        "capped": capped,
        "sizing_math": math_txt,
        "horizon": "Swing: about 2-4 weeks (opinion; exit early if the stop or an invalidation triggers)",
    }


# --------------------------------------------------------------------------- scoring
def score_components(ind, plans):
    """Return (long_points, short_points) as lists of (points, text)."""
    L, S = [], []
    p, s50, s200, r = ind["price"], ind["sma50"], ind["sma200"], ind["rsi"]
    rv = ind["rel_vol"]

    if s50:
        (L if p > s50 else S).append((15, f"Price {'above' if p > s50 else 'below'} 50-day MA ({s50:.2f})"))
    if s200:
        (L if p > s200 else S).append((10, f"Price {'above' if p > s200 else 'below'} 200-day MA ({s200:.2f})"))
    if s50 and s200:
        if s50 > s200:
            L.append((10, "50-day MA above 200-day MA (uptrend structure)"))
        else:
            S.append((10, "50-day MA below 200-day MA (downtrend structure)"))
    if ind["ret20"] is not None:
        (L if ind["ret20"] > 0 else S).append((8, f"20-day return {ind['ret20'] * 100:+.1f}%"))
    if ind["ret5"] is not None:
        (L if ind["ret5"] > 0 else S).append((5, f"5-day return {ind['ret5'] * 100:+.1f}%"))

    if r is not None:
        if 50 <= r <= 68:
            L.append((12, f"RSI {r:.0f}: bullish momentum, not overbought"))
        elif 40 <= r < 50:
            L.append((4, f"RSI {r:.0f}: neutral, room to rise"))
        elif r > 75:
            L.append((-10, f"RSI {r:.0f}: overbought, chasing risk for a long"))
        if r < 30:
            L.append((6, f"RSI {r:.0f}: oversold, bounce candidate"))
        if 32 <= r <= 50:
            S.append((12, f"RSI {r:.0f}: bearish momentum, not yet oversold"))
        elif r < 25:
            S.append((-10, f"RSI {r:.0f}: deeply oversold, squeeze risk for a short"))

    if s50 and s200 and p > s200 and p < s50 and r is not None and 35 <= r <= 50:
        L.append((12, "Pullback inside an uptrend (above 200-day, below 50-day, RSI 35-50)"))

    if rv is not None:
        if rv >= 1.5:
            (L if ind["day_up"] else S).append((10, f"Volume {rv:.1f}x average on a {'up' if ind['day_up'] else 'down'} day"))
        elif rv >= 1.2:
            (L if ind["day_up"] else S).append((5, f"Volume {rv:.1f}x average on a {'up' if ind['day_up'] else 'down'} day"))

    if ind["pct_from_hi52"] is not None and ind["pct_from_hi52"] >= -0.05:
        L.append((10, f"Within {abs(ind['pct_from_hi52']) * 100:.1f}% of its 52-week high"))
    if ind["lo52"] and p <= ind["lo52"] * 1.05:
        S.append((10, "Within 5% of its 52-week low"))

    for d, lst in (("long", L), ("short", S)):
        pl = plans.get(d)
        if pl and pl["t1_structural"] and pl["rr1"] >= 2:
            lst.append((10, f"Chart level gives {pl['rr1']:.1f}R of room to Target 1"))
    return L, S


def finalize(points):
    raw = sum(pts for pts, _ in points)
    return int(max(0, min(100, round(raw / 90 * 100))))


def make_flags(ind, direction, earnings_days=None):
    f = []
    if ind["avg_dollar_vol"] is not None and ind["avg_dollar_vol"] < 20_000_000:
        f.append("Thin liquidity: under $20M average daily dollar volume")
    if abs(ind["change_pct"]) >= 0.05:
        f.append(f"Big move today ({ind['change_pct'] * 100:+.1f}%): chasing risk")
    if ind["rsi"] is not None and ind["rsi"] > 75 and direction == "long":
        f.append("RSI above 75 on a long idea")
    if ind["rsi"] is not None and ind["rsi"] < 25 and direction == "short":
        f.append("RSI below 25 on a short idea")
    if earnings_days is not None and 0 <= earnings_days <= 5:
        f.append(f"Earnings in {earnings_days} day(s): gap risk can jump past any stop")
    return f


def analyse(t, name, df, live, cfg, now_et, market_open, frac):
    ind = compute_indicators(df, live_price=live, now_et=now_et, market_open=market_open, frac=frac)
    if not ind:
        return None
    plans = {d: build_plan(ind, d, cfg) for d in ("long", "short")}
    if not plans["long"] or not plans["short"]:
        return None
    L, S = score_components(ind, plans)
    ls, ss = finalize(L), finalize(S)
    direction = "long" if ls >= ss else "short"
    mine, theirs = (L, S) if direction == "long" else (S, L)
    plan = plans[direction]
    trend_ok = ind["sma50"] and ind["sma200"]
    if direction == "long":
        if any("Pullback" in tx for _, tx in L):
            setup = "Pullback in uptrend"
        elif trend_ok and ind["price"] > ind["sma50"] and ind["price"] > ind["sma200"]:
            setup = "Trend continuation"
        elif ind["rsi"] is not None and ind["rsi"] < 30:
            setup = "Oversold bounce"
        else:
            setup = "Mixed signals"
    else:
        setup = "Downtrend continuation" if trend_ok and ind["price"] < ind["sma50"] and ind["price"] < ind["sma200"] else "Mixed signals"

    bull_pts = L if direction == "long" else S
    bear_pts = S if direction == "long" else L
    bull = [tx for pts, tx in sorted(bull_pts, reverse=True) if pts > 0]
    bear = [tx for pts, tx in sorted(bear_pts, reverse=True) if pts > 0]
    bear += [tx for pts, tx in bull_pts if pts < 0]  # penalties against the chosen side
    reasons = bull[:5]

    rp = ind["range_pos"]
    priced_in = (
        f"Up/down {ind['ret63'] * 100:+.0f}% over ~3 months; sits at "
        f"{rp * 100:.0f}% of its 52-week range" if ind["ret63"] is not None and rp is not None else NF
    )
    invalid = [f"Price closes beyond the stop at {plan['stop']:.2f}"]
    if direction == "long" and ind["sma200"] and ind["price"] > ind["sma200"]:
        invalid.append(f"Daily close below the 200-day MA ({ind['sma200']:.2f})")
    if direction == "short" and ind["sma200"] and ind["price"] < ind["sma200"]:
        invalid.append(f"Daily close above the 200-day MA ({ind['sma200']:.2f})")
    invalid.append("Company news (earnings, guidance, executive or regulatory change) that these price signals cannot see")
    invalid.append("A macro shock: oil, yields or rate-decision surprises")

    edge = abs(ls - ss)
    conviction = max(ls, ss)
    flags = make_flags(ind, direction)
    return {
        "ticker": t,
        "name": name,
        "currency": "CAD" if t.endswith(".TO") else "USD",
        "market": "TSX" if t.endswith(".TO") else "US",
        "price": _f(ind["price"], 4),
        "change_pct": _f(ind["change_pct"], 4),
        "direction": direction,
        "long_score": ls,
        "short_score": ss,
        "conviction": conviction,
        "edge": edge,
        "setup": setup,
        "indicators": {k: v for k, v in ind.items() if k not in ("price", "prev_close", "change_pct", "day_up")},
        "plan": plan,
        "reasons": reasons,
        "bull": bull,
        "bear": bear,
        "priced_in": priced_in,
        "invalidation": invalid,
        "flags": flags,
        "actionable": bool(
            conviction >= cfg.get("min_conviction", 60)
            and plan["rr1"] >= 1.5
            and not any(fl.startswith("Thin liquidity") for fl in flags)
        ),
    }


# --------------------------------------------------------------------------- market data (yfinance)
def split_batch(raw, tickers):
    out = {}
    if raw is None or getattr(raw, "empty", True):
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        top = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t in top:
                d = raw[t].dropna(how="all")
                if not d.empty:
                    out[t] = d
    elif len(tickers) == 1:
        out[tickers[0]] = raw.dropna(how="all")
    return out


def fetch_daily(tickers, period="1y"):
    import yfinance as yf

    for attempt in range(3):
        try:
            raw = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                              auto_adjust=True, progress=False, threads=True)
            res = split_batch(raw, tickers)
            if res:
                return res
        except Exception as e:  # noqa: BLE001
            print(f"daily download attempt {attempt + 1} failed: {e}", file=sys.stderr)
        time.sleep(2 * (attempt + 1))
    return {}


def fetch_latest(tickers):
    """{ticker: (price, timestamp_et_str, timestamp_utc)} from 5-minute bars."""
    import yfinance as yf

    out = {}
    try:
        raw = yf.download(tickers, period="1d", interval="5m", group_by="ticker",
                          progress=False, threads=True, prepost=False)
        for t, d in split_batch(raw, tickers).items():
            d = d.dropna(subset=["Close"])
            if d.empty:
                continue
            ts = pd.Timestamp(d.index[-1])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
            out[t] = (float(d["Close"].iloc[-1]), ts.tz_convert(ET).strftime("%Y-%m-%d %H:%M ET"), ts.to_pydatetime())
    except Exception as e:  # noqa: BLE001
        print(f"intraday download failed: {e}", file=sys.stderr)
    return out


def get_earnings(t):
    """(iso_date or None, days_until or None)."""
    try:
        import yfinance as yf

        cal = yf.Ticker(t).calendar
        d = None
        if isinstance(cal, dict):
            v = cal.get("Earnings Date")
            d = v[0] if isinstance(v, (list, tuple)) and v else v
        elif hasattr(cal, "loc") and "Earnings Date" in getattr(cal, "index", []):
            d = cal.loc["Earnings Date"].iloc[0]
        if d is None:
            return None, None
        d = pd.Timestamp(d).date()
        return d.isoformat(), (d - datetime.now(ET).date()).days
    except Exception:  # noqa: BLE001
        return None, None


def get_news(t, n=3):
    try:
        import yfinance as yf

        out = []
        for it in (yf.Ticker(t).news or [])[:n]:
            c = it.get("content") or it
            title = c.get("title")
            if not title:
                continue
            pub = c.get("pubDate") or it.get("providerPublishTime")
            if isinstance(pub, (int, float)):
                pub = datetime.fromtimestamp(pub, timezone.utc).isoformat()
            src = (c.get("provider") or {}).get("displayName") or it.get("publisher") or NF
            url = ((c.get("canonicalUrl") or {}).get("url")) or it.get("link")
            out.append({"title": title, "source": src, "published": pub or NF, "url": url})
        return out
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- SEC EDGAR fundamentals
FLOW = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
}
INSTANT = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
}
SHARES = ["WeightedAverageNumberOfDilutedSharesOutstanding"]


def series_for(gaap, concepts, kind, unit="USD"):
    """Latest 4 annual (10-K) values, newest first. Picks the tag with the most recent data."""
    best_tag, best = None, {}
    for c in concepts:
        node = gaap.get(c)
        if not node:
            continue
        found = {}
        for e in node.get("units", {}).get(unit, []):
            if e.get("form") != "10-K":
                continue
            if kind == "flow":
                s, en = e.get("start"), e.get("end")
                if not s or not en:
                    continue
                days = (date.fromisoformat(en) - date.fromisoformat(s)).days
                if not 350 <= days <= 380:
                    continue
            key = e["end"]
            if key not in found or e["filed"] > found[key]["filed"]:
                found[key] = e
        if found and (best_tag is None or max(found) > max(best)):
            best_tag, best = c, found
    rows = []
    for end in sorted(best, reverse=True)[:4]:
        e = best[end]
        rows.append({"end": end, "val": e["val"], "fy": e.get("fy"), "accn": e.get("accn"), "tag": best_tag})
    return rows


def sec_get(url, ua):
    import requests

    for attempt in range(3):
        r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429):
            time.sleep(2 * (attempt + 1))
            continue
        break
    return None


def load_cik_map(ua):
    data = sec_get("https://www.sec.gov/files/company_tickers.json", ua)
    if not data:
        return {}
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def money(v):
    if v is None:
        return NF
    a = abs(v)
    s = f"${a / 1e9:,.2f}B" if a >= 1e9 else f"${a / 1e6:,.1f}M" if a >= 1e6 else f"${a:,.0f}"
    return f"-{s}" if v < 0 else s


def summarize_facts(facts, cik):
    """Turn raw companyfacts JSON into FACT / CALCULATION rows with citations."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    if not gaap:
        return None
    S = {k: series_for(gaap, v, "flow") for k, v in FLOW.items()}
    S.update({k: series_for(gaap, v, "instant") for k, v in INSTANT.items()})
    S["shares"] = series_for(gaap, SHARES, "flow", unit="shares")
    rev = S["revenue"]
    if not rev:
        return None

    def cite(e):
        return f"10-K FY{e['fy']} (period ending {e['end']}), accession {e['accn']}, us-gaap:{e['tag']}"

    def at(key, end):
        return next((e for e in S[key] if e["end"] == end), None)

    latest_end = rev[0]["end"]
    rows = []

    def fact(label, key, end=None):
        e = at(key, end or latest_end)
        rows.append({"label": label, "kind": "FACT", "value": money(e["val"]) if e else NF,
                     "cite": cite(e) if e else "No matching 10-K value for this period"})
        return e

    for i, e in enumerate(rev[:3]):
        rows.append({"label": f"Revenue, FY ending {e['end']}", "kind": "FACT", "value": money(e["val"]), "cite": cite(e)})
    for i in range(min(3, len(rev) - 1)):
        g = rev[i]["val"] / rev[i + 1]["val"] - 1 if rev[i + 1]["val"] else None
        rows.append({"label": f"Revenue growth, FY ending {rev[i]['end']}", "kind": "CALCULATION",
                     "value": f"{g * 100:+.1f}%" if g is not None else NF,
                     "cite": f"({money(rev[i]['val'])} / {money(rev[i + 1]['val'])}) - 1"})
    gp, oi, ni = fact("Gross profit", "gross_profit"), fact("Operating income", "operating_income"), fact("Net income", "net_income")
    for label, e in (("Gross margin", gp), ("Operating margin", oi), ("Net margin", ni)):
        rows.append({"label": label, "kind": "CALCULATION",
                     "value": f"{e['val'] / rev[0]['val'] * 100:.1f}%" if e and rev[0]["val"] else NF,
                     "cite": f"{money(e['val'])} / {money(rev[0]['val'])}" if e else "Underlying value NOT FOUND"})
    cfo, capex = fact("Operating cash flow", "cfo"), fact("Capital expenditures", "capex")
    fcf = None
    if cfo and capex:
        fcf = cfo["val"] - capex["val"]
    rows.append({"label": "Free cash flow", "kind": "CALCULATION", "value": money(fcf) if fcf is not None else NF,
                 "cite": f"{money(cfo['val'])} - {money(capex['val'])}" if cfo and capex else "Needs both operating cash flow and capex"})
    cash, debt = fact("Cash & equivalents", "cash"), fact("Long-term debt", "debt")
    ie = fact("Interest expense", "interest_expense")
    cov = None
    if oi and ie and ie["val"]:
        cov = oi["val"] / ie["val"]
    rows.append({"label": "Interest coverage", "kind": "CALCULATION", "value": f"{cov:.1f}x" if cov is not None else NF,
                 "cite": f"{money(oi['val'])} / {money(ie['val'])}" if oi and ie else "Needs operating income and interest expense"})
    bb = fact("Share repurchases", "buybacks")
    sh = S["shares"]
    dil = None
    if len(sh) >= 3:
        dil = sh[0]["val"] / sh[2]["val"] - 1
    rows.append({"label": "Diluted share count change, 2 years", "kind": "CALCULATION",
                 "value": f"{dil * 100:+.1f}%" if dil is not None else NF,
                 "cite": f"{sh[0]['val'] / 1e6:,.0f}M / {sh[2]['val'] / 1e6:,.0f}M - 1" if dil is not None else "Needs 3 years of diluted shares"})

    flags = []
    if ni and ni["val"] < 0:
        flags.append("Net loss in the latest fiscal year")
    if fcf is not None and fcf < 0:
        flags.append("Negative free cash flow in the latest fiscal year")
    if cov is not None and cov < 3:
        flags.append(f"Interest coverage only {cov:.1f}x")
    if dil is not None and dil > 0.05:
        flags.append(f"Share count up {dil * 100:.1f}% in two years (dilution)")
    for label in ("Segment breakdown", "Top 3 risk factors", "Going-concern language, restatements, auditor changes, contingent liabilities"):
        rows.append({"label": label, "kind": "FACT", "value": NF,
                     "cite": "Needs the 10-K text itself (Item 1A, Item 8 notes). Ask Claude to read the filing."})
    return {
        "rows": rows,
        "flags": flags,
        "latest_fy_end": latest_end,
        "filings_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik:010d}&type=10-K",
        "note": "Values come from SEC XBRL data tagged in each 10-K. Cites give fiscal year, period end, accession number and tag.",
    }


def get_fundamentals(ticker, cik_map, ua, cache):
    if ticker.endswith(".TO"):
        return {"available": False, "reason": f"{NF}: not filed on EDGAR as a 10-K (Canadian filings are on SEDAR+)"}
    key = ticker.replace("-", ".").upper()
    cik = cik_map.get(key) or cik_map.get(ticker.upper())
    if not cik:
        return {"available": False, "reason": f"{NF}: no EDGAR CIK found for {ticker}"}
    c = cache.get(ticker)
    if c and (datetime.now(timezone.utc) - datetime.fromisoformat(c["fetched"])).days < 7:
        return c["data"]
    facts = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", ua)
    time.sleep(0.3)
    if not facts:
        return {"available": False, "reason": f"{NF}: SEC request failed (check SEC_USER_AGENT)"}
    summ = summarize_facts(facts, cik)
    if not summ:
        return {"available": False, "reason": f"{NF}: no annual USD revenue in 10-K XBRL data"}
    summ["available"] = True
    cache[ticker] = {"fetched": datetime.now(timezone.utc).isoformat(), "data": summ}
    return summ


# --------------------------------------------------------------------------- demo data
def synth_df(seed, n=300, end=None):
    rng = np.random.default_rng(seed)
    drift = rng.choice([-0.0008, -0.0003, 0.0002, 0.0006, 0.0010])
    r = rng.normal(drift, 0.017, n)
    close = 100 * rng.uniform(0.5, 4) * np.exp(np.cumsum(r))
    high = close * (1 + np.abs(rng.normal(0, 0.007, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.007, n)))
    op = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(3_000_000, 25_000_000, n).astype(float)
    vol[-1] *= rng.uniform(0.6, 2.2)
    idx = pd.bdate_range(end=end or datetime.now(ET).date(), periods=n)
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


# --------------------------------------------------------------------------- main
def run(demo=False, use_sec=True, top_n=None, cfg_path=None):
    cfg = load_config(cfg_path)
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    market_open, frac = market_state(now_et)
    names = {x["t"]: x["n"] for x in cfg["tickers"]}
    tickers = list(names)
    top_n = top_n or cfg.get("top_n_detail", 8)

    failed, ideas, macro = [], [], []
    if demo:
        daily = {t: synth_df(i + 1) for i, t in enumerate(tickers)}
        latest = {}
        for i, (k, sym) in enumerate(MACRO.items()):
            base = [7700, 15, 5.2, 78, 82, 3900, 1.38][i]
            macro.append({"label": k, "value": base, "change_pct": [-0.004, 0.02, 0.01, 0.015, 0.012, -0.003, 0.001][i], "time": "demo"})
    else:
        daily = fetch_daily(tickers)
        latest = fetch_latest(list(daily))
        md = fetch_daily(list(MACRO.values()), period="10d")
        ml = fetch_latest(list(MACRO.values()))
        for k, sym in MACRO.items():
            d = md.get(sym)
            if d is None or d.empty:
                macro.append({"label": k, "value": None, "change_pct": None, "time": NF})
                continue
            price, tstr = (ml[sym][0], ml[sym][1]) if sym in ml else (float(d["Close"].iloc[-1]), f"{pd.Timestamp(d.index[-1]).date()} close")
            prev = float(d["Close"].iloc[-2]) if len(d) > 1 and sym not in ml else float(d["Close"].iloc[-1])
            if sym in ml and len(d) > 1:
                last_day = pd.Timestamp(d.index[-1]).date()
                prev = float(d["Close"].iloc[-2]) if last_day == now_et.date() else float(d["Close"].iloc[-1])
            macro.append({"label": k, "value": _f(price, 3), "change_pct": _f(price / prev - 1, 4), "time": tstr})

    stale_cut = 30 * 60
    for t in tickers:
        df = daily.get(t)
        if df is None:
            failed.append(t)
            continue
        live, ptime, stale = None, None, False
        if t in latest:
            live, ptime, ts = latest[t]
            stale = market_open and (now_utc - ts).total_seconds() > stale_cut
        try:
            idea = analyse(t, names[t], df, live, cfg, now_et, market_open, frac)
        except Exception as e:  # noqa: BLE001
            print(f"{t}: analysis error {e}", file=sys.stderr)
            idea = None
        if not idea:
            failed.append(t)
            continue
        idea["price_time"] = ptime or ("demo" if demo else f"{pd.Timestamp(df.index[-1]).date()} close")
        if stale:
            idea["flags"].append("Price is more than 30 minutes old")
        ideas.append(idea)

    ideas.sort(key=lambda i: (i["conviction"], i["edge"]), reverse=True)
    for rank, idea in enumerate(ideas, 1):
        idea["rank"] = rank

    # enrich the leaders
    ua = os.environ.get("SEC_USER_AGENT") or cfg.get("sec_user_agent", "")
    sec_ok = use_sec and not demo and "example.com" not in ua and ua.strip() != ""
    cik_map, cache = {}, {}
    if sec_ok:
        cik_map = load_cik_map(ua)
        if FUND_CACHE.exists():
            try:
                cache = json.loads(FUND_CACHE.read_text())
            except Exception:  # noqa: BLE001
                cache = {}
    for idea in ideas[:top_n]:
        if demo:
            idea["earnings"] = {"date": NF, "days": None}
            idea["news"] = []
            idea["fundamentals"] = {"available": False, "reason": f"{NF}: demo mode"}
            continue
        ed, days = get_earnings(idea["ticker"])
        idea["earnings"] = {"date": ed or NF, "days": days}
        if days is not None and 0 <= days <= 5:
            idea["flags"].append(f"Earnings in {days} day(s): gap risk can jump past any stop")
        idea["news"] = get_news(idea["ticker"])
        if sec_ok:
            idea["fundamentals"] = get_fundamentals(idea["ticker"], cik_map, ua, cache)
        else:
            idea["fundamentals"] = {"available": False, "reason": f"{NF}: set the SEC_USER_AGENT secret to enable EDGAR pulls"}
    if sec_ok and cache:
        FUND_CACHE.write_text(json.dumps(cache))

    out = {
        "demo": demo,
        "generated_at": now_utc.isoformat(),
        "generated_et": now_et.strftime("%Y-%m-%d %H:%M ET"),
        "market_open": market_open,
        "account": {"size": cfg["account_size"], "risk_pct": cfg["risk_pct"], "max_position_pct": cfg.get("max_position_pct", 25)},
        "universe": len(tickers),
        "scanned": len(ideas),
        "failed": failed,
        "macro": macro,
        "ideas": ideas,
        "notes": [
            "Prices come from Yahoo Finance and can be delayed up to about 15 minutes; TSX is often delayed.",
            "Scores are a rule-based point system on price and volume only. They do not read news or filings.",
            "Fundamentals come from SEC EDGAR 10-K XBRL data; anything not in that data is marked NOT FOUND.",
            "Paper trading and education only. Not financial advice. Verify before using real money.",
        ],
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(out), indent=1, allow_nan=False))
    tmp.replace(OUT_FILE)
    print(f"Wrote {OUT_FILE} - {len(ideas)} scanned, {len(failed)} failed, market_open={market_open}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    ap.add_argument("--no-sec", action="store_true", help="skip SEC EDGAR pulls")
    ap.add_argument("--top", type=int, default=None, help="how many top ideas get news/fundamentals")
    a = ap.parse_args()
    res = run(demo=a.demo, use_sec=not a.no_sec, top_n=a.top)
    if not a.demo and res["scanned"] == 0:
        sys.exit("No tickers could be scanned (data source unreachable?).")
