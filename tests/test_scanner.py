"""Run with:  python -m unittest discover -s tests -v"""
import json
import math
import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scanner as sc  # noqa: E402

CFG = {"account_size": 100000, "risk_pct": 1.0, "max_position_pct": 25, "atr_stop_mult": 1.5, "min_conviction": 60}
NOW = datetime(2026, 9, 29, 11, 0, tzinfo=sc.ET)


def trend_df(kind="up", n=300):
    idx = pd.bdate_range(end=NOW.date(), periods=n)
    steps = np.linspace(0, 1, n)
    base = 100 + (60 * steps if kind == "up" else -50 * steps)
    noise = np.sin(np.arange(n) / 3.0) * 1.2
    close = base + noise
    return pd.DataFrame(
        {"Open": close, "High": close + 1.5, "Low": close - 1.5, "Close": close, "Volume": np.full(n, 5e6)}, index=idx
    )


class TestIndicators(unittest.TestCase):
    def test_rsi_bounds(self):
        r = sc.rsi_wilder(trend_df("up")["Close"]).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())

    def test_rsi_all_gains_is_100(self):
        s = pd.Series(np.arange(1, 60, dtype=float))
        self.assertEqual(round(sc.rsi_wilder(s).iloc[-1], 1), 100.0)

    def test_sma_matches_pandas(self):
        df = trend_df("up")
        ind = sc.compute_indicators(df, live_price=None, now_et=NOW, market_open=False)
        self.assertAlmostEqual(ind["sma50"], df["Close"].tail(50).mean(), places=2)
        self.assertAlmostEqual(ind["sma200"], df["Close"].tail(200).mean(), places=2)

    def test_too_short_history(self):
        self.assertIsNone(sc.compute_indicators(trend_df("up", 40), now_et=NOW))

    def test_live_price_changes_daychange(self):
        df = trend_df("up")
        last = float(df["Close"].iloc[-1])
        ind = sc.compute_indicators(df, live_price=last * 1.02, now_et=NOW, market_open=True, frac=0.3)
        self.assertAlmostEqual(ind["price"], last * 1.02, places=3)
        self.assertAlmostEqual(ind["change_pct"], last * 1.02 / float(df["Close"].iloc[-2]) - 1, places=4)


class TestScoringAndPlan(unittest.TestCase):
    def setUp(self):
        self.up = sc.analyse("UP", "Uptrend Co", trend_df("up"), None, CFG, NOW, False, 0)
        self.dn = sc.analyse("DN", "Downtrend Co", trend_df("down"), None, CFG, NOW, False, 0)

    def test_direction(self):
        self.assertEqual(self.up["direction"], "long")
        self.assertEqual(self.dn["direction"], "short")

    def test_scores_bounded(self):
        for i in (self.up, self.dn):
            for k in ("long_score", "short_score", "conviction"):
                self.assertTrue(0 <= i[k] <= 100)

    def test_long_plan_geometry(self):
        p = self.up["plan"]
        self.assertLess(p["stop"], p["entry"])
        self.assertGreater(p["t1"], p["entry"])
        self.assertGreater(p["t2"], p["t1"])
        self.assertGreaterEqual(p["rr1"], 1.25)

    def test_short_plan_geometry(self):
        p = self.dn["plan"]
        self.assertGreater(p["stop"], p["entry"])
        self.assertLess(p["t1"], p["entry"])
        self.assertLess(p["t2"], p["t1"])

    def test_risk_never_exceeds_budget(self):
        for i in (self.up, self.dn):
            p = i["plan"]
            self.assertLessEqual(p["shares"] * p["risk_per_share"], CFG["account_size"] * CFG["risk_pct"] / 100 + 1e-6)
            self.assertLessEqual(p["position_value"], CFG["account_size"] * CFG["max_position_pct"] / 100 + i["price"])

    def test_rr_math(self):
        p = self.up["plan"]
        rr = abs(p["t1"] - p["entry"]) / abs(p["entry"] - p["stop"])
        self.assertAlmostEqual(rr, p["rr1"], places=1)

    def test_json_safe(self):
        json.dumps(sc.clean(self.up), allow_nan=False)


class TestSec(unittest.TestCase):
    def _facts(self):
        def flow(vals):
            out = []
            for fy, v in vals:
                out.append({"start": f"{fy}-01-01", "end": f"{fy}-12-31", "val": v, "fy": fy, "form": "10-K",
                            "filed": f"{fy + 1}-02-15", "accn": f"0000000000-{fy + 1}-000001"})
            return {"units": {"USD": out}}

        def inst(vals):
            return {"units": {"USD": [{"end": f"{fy}-12-31", "val": v, "fy": fy, "form": "10-K",
                                       "filed": f"{fy + 1}-02-15", "accn": f"0000000000-{fy + 1}-000001"} for fy, v in vals]}}

        sh = {"units": {"shares": [{"start": f"{fy}-01-01", "end": f"{fy}-12-31", "val": v, "fy": fy, "form": "10-K",
                                    "filed": f"{fy + 1}-02-15", "accn": "x"} for fy, v in [(2023, 1e9), (2024, 1.02e9), (2025, 1.1e9)]]}}
        return {"facts": {"us-gaap": {
            "Revenues": flow([(2022, 80e9), (2023, 90e9), (2024, 100e9), (2025, 120e9)]),
            "GrossProfit": flow([(2025, 60e9)]),
            "OperatingIncomeLoss": flow([(2025, 24e9)]),
            "NetIncomeLoss": flow([(2025, 18e9)]),
            "NetCashProvidedByUsedInOperatingActivities": flow([(2025, 30e9)]),
            "PaymentsToAcquirePropertyPlantAndEquipment": flow([(2025, 10e9)]),
            "InterestExpense": flow([(2025, 2e9)]),
            "CashAndCashEquivalentsAtCarryingValue": inst([(2025, 15e9)]),
            "LongTermDebt": inst([(2025, 40e9)]),
            "WeightedAverageNumberOfDilutedSharesOutstanding": sh,
        }}}

    def test_summary_values(self):
        s = sc.summarize_facts(self._facts(), 123)
        rows = {r["label"]: r for r in s["rows"]}
        self.assertEqual(rows["Free cash flow"]["value"], "$20.00B")
        self.assertEqual(rows["Interest coverage"]["value"], "12.0x")
        self.assertEqual(rows["Gross margin"]["value"], "50.0%")
        self.assertTrue(rows["Revenue growth, FY ending 2025-12-31"]["value"].startswith("+20.0%"))
        self.assertTrue(any("dilution" in f for f in s["flags"]))  # 1.1/1.0 -1 = +10%
        self.assertEqual(rows["Segment breakdown"]["value"], "NOT FOUND")
        self.assertIn("accession", rows["Operating income"]["cite"])

    def test_missing_revenue_returns_none(self):
        self.assertIsNone(sc.summarize_facts({"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": []}}}}}, 1))

    def test_tsx_not_found(self):
        r = sc.get_fundamentals("RY.TO", {}, "ua", {})
        self.assertFalse(r["available"])
        self.assertIn("NOT FOUND", r["reason"])


class TestMarketState(unittest.TestCase):
    def test_states(self):
        self.assertEqual(sc.market_state(datetime(2026, 9, 29, 11, 0, tzinfo=sc.ET))[0], True)
        self.assertEqual(sc.market_state(datetime(2026, 9, 29, 17, 0, tzinfo=sc.ET))[0], False)
        self.assertEqual(sc.market_state(datetime(2026, 9, 26, 11, 0, tzinfo=sc.ET))[0], False)  # Saturday


if __name__ == "__main__":
    unittest.main()
