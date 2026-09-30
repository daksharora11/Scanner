# Bull & Bear Ledger

A paper-trading idea scanner. It re-scans about 45 US and TSX stocks every 5 minutes during market hours, ranks them by bull score and bear score, and gives each one a full paper-trade plan. It has no broker connection and cannot place real trades.

**Educational only. Not financial advice. Verify anything before using real money.**

## What you get

- **Ranked list.** Every stock gets a bull score (long setup) and a bear score (short setup), 0 to 100. The "Trade now" filter shows only ideas with conviction 60+, at least 1.5R to the first target, and enough liquidity.
- **Paper-trade plan for each idea.** Entry zone, stop-loss, two targets (each with its reasoning), risk/reward math, position size from your account and risk %, horizon, and what would invalidate it.
- **Bull case, bear case, and what may be priced in** from price action.
- **Fundamentals from SEC 10-K filings** for the top-ranked ideas: 3-year revenue growth, margins, free cash flow, debt, interest coverage, cash, buybacks and dilution. Each number carries its filing citation (fiscal year, period end, accession number, XBRL tag).
- **Macro strip:** S&P 500, VIX, 10-year yield, WTI, Brent, gold, USD/CAD.
- **Paper log** (Ideas tab, then Paper log tab): log trades, close them with "what I got right / wrong", see win rate and average R, export to CSV. Stored in your browser only.

Every figure is labelled **fact**, **calc** (with the formula shown) or **opinion**. Anything it cannot verify shows **NOT FOUND** instead of an estimate.

## Set-up (about 10 minutes, no coding)

1. **Create a public repository** on GitHub (public is required for free Actions minutes and free Pages hosting). Name it anything, for example `ledger`.
2. **Upload everything in this folder.** On the repo page choose Add file, then Upload files, and drag in the contents. If the hidden `.github` folder did not upload (common on Mac), choose Add file, then Create new file, type `.github/workflows/scan.yml` as the name, and paste in the contents of that file.
3. **Turn on the website.** Settings, then Pages. Under Build and deployment choose Deploy from a branch, branch `main`, folder `/docs`, then Save. Your dashboard link appears there within a minute or two.
4. **Add your SEC contact.** Settings, then Secrets and variables, then Actions, then New repository secret. Name: `SEC_USER_AGENT`. Value: your name and email, for example `Daksh Arora daksh@example.com`. The SEC requires this on every request; without it the fundamentals show NOT FOUND.
5. **Allow the bot to save results.** Settings, then Actions, then General, then Workflow permissions: choose Read and write permissions, then Save.
6. **Run the first scan.** Actions tab, then scan, then Run workflow. After about a minute, open your dashboard link. From then on it re-scans every 5 minutes on weekdays during market hours.

Change the watchlist, account size or risk % by editing `config.json` in the GitHub website (pencil icon). Tickers ending in `.TO` are TSX.

## How accurate is "live"? Be realistic

- Prices come from Yahoo Finance's unofficial feed. Expect up to about 15 minutes of delay, more on TSX. Each idea shows the timestamp of its price.
- GitHub's 5-minute timer is best effort and often runs several minutes late. The page shows how old the scan is and warns if it is stale. Open the page and it re-checks every 60 seconds.
- This is a research aid, not a trading terminal. Do not use it for fast intraday execution.
- The scanner does not know exchange holidays. On a holiday it simply re-scans the last close.

## How the scores work

Bull points: price above 50-day MA (15), above 200-day MA (10), 50-day above 200-day (10), positive 20-day return (8), positive 5-day return (5), RSI 50-68 (12), pullback inside an uptrend (12), high volume on an up day (up to 10), within 5% of the 52-week high (10), at least 2R of chart room to Target 1 (10). Overbought RSI above 75 subtracts 10. Bear points mirror these. The total is scaled to 0-100 (90 raw points = 100).

Trade plan: stop sits just beyond the nearest support or resistance if it is 0.75 to 2.5 ATR away, otherwise 1.5 x ATR. Target 1 is the nearest chart level at least 1.25R away, otherwise 1.5R. Target 2 is 3R. Size = (account x risk %) / risk per share, capped at 25% of the account.

## What it cannot do (ask Claude for these)

- Read news or judge what the market expects. The scores use price and volume only.
- Read the 10-K text: risk factors, segment breakdown, going-concern language, restatements, auditor changes and contingent liabilities all show NOT FOUND. Paste a ticker to Claude and ask for a filing review.
- Verify earnings dates. The date shown is from Yahoo; confirm it on the company's investor page.
- Canadian filers: TSX names are not on EDGAR as 10-Ks, so fundamentals show NOT FOUND (see SEDAR+).

## Files

- `scanner.py` - the engine. `python scanner.py --demo` writes synthetic data to preview the page.
- `docs/index.html` - the dashboard.
- `docs/data/latest.json` - the newest scan (starts as clearly labelled demo data).
- `.github/workflows/scan.yml` - the 5-minute schedule.
- `config.json` - watchlist and account settings.
- `tests/test_scanner.py` - checks the math. Run `python -m unittest discover -s tests -v`.
