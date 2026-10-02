# 🤖 FinBERT Financial News Analyzer

AI-powered financial **news sentiment** analysis using the FinBERT transformer, with optional live news (NewsAPI) and a sentiment-vs-price backtest.

## Overview

This tool reads recent news about a company or topic, uses **FinBERT** (a BERT model fine-tuned on financial text) to judge whether the coverage *reads* positive, negative, or neutral, and turns that into a score from **−10 to +10**. It can analyze a single stock, compare several, and — with a NewsAPI key — **backtest whether that sentiment actually lined up with next-day price moves**.

> **What it measures:** the *tone of the news writing*, not market data. It does not look at prices, fundamentals, or volume when scoring sentiment (prices are used only in the backtest, to check sentiment against reality). Positive-sounding news does **not** guarantee the stock rises.

## Key Features

- **FinBERT sentiment analysis** — a model fine-tuned for financial language, so it reads terms like "beat estimates," "raised guidance," and "margin expansion" in context.
- **Optional live news** — fetches recent articles via [NewsAPI](https://newsapi.org); falls back to built-in sample articles if no key is set.
- **Trading signal generation** — BULLISH / BEARISH / NEUTRAL with BUY / SELL / HOLD labels and a confidence level.
- **Multi-asset comparison** — rank several stocks by how positive their recent coverage is.
- **Backtesting** — test whether news sentiment predicted next-day returns (correlation, directional hit-rate, and a simple strategy vs. buy-and-hold).
- **Command-line interface** — analyze any stock without editing code.
- **Investment reports** — formatted research-report output.

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- `pip`

### Setup

1. **Clone the repository.**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   First run downloads the FinBERT model (~400 MB) from Hugging Face.

3. **(Optional) Get a free NewsAPI key** for live news and backtesting:
   - Register at [newsapi.org/register](https://newsapi.org/register)
   - Free tier: **100 requests/day**, and articles from roughly the **last 30 days**
   - Without a key the tool still runs, using built-in **sample** news

4. **Add your key** to a `.env` file in the project root (copy the template):
   ```bash
   cp .env.example .env
   # then edit .env and paste your key:
   # NEWSAPI_KEY=your-api-key-here
   ```
   The `.env` file is gitignored, so your key is never committed. (You can also just `export NEWSAPI_KEY=...` in your shell instead.)

## 🚀 Usage

The script is run via its command-line interface. Because the filename contains spaces, quote it:

```bash
# Analyze a single stock's recent news
python "fin news analyser.py" --query "Tesla" --count 5

# Compare several stocks
python "fin news analyser.py" --compare "Apple,Microsoft,NVIDIA"

# Backtest: did sentiment predict next-day price? (needs NEWSAPI_KEY)
python "fin news analyser.py" --backtest --query "Apple" --ticker AAPL

# Run the original three-example showcase
python "fin news analyser.py" --demo

# See all options
python "fin news analyser.py" --help
```

| Flag | Description |
|------|-------------|
| `--query` | Company or topic to analyze |
| `--count` | Number of articles to fetch (default 3) |
| `--compare` | Comma-separated list of assets to compare |
| `--backtest` | Backtest sentiment vs. next-day price moves |
| `--ticker` | Stock symbol for `--backtest` (e.g. `AAPL`); inferred for well-known names |
| `--days` | Backtest lookback window in days (max 30 on free NewsAPI) |
| `--demo` | Run the original three-example showcase |

### Example output (single stock)

```
Article 1: NEUTRAL (confidence: 70.11%)
  Score: -1.93/10
  Title: Stock Bets Put Prediction Markets Under Fresh Regulatory Scrutiny...

INVESTMENT RECOMMENDATION: HOLD
OVERALL SENTIMENT: NEUTRAL (Score: -1.34/10)
CONFIDENCE LEVEL: LOW
```

*(Live results vary with the news of the day — real coverage is often mixed, so NEUTRAL/HOLD is a common and honest outcome.)*

## 📈 Backtesting

The backtest checks whether news tone had any predictive relationship with price.

**Method (no look-ahead bias):**
- For each day, news published since the previous market close is attributed to the next trading day `T`.
- That day's sentiment is tested against the **forward return** (close `T` → close `T+1`) — i.e. you only act on news already seen.
- Reports: **Pearson correlation**, **directional hit-rate** (did sentiment's sign match the next day's?), and a simple **long/short strategy vs. buy-and-hold**.

**Example run (illustrative — numbers change daily):**
```
  Trading days tested:        21
  Correlation (sentiment vs next-day return): +0.391
  Directional hit-rate:       64%  (coin-flip = 50%)
  Strategy (|sentiment| > 2.0, long/short):   +9.11%
  Buy & hold over same window:                +2.44%
```

⚠️ **This is a methodology demo, not a validated signal.** The free NewsAPI tier only serves ~30 days of history, so a backtest has ~10–20 data points — far too few to be statistically significant. A credible study needs months or years of historical news (paid NewsAPI, or a free archive such as [GDELT](https://www.gdeltproject.org/)). Each backtest run uses ~30 API requests (one per day), so the free tier allows ~3 backtests/day.

## 📊 Output Format

- **Score range:** −10 (very bearish) to +10 (very bullish)
- **Sentiment labels:** BULLISH / BEARISH / NEUTRAL
- **Trading signals:** BUY / SELL / HOLD
- **Confidence levels:** HIGH / MEDIUM / LOW

Report sections: Executive Summary · Investment Recommendation · Key Insights · Positive Catalysts · Risk Factors · Recommended Action · Article-by-Article Breakdown · News Sources.

> Note: the "insights / catalysts / risks" sections use simple keyword matching (not the ML model) and are best treated as rough highlights.

## 🧠 Model Information

**FinBERT** (`ProsusAI/finbert` on Hugging Face)

- A BERT model fine-tuned on financial text (Financial PhraseBank) for sentiment classification.
- Outputs three probabilities per text: positive / negative / neutral.
- Substantially outperforms general-purpose sentiment models on financial sentences, because it understands domain terms:
  - "beat estimates" → positive
  - "missed guidance" → negative
  - "margin expansion" → positive
  - "regulatory headwinds" → negative

> Accuracy figures quoted online (often ~86–97%) come from the original paper's benchmark dataset. **Real-world accuracy on arbitrary live news is lower** — treat the scores as a signal, not ground truth.

## ⚠️ Limitations

- Scores reflect the **tone of news text**, not actual market performance.
- The leap from "positive news" to "BUY" is an assumption the tool makes; the backtest is there precisely to test it, not to confirm it.
- Free NewsAPI history (~30 days) makes rigorous backtesting impossible.
- Keyword-based insights/risks/catalysts are brittle on real-world news.
- Not financial advice (see disclaimer).

## 🧰 Technical Stack

- **ML / NLP:** Hugging Face Transformers, PyTorch, FinBERT
- **Data:** Pandas, NumPy
- **News:** NewsAPI (`newsapi-python`)
- **Prices (backtest):** yfinance
- **Config:** python-dotenv

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It does **not** constitute investment advice. Past performance does not guarantee future results. Always do your own research and consult a qualified financial advisor before making investment decisions.

## 👤 Author

Athul VM

---

⭐ Star this repo if you find it useful! Built for the finance and AI community.
