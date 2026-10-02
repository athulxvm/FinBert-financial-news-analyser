"""
AI-Powered Financial News Analysis System using FinBERT
Uses FinBERT transformer model for financial sentiment analysis.
100% Free - No API keys required!
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
import os
import argparse
warnings.filterwarnings('ignore')

# Load environment variables from a local .env file (e.g. NEWSAPI_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; env vars can still be set in the shell

# Financial sentiment analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Real news fetching
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    print("⚠️  NewsAPI not installed. Run: pip install newsapi-python")
    print("    Using sample news for now.\n")

# Historical prices for backtesting
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Convenience map so news queries (names) resolve to stock tickers for backtests
TICKER_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "tesla": "TSLA", "nvidia": "NVDA",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "facebook": "META", "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
}


class FinancialNewsAnalyzer:
    """
    Professional financial news analysis system using FinBERT.
    
    Features:
    - FinBERT sentiment analysis (trained on financial text)
    - News aggregation and scoring
    - Trading signal generation
    - Risk assessment
    - Multi-asset comparison
    - Professional reporting
    """
    
    def __init__(self, newsapi_key: str = None):
        """
        Initialize the analyzer with FinBERT model and NewsAPI.
        
        Args:
            newsapi_key: NewsAPI key (get free at https://newsapi.org)
                        If None, will try to read from environment variable NEWSAPI_KEY
        """
        print("🤖 Loading FinBERT model (this may take a moment first time)...")
        
        # Load FinBERT model
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize NewsAPI
        self.newsapi = None
        if NEWSAPI_AVAILABLE:
            api_key = newsapi_key or os.getenv('NEWSAPI_KEY')
            if api_key:
                try:
                    self.newsapi = NewsApiClient(api_key=api_key)
                    print("✅ NewsAPI initialized - Using REAL news!")
                except Exception as e:
                    print(f"⚠️  NewsAPI error: {e}")
                    print("    Using sample news instead.")
            else:
                print("⚠️  No NewsAPI key found. Set NEWSAPI_KEY environment variable")
                print("    or pass newsapi_key parameter. Using sample news.")
        
        self.analysis_history = []
        print("✅ FinBERT model loaded successfully!\n")
    
    def analyze_text_sentiment(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Analyze sentiment of a single text using FinBERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            (sentiment_label, confidence_score, all_probabilities)
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT outputs: [positive, negative, neutral]
        probs = predictions[0].cpu().numpy()
        labels = ['positive', 'negative', 'neutral']
        
        # Get dominant sentiment
        max_idx = np.argmax(probs)
        sentiment = labels[max_idx]
        confidence = float(probs[max_idx])
        
        prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
        
        return sentiment, confidence, prob_dict
    
    def fetch_news(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Fetch financial news from NewsAPI or use sample data.
        
        Args:
            query: Search query (company name, ticker, or topic)
            num_results: Number of articles to fetch
            
        Returns:
            List of news articles with metadata
        """
        print(f"📰 Fetching news for: {query}")
        
        # Try to fetch real news if NewsAPI is available
        if self.newsapi:
            try:
                return self._fetch_real_news(query, num_results)
            except Exception as e:
                print(f"⚠️  NewsAPI error: {e}")
                print("    Falling back to sample news...")
        
        # Fallback to sample news
        return self._fetch_sample_news(query, num_results)
    
    def _fetch_real_news(self, query: str, num_results: int) -> List[Dict]:
        """Fetch real news from NewsAPI."""
        # Calculate date range (last 7 days)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)
        
        # Build search query for financial news
        search_query = f'{query} AND (stock OR trading OR financial OR earnings OR revenue)'
        
        # Fetch from NewsAPI
        response = self.newsapi.get_everything(
            q=search_query,
            from_param=from_date.strftime('%Y-%m-%d'),
            to=to_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy',
            page_size=min(num_results, 20)  # NewsAPI max per request
        )
        
        if not response.get('articles'):
            print("⚠️  No articles found, using sample news")
            return self._fetch_sample_news(query, num_results)
        
        articles = []
        for article in response['articles'][:num_results]:
            # Clean and structure article data
            articles.append({
                'title': article.get('title', 'No title'),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'date': article.get('publishedAt', '')[:10],
                'content': article.get('description', '') or article.get('content', 'No content available'),
                'url': article.get('url', '')
            })
        
        print(f"✅ Fetched {len(articles)} real news articles\n")
        return articles
    
    def _fetch_sample_news(self, query: str, num_results: int) -> List[Dict]:
        """Fetch sample news for demonstration (when NewsAPI unavailable)."""
        
        # Sample news database (replace with real API)
        news_db = {
            "apple": [
                {
                    "title": "Apple Reports Record Q4 Revenue Driven by iPhone 15 Sales",
                    "source": "Bloomberg",
                    "date": "2024-11-22",
                    "content": "Apple Inc. reported quarterly revenue of $89.5 billion, exceeding analyst estimates of $87.2 billion. iPhone sales surged 12% year-over-year, with strong demand in China offsetting weakness in Europe. Services revenue hit all-time high of $22.3 billion. CEO Tim Cook raised full-year guidance, citing robust demand and operational efficiency gains. Gross margins expanded to 46.2%, beating expectations.",
                    "url": "https://bloomberg.com/apple-q4"
                },
                {
                    "title": "EU Opens Antitrust Investigation Into Apple's App Store",
                    "source": "Reuters",
                    "date": "2024-11-21",
                    "content": "The European Commission launched a formal antitrust probe into Apple's App Store practices. Regulators are examining whether Apple abuses its dominant position by charging excessive fees and restricting competition. Potential fines could reach up to 10% of global revenue. Apple responded that it will cooperate but maintains its practices are legal and pro-consumer.",
                    "url": "https://reuters.com/apple-eu"
                },
                {
                    "title": "Apple Unveils Advanced AI Chip for Next-Gen Devices",
                    "source": "Wall Street Journal",
                    "date": "2024-11-20",
                    "content": "Apple introduced its M4 chip featuring breakthrough AI processing capabilities. The new processor delivers 40% faster machine learning performance while reducing power consumption by 30%. Analysts predict this will strengthen Apple's competitive position in AI-powered devices. Morgan Stanley raised price target to $210, citing margin expansion potential from in-house chip development.",
                    "url": "https://wsj.com/apple-chip"
                }
            ],
            "tesla": [
                {
                    "title": "Tesla Deliveries Surge 15% in Q4, Stock Jumps",
                    "source": "CNBC",
                    "date": "2024-11-23",
                    "content": "Tesla delivered 485,000 vehicles in Q4, crushing analyst expectations of 470,000. Model Y became world's best-selling vehicle with 310,000 units delivered. Production ramp in Berlin and Texas factories exceeded targets. Elon Musk announced price cuts have stabilized demand. Free cash flow reached $3.2 billion. Stock surged 8% in pre-market on strong delivery numbers.",
                    "url": "https://cnbc.com/tesla-q4"
                },
                {
                    "title": "Tesla Faces Headwinds from Increased Competition in China",
                    "source": "Financial Times",
                    "date": "2024-11-21",
                    "content": "Tesla's market share in China declined to 8.7% as BYD and local competitors gain ground. Price wars intensified with Tesla cutting Model 3 prices by 5%. Analysts warn margin pressure could persist through 2025. However, Tesla's Gigafactory Shanghai remains profitable and production efficiency continues improving. Management remains confident in long-term China strategy.",
                    "url": "https://ft.com/tesla-china"
                }
            ],
            "microsoft": [
                {
                    "title": "Microsoft Cloud Revenue Exceeds $100B Annually",
                    "source": "Bloomberg",
                    "date": "2024-11-22",
                    "content": "Microsoft's Azure cloud platform generated $25.8 billion in quarterly revenue, up 28% year-over-year. Total commercial cloud revenue crossed $100 billion annual run rate. CEO Satya Nadella highlighted strong enterprise AI adoption with 18,000 Azure OpenAI customers. Office 365 Copilot seeing rapid uptake. Operating margins improved to 47%, demonstrating operating leverage at scale.",
                    "url": "https://bloomberg.com/msft-cloud"
                },
                {
                    "title": "Microsoft Announces Major Layoffs in Gaming Division",
                    "source": "The Verge",
                    "date": "2024-11-20",
                    "content": "Microsoft confirmed layoffs affecting 1,900 employees in its gaming division following Activision acquisition integration. The move aims to eliminate redundancies and streamline operations. Xbox Game Pass subscriptions grew slower than expected. Management emphasized long-term commitment to gaming but acknowledged near-term restructuring needed. Severance costs estimated at $450 million.",
                    "url": "https://theverge.com/msft-gaming"
                }
            ],
            "federal reserve": [
                {
                    "title": "Fed Holds Rates Steady, Signals Cautious Approach",
                    "source": "Wall Street Journal",
                    "date": "2024-11-22",
                    "content": "Federal Reserve maintained interest rates at 5.25-5.50% as expected. Chair Powell stated inflation remains above target, requiring continued vigilance. However, FOMC members expressed growing concern about labor market softening. Dot plot suggests only one rate cut likely in 2025, fewer than markets anticipated. Treasury yields jumped 15 basis points on hawkish tone.",
                    "url": "https://wsj.com/fed-meeting"
                },
                {
                    "title": "Core Inflation Stays Sticky Above Fed's 2% Target",
                    "source": "Reuters",
                    "date": "2024-11-21",
                    "content": "Core PCE inflation registered 2.8% year-over-year, unchanged from previous month. Services inflation remains elevated at 4.1%, driven by housing and healthcare costs. Fed officials acknowledge challenging path to 2% target. Market-implied probability of rate cut by March fell to 35%. Some economists warn prolonged high rates risk triggering recession in H2 2025.",
                    "url": "https://reuters.com/inflation"
                }
            ],
            "nvidia": [
                {
                    "title": "NVIDIA Crushes Earnings on AI Chip Demand Explosion",
                    "source": "CNBC",
                    "date": "2024-11-23",
                    "content": "NVIDIA reported revenue of $18.1 billion, up 206% year-over-year, demolishing estimates. Data center revenue hit $14.5 billion driven by insatiable AI chip demand. H100 chips completely sold out through 2025. CEO Jensen Huang called AI demand 'incredible' with enterprise adoption accelerating. Gross margins expanded to 75%. Guidance implies continued triple-digit growth.",
                    "url": "https://cnbc.com/nvidia-earnings"
                },
                {
                    "title": "US Expands Export Restrictions on NVIDIA's China Sales",
                    "source": "Bloomberg",
                    "date": "2024-11-20",
                    "content": "Biden administration tightened export controls on advanced AI chips to China. New restrictions prevent NVIDIA from selling H100 and certain A100 chips to Chinese customers. Management estimates $5-6 billion revenue impact in fiscal 2025. NVIDIA developing China-specific chips compliant with new rules. Analysts view regulatory risk as manageable given strong global demand.",
                    "url": "https://bloomberg.com/nvidia-china"
                }
            ]
        }
        
        # Match query to news
        query_lower = query.lower()
        articles = []
        
        for key in news_db:
            if key in query_lower:
                articles = news_db[key][:num_results]
                break
        
        if not articles:
            # Default to Apple if no match
            articles = news_db["apple"][:num_results]
        
        return articles
    
    def analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Comprehensive sentiment analysis using FinBERT.
        
        Args:
            articles: List of news articles
            
        Returns:
            Detailed analysis with signals and recommendations
        """
        print(f"🔍 Analyzing {len(articles)} articles with FinBERT...\n")
        
        sentiments = []
        article_scores = []
        
        # Analyze each article
        for i, article in enumerate(articles, 1):
            text = f"{article['title']}. {article['content']}"
            sentiment, confidence, probs = self.analyze_text_sentiment(text)
            
            # Calculate sentiment score (-10 to +10)
            score = (probs['positive'] - probs['negative']) * 10
            
            sentiments.append(sentiment)
            article_scores.append({
                'article_num': i,
                'title': article['title'],
                'sentiment': sentiment,
                'confidence': confidence,
                'score': score,
                'probabilities': probs
            })
            
            print(f"Article {i}: {sentiment.upper()} (confidence: {confidence:.2%})")
            print(f"  Score: {score:.2f}/10")
            print(f"  Title: {article['title'][:70]}...")
            print()
        
        # Aggregate sentiment
        avg_score = np.mean([a['score'] for a in article_scores])
        positive_count = sum(1 for s in sentiments if s == 'positive')
        negative_count = sum(1 for s in sentiments if s == 'negative')
        
        # Determine overall sentiment
        if avg_score > 3:
            overall_sentiment = "BULLISH"
            trading_signal = "BUY" if avg_score > 5 else "BUY"
        elif avg_score < -3:
            overall_sentiment = "BEARISH"
            trading_signal = "SELL" if avg_score < -5 else "SELL"
        else:
            overall_sentiment = "NEUTRAL"
            trading_signal = "HOLD"
        
        # Confidence based on agreement
        confidence_level = "HIGH" if max(positive_count, negative_count) >= len(articles) * 0.7 else "MEDIUM"
        if positive_count == negative_count:
            confidence_level = "LOW"
        
        # Generate insights
        key_insights = self._extract_insights(articles, article_scores)
        risk_factors = self._extract_risks(articles, article_scores)
        catalysts = self._extract_catalysts(articles, article_scores)
        
        analysis = {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(avg_score, 2),
            "confidence_level": confidence_level,
            "trading_signal": trading_signal,
            "article_breakdown": article_scores,
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": len(sentiments) - positive_count - negative_count
            },
            "key_insights": key_insights,
            "risk_factors": risk_factors,
            "catalysts": catalysts,
            "recommended_action": self._generate_recommendation(avg_score, overall_sentiment, articles),
            "analyst_summary": self._generate_summary(avg_score, overall_sentiment, positive_count, negative_count)
        }
        
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "articles": articles,
            "analysis": analysis
        })
        
        return analysis
    
    def _extract_insights(self, articles: List[Dict], scores: List[Dict]) -> List[str]:
        """Extract key insights from articles."""
        insights = []
        
        for article, score in zip(articles, scores):
            content = article['content'].lower()
            
            # Revenue/earnings mentions
            if any(word in content for word in ['revenue', 'earnings', 'profit', 'beat estimates']):
                if score['sentiment'] == 'positive':
                    insights.append(f"Strong financial performance reported by {article['source']}")
            
            # Growth mentions
            if any(word in content for word in ['growth', 'expansion', 'increased']):
                if 'revenue' in content or 'sales' in content:
                    insights.append("Accelerating revenue growth trajectory")
            
            # Competition/market share
            if any(word in content for word in ['market share', 'competition', 'competitors']):
                if score['sentiment'] == 'negative':
                    insights.append("Facing increased competitive pressure")
                else:
                    insights.append("Gaining competitive advantage in market")
        
        return insights[:3] if insights else ["Mixed signals from recent news flow"]
    
    def _extract_risks(self, articles: List[Dict], scores: List[Dict]) -> List[str]:
        """Extract risk factors."""
        risks = []
        
        for article, score in zip(articles, scores):
            content = article['content'].lower()
            
            if any(word in content for word in ['investigation', 'lawsuit', 'regulatory']):
                risks.append("Regulatory/legal uncertainties present downside risk")
            
            if any(word in content for word in ['competition', 'market share decline']):
                risks.append("Competitive dynamics pressuring margins")
            
            if any(word in content for word in ['weakness', 'slowdown', 'declined']):
                risks.append("Demand headwinds in key markets")
        
        return risks[:2] if risks else ["Limited near-term risk factors identified"]
    
    def _extract_catalysts(self, articles: List[Dict], scores: List[Dict]) -> List[str]:
        """Extract positive catalysts."""
        catalysts = []
        
        for article, score in zip(articles, scores):
            if score['sentiment'] == 'positive':
                content = article['content'].lower()
                
                if 'earnings' in content or 'revenue' in content:
                    catalysts.append("Strong earnings momentum likely to continue")
                
                if any(word in content for word in ['new product', 'innovation', 'breakthrough']):
                    catalysts.append("Product innovation expanding addressable market")
                
                if 'margin' in content:
                    catalysts.append("Operating leverage driving margin expansion")
        
        return catalysts[:2] if catalysts else ["Limited near-term catalysts identified"]
    
    def _generate_recommendation(self, score: float, sentiment: str, articles: List[Dict]) -> str:
        """Generate trading recommendation."""
        if score > 5:
            return "Strong buy signal. Consider initiating or adding to positions. News flow suggests positive momentum with limited downside risk. Set stop-loss at -8% to manage risk."
        elif score > 2:
            return "Moderate buy signal. Accumulate on weakness. Positive fundamentals offset by some uncertainty. Consider scale-in approach with 50% position initially."
        elif score < -5:
            return "Strong sell signal. Consider reducing exposure or hedging positions. Multiple negative catalysts suggest downside risk. Consider protective puts or exit positions."
        elif score < -2:
            return "Moderate sell signal. Trim positions or tighten stops. Risk-reward appears unfavorable near-term. Wait for clearer technical support before re-entering."
        else:
            return "Neutral outlook. Maintain current positions but avoid adding. Wait for clearer directional signal. Consider range-trading strategies if established positions exist."
    
    def _generate_summary(self, score: float, sentiment: str, pos: int, neg: int) -> str:
        """Generate executive summary."""
        if pos > neg:
            return f"Recent news flow is predominantly positive ({pos} positive vs {neg} negative articles). Fundamentals appear strong with upside momentum. Risk-reward favors long positioning with defined risk management."
        elif neg > pos:
            return f"News sentiment skews negative ({neg} negative vs {pos} positive articles). Multiple headwinds suggest caution warranted. Consider defensive positioning or reducing exposure until clearer picture emerges."
        else:
            return "Mixed news flow with balanced positive and negative signals. Lack of clear directional catalyst suggests sideways consolidation likely near-term. Maintain neutral stance."
    
    def compare_multiple_assets(self, queries: List[str]) -> pd.DataFrame:
        """Compare sentiment across multiple assets."""
        print("\n" + "="*90)
        print("MULTI-ASSET COMPARATIVE ANALYSIS")
        print("="*90 + "\n")
        
        results = []
        for query in queries:
            articles = self.fetch_news(query, num_results=3)
            analysis = self.analyze_sentiment(articles)
            
            results.append({
                "Asset": query.upper(),
                "Sentiment": analysis["overall_sentiment"],
                "Score": analysis["sentiment_score"],
                "Signal": analysis["trading_signal"],
                "Confidence": analysis["confidence_level"],
                "Positive": analysis["sentiment_distribution"]["positive"],
                "Negative": analysis["sentiment_distribution"]["negative"]
            })
        
        return pd.DataFrame(results)
    
    def backtest(self, query: str, ticker: str, days: int = 30,
                 articles_per_day: int = 8, signal_threshold: float = 2.0) -> Dict:
        """
        Test whether news sentiment predicts next-day price moves.

        For each trading day it aggregates the sentiment of news published since
        the prior close, then measures the stock's return from that day's close
        to the next day's close (so trades only act on news already seen — no
        look-ahead). Reports correlation, directional hit-rate, and a simple
        long/short strategy vs. buy-and-hold.

        NOTE: NewsAPI's free tier serves only ~30 days of history, so the sample
        is tiny (typically 10-20 trading days). Treat the output as a methodology
        demonstration, NOT a statistically validated trading signal.

        Args:
            query: news search term (e.g. "Apple")
            ticker: stock symbol for price data (e.g. "AAPL")
            days: lookback window in days (capped at 30 by free NewsAPI)
            max_articles: max articles to pull in the single API request
            signal_threshold: |sentiment| above which the strategy takes a position
        """
        print("\n" + "=" * 90)
        print(f"BACKTEST: Does news sentiment predict {ticker.upper()} price moves?")
        print("=" * 90 + "\n")

        if not YFINANCE_AVAILABLE:
            print("❌ yfinance not installed. Run: pip install yfinance")
            return {}

        if not self.newsapi:
            print("❌ Backtesting needs REAL news (NewsAPI) — sample data has no usable dates.")
            print("   Set NEWSAPI_KEY in your .env and try again.")
            return {}

        # Free NewsAPI tier caps history at ~1 month
        days = min(days, 30)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        # 1 & 2. Fetch news ONE DAY AT A TIME so coverage spreads across the window.
        # (A single sorted request just returns the newest ~100 articles, which for a
        #  popular query all cluster on the last day — giving only one data point.)
        print(f"📰 Fetching ~{articles_per_day} '{query}' articles/day across {days} days "
              f"(≈{days} API requests)...")
        records = []
        total_articles = 0
        for offset in range(days, -1, -1):
            day_str = (to_date - timedelta(days=offset)).strftime('%Y-%m-%d')
            try:
                resp = self.newsapi.get_everything(
                    q=query,
                    from_param=day_str,
                    to=day_str,
                    language='en',
                    sort_by='relevancy',
                    page_size=min(articles_per_day, 20),
                )
            except Exception as e:
                print(f"⚠️  NewsAPI error on {day_str}: {e}")
                continue
            arts = resp.get('articles', [])
            total_articles += len(arts)
            for a in arts:
                title = a.get('title') or ''
                body = a.get('description') or a.get('content') or ''
                text = f"{title}. {body}".strip()
                if not text or text == '.':
                    continue
                _, _, probs = self.analyze_text_sentiment(text)
                score = (probs['positive'] - probs['negative']) * 10
                records.append({'date': day_str, 'score': score})

        if not records:
            print("❌ No scorable articles returned for this query/date range.")
            return {}
        print(f"✅ Scored {total_articles} articles across the window.")

        news_df = pd.DataFrame(records)
        news_df['date'] = pd.to_datetime(news_df['date'])

        # 3. Download daily prices (pad end so the last day has a next-day close)
        print(f"📈 Downloading {ticker.upper()} prices from Yahoo Finance...\n")
        prices = yf.download(
            ticker,
            start=from_date.strftime('%Y-%m-%d'),
            end=(to_date + timedelta(days=4)).strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True,
        )
        if prices is None or prices.empty:
            print(f"❌ No price data for ticker '{ticker}'. Check the symbol.")
            return {}

        # yfinance can return MultiIndex columns even for a single ticker
        close = prices['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        close.index = close.index.normalize()
        trading_days = close.index.sort_values()

        # 4. Attribute each article to the next trading day on/after its publish date
        #    (news over a weekend counts toward the following Monday)
        def next_trading_day(d):
            pos = trading_days.searchsorted(d.normalize())
            return trading_days[pos] if pos < len(trading_days) else pd.NaT

        news_df['tday'] = news_df['date'].apply(next_trading_day)
        news_df = news_df.dropna(subset=['tday'])
        daily_sent = news_df.groupby('tday')['score'].mean()
        article_counts = news_df.groupby('tday')['score'].count()

        # 5. Forward return: close[T] -> close[T+1], recorded on day T
        fwd_return = close.pct_change().shift(-1)

        merged = pd.DataFrame({'sentiment': daily_sent})
        merged['articles'] = article_counts
        merged['fwd_return'] = fwd_return.reindex(merged.index)
        merged = merged.dropna(subset=['fwd_return']).sort_index()

        if len(merged) == 0:
            print("❌ No overlap between news days and trading days — nothing to test.")
            return {}

        # 6. Metrics
        sent = merged['sentiment'].values
        ret = merged['fwd_return'].values
        n = len(merged)

        if n >= 2 and np.std(sent) > 0 and np.std(ret) > 0:
            corr = float(np.corrcoef(sent, ret)[0, 1])
        else:
            corr = float('nan')

        # Directional hit-rate (skip near-neutral days)
        mask = np.abs(sent) > 0.5
        hit_rate = float(np.mean(np.sign(sent[mask]) == np.sign(ret[mask]))) if mask.sum() else float('nan')

        # Simple strategy: long if sentiment > thr, short if < -thr, else flat
        position = np.where(sent > signal_threshold, 1,
                            np.where(sent < -signal_threshold, -1, 0))
        strat_daily = position * ret
        strat_cum = float(np.prod(1 + strat_daily) - 1)
        bh_cum = float(np.prod(1 + ret) - 1)
        trades = int(np.sum(position != 0))

        # 7. Day-by-day table
        print(f"{'Date':<12}{'Articles':>9}{'Sentiment':>11}{'Next-day %':>12}{'Agree?':>8}")
        print("-" * 52)
        for day, row in merged.iterrows():
            s = row['sentiment']
            r = row['fwd_return'] * 100
            agree = "—" if abs(s) <= 0.5 else ("✓" if np.sign(s) == np.sign(r) else "✗")
            print(f"{day.strftime('%Y-%m-%d'):<12}{int(row['articles']):>9}"
                  f"{s:>+11.2f}{r:>+11.2f}%{agree:>8}")

        # 8. Summary
        print("\n" + "=" * 90)
        print("BACKTEST RESULTS")
        print("=" * 90)
        print(f"  Trading days tested:        {n}")
        print(f"  Correlation (sentiment vs next-day return): "
              f"{corr:+.3f}" if corr == corr else "  Correlation: n/a (not enough variation)")
        if hit_rate == hit_rate:
            print(f"  Directional hit-rate:       {hit_rate:.0%}  "
                  f"(coin-flip = 50%; {int(mask.sum())} non-neutral days)")
        else:
            print("  Directional hit-rate:       n/a (no non-neutral days)")
        print(f"\n  Strategy (|sentiment| > {signal_threshold}, long/short): "
              f"{strat_cum:+.2%}  over {trades} trade-days")
        print(f"  Buy & hold over same window:                 {bh_cum:+.2%}")
        edge = strat_cum - bh_cum
        print(f"  Strategy edge vs buy & hold:                 {edge:+.2%}")

        print("\n" + "-" * 90)
        print("⚠️  INTERPRETATION")
        print("-" * 90)
        if n < 15:
            print(f"  Only {n} data points — this is FAR too small to conclude anything.")
            print("  The free NewsAPI tier (~30 days history) makes a real backtest impossible;")
            print("  this run demonstrates the METHOD. For real results you'd need months/years")
            print("  of historical news (paid NewsAPI, or an archive like GDELT).")
        else:
            print("  Still a small sample — directional/correlation results are suggestive at best.")
        if corr == corr:
            direction = "positively" if corr > 0 else "negatively"
            strength = ("essentially no" if abs(corr) < 0.1 else
                        "a weak" if abs(corr) < 0.3 else "a moderate")
            print(f"  Observed: sentiment correlates {direction} with next-day returns "
                  f"({strength} relationship).")
        print("=" * 90 + "\n")

        return {
            "ticker": ticker.upper(),
            "query": query,
            "days_tested": n,
            "correlation": corr,
            "hit_rate": hit_rate,
            "strategy_return": strat_cum,
            "buy_hold_return": bh_cum,
            "edge": edge,
            "data": merged,
        }

    def generate_report(self, query: str, analysis: Dict, articles: List[Dict]) -> str:
        """Generate professional investment report."""
        
        report = f"""
{'='*90}
FINBERT-POWERED INVESTMENT RESEARCH REPORT
{'='*90}

SUBJECT: {query.upper()}
DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
MODEL: FinBERT (Financial Sentiment Analysis Transformer)
ARTICLES ANALYZED: {len(articles)}

{'='*90}
EXECUTIVE SUMMARY
{'='*90}

{analysis['analyst_summary']}

INVESTMENT RECOMMENDATION: {analysis['trading_signal']}
OVERALL SENTIMENT: {analysis['overall_sentiment']} (Score: {analysis['sentiment_score']:.2f}/10)
CONFIDENCE LEVEL: {analysis['confidence_level']}

SENTIMENT BREAKDOWN:
  Positive Articles: {analysis['sentiment_distribution']['positive']}
  Negative Articles: {analysis['sentiment_distribution']['negative']}
  Neutral Articles:  {analysis['sentiment_distribution']['neutral']}

{'='*90}
KEY INSIGHTS
{'='*90}
"""
        for i, insight in enumerate(analysis['key_insights'], 1):
            report += f"{i}. {insight}\n"
        
        report += f"""
{'='*90}
POSITIVE CATALYSTS
{'='*90}
"""
        for i, catalyst in enumerate(analysis['catalysts'], 1):
            report += f"{i}. {catalyst}\n"
        
        report += f"""
{'='*90}
RISK FACTORS
{'='*90}
"""
        for i, risk in enumerate(analysis['risk_factors'], 1):
            report += f"{i}. {risk}\n"
        
        report += f"""
{'='*90}
RECOMMENDED ACTION
{'='*90}

{analysis['recommended_action']}

{'='*90}
ARTICLE-BY-ARTICLE BREAKDOWN
{'='*90}
"""
        for article_analysis in analysis['article_breakdown']:
            report += f"""
Article {article_analysis['article_num']}: {article_analysis['sentiment'].upper()}
  Confidence: {article_analysis['confidence']:.1%}
  Score: {article_analysis['score']:.2f}/10
  Title: {article_analysis['title']}
  
  Sentiment Probabilities:
    Positive: {article_analysis['probabilities']['positive']:.1%}
    Negative: {article_analysis['probabilities']['negative']:.1%}
    Neutral:  {article_analysis['probabilities']['neutral']:.1%}
"""
        
        report += f"""
{'='*90}
NEWS SOURCES
{'='*90}
"""
        for i, article in enumerate(articles, 1):
            report += f"""
{i}. {article['title']}
   Source: {article['source']} | Date: {article['date']}
   URL: {article['url']}
"""
        
        report += f"""
{'='*90}
METHODOLOGY
{'='*90}

This analysis uses FinBERT, a BERT-based transformer model fine-tuned on financial
text (10K+ financial news articles). FinBERT achieves 97% accuracy on financial
sentiment classification and understands domain-specific context like "beat estimates",
"raised guidance", and "margin expansion".

Model: ProsusAI/finbert (Hugging Face)
Sentiment Scale: -10 (Very Bearish) to +10 (Very Bullish)

{'='*90}
DISCLAIMER
{'='*90}

This report is generated using AI-powered sentiment analysis for informational
purposes only. It does not constitute investment advice. Past performance does not
guarantee future results. Please conduct your own due diligence and consult with
financial advisors before making investment decisions.

{'='*90}
"""
        return report


def run_demo(analyzer):
    """Run the original three-example showcase."""
    # Example 1: Single asset analysis
    print("\n" + "="*90)
    print("EXAMPLE 1: DEEP DIVE - SINGLE ASSET ANALYSIS")
    print("="*90 + "\n")
    
    query = "Apple"
    articles = analyzer.fetch_news(query, num_results=3)
    analysis = analyzer.analyze_sentiment(articles)
    
    # Generate report
    report = analyzer.generate_report(query, analysis, articles)
    print(report)
    
    # Example 2: Multi-asset comparison
    print("\n" + "="*90)
    print("EXAMPLE 2: COMPARATIVE ANALYSIS - TECH STOCKS")
    print("="*90 + "\n")
    
    assets = ["Apple", "Microsoft", "NVIDIA"]
    comparison_df = analyzer.compare_multiple_assets(assets)
    
    print("\n📊 SENTIMENT COMPARISON TABLE")
    print("="*90)
    print(comparison_df.to_string(index=False))
    print()
    
    # Example 3: Macro analysis
    print("\n" + "="*90)
    print("EXAMPLE 3: MACRO EVENT ANALYSIS")
    print("="*90 + "\n")
    
    macro_query = "Federal Reserve"
    macro_articles = analyzer.fetch_news(macro_query, num_results=2)
    macro_analysis = analyzer.analyze_sentiment(macro_articles)
    
    print(f"📈 MACRO SENTIMENT: {macro_analysis['overall_sentiment']}")
    print(f"🎯 SIGNAL: {macro_analysis['trading_signal']}")
    print(f"💭 SUMMARY: {macro_analysis['analyst_summary']}")
    
    print("\n" + "="*90)
    print("✅ ANALYSIS COMPLETE")
    print("="*90)
    print(f"\nTotal analyses performed: {len(analyzer.analysis_history)}")
    print("\n📚 NEXT STEPS:")
    print("  1. Replace sample news with real API (NewsAPI, Alpha Vantage)")
    print("  2. Add historical sentiment tracking")
    print("  3. Integrate with portfolio management system")
    print("  4. Build automated alert system")
    print("  5. Add backtesting: sentiment vs. price movements")


def main():
    """Parse CLI arguments and dispatch."""
    parser = argparse.ArgumentParser(
        description="FinBERT financial news sentiment analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Analyze one stock's recent news
  python "fin news analyser.py" --query "Tesla" --count 5

  # Compare several stocks
  python "fin news analyser.py" --compare "Apple,Microsoft,NVIDIA"

  # Backtest: does sentiment predict next-day price? (needs NEWSAPI_KEY)
  python "fin news analyser.py" --backtest --query "Apple" --ticker AAPL

  # Run the original three-example showcase
  python "fin news analyser.py" --demo
""",
    )
    parser.add_argument('--query', help='Company or topic to analyze')
    parser.add_argument('--count', type=int, default=3, help='Number of articles (default 3)')
    parser.add_argument('--compare', help='Comma-separated list of assets to compare')
    parser.add_argument('--backtest', action='store_true',
                        help='Backtest news sentiment vs. next-day price moves')
    parser.add_argument('--ticker', help='Stock ticker for --backtest (e.g. AAPL); '
                                         'inferred from --query for well-known names')
    parser.add_argument('--days', type=int, default=30,
                        help='Backtest lookback window in days (max 30 on free NewsAPI)')
    parser.add_argument('--demo', action='store_true',
                        help='Run the original three-example showcase')
    args = parser.parse_args()

    print("=" * 90)
    print("FINBERT FINANCIAL NEWS ANALYSIS SYSTEM")
    print("=" * 90)
    print("\n🚀 Initializing FinBERT sentiment analyzer...")
    print("Note: First run will download the model (~400MB)\n")

    newsapi_key = os.getenv('NEWSAPI_KEY')
    if not newsapi_key:
        print("⚠️  No NEWSAPI_KEY found — running on built-in SAMPLE news.")
        print("    Get a free key at https://newsapi.org and add it to your .env file.\n")

    analyzer = FinancialNewsAnalyzer(newsapi_key=newsapi_key)

    # Dispatch based on arguments
    if args.backtest:
        query = args.query or args.ticker
        ticker = args.ticker or TICKER_MAP.get((args.query or '').lower())
        if not query or not ticker:
            parser.error("--backtest needs --ticker (and ideally --query). "
                         "e.g. --backtest --query Apple --ticker AAPL")
        analyzer.backtest(query, ticker, days=args.days)

    elif args.compare:
        assets = [a.strip() for a in args.compare.split(',') if a.strip()]
        comparison_df = analyzer.compare_multiple_assets(assets)
        print("\n📊 SENTIMENT COMPARISON TABLE")
        print("=" * 90)
        print(comparison_df.to_string(index=False))
        print()

    elif args.query:
        articles = analyzer.fetch_news(args.query, num_results=args.count)
        analysis = analyzer.analyze_sentiment(articles)
        print(analyzer.generate_report(args.query, analysis, articles))

    else:
        # No action specified → run the showcase
        run_demo(analyzer)


if __name__ == "__main__":
    main()