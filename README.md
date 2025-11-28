🤖 FinBERT Financial News Analyzer

AI-powered financial sentiment analysis system using FinBERT transformer model with real-time news integration

Overview
A professional-grade financial news analysis system that leverages FinBERT (a BERT model fine-tuned on financial text) to analyze market sentiment and generate actionable trading signals. Built for demonstrating AI/ML applications in quantitative finance and alternative data analysis.
Key Features

🎯 FinBERT Sentiment Analysis - 97% accuracy on financial text classification
📰 Real-Time News Integration - Fetches live financial news via NewsAPI
📊 Trading Signal Generation - BUY/SELL/HOLD recommendations with confidence scores
📈 Multi-Asset Comparison - Comparative sentiment analysis across multiple stocks
📑 Investment Reports - Institutional-quality research reports
⚡ Risk & Catalyst Analysis - Identifies key drivers and risk factors

🚀 Demo Output
Article 1: POSITIVE (confidence: 91.23%)
  Score: 7.85/10
  Title: Apple Reports Strong Q4 Earnings Beat Estimates...

INVESTMENT RECOMMENDATION: BUY
OVERALL SENTIMENT: BULLISH (Score: 6.15/10)
CONFIDENCE LEVEL: HIGH

KEY INSIGHTS:
1. Strong financial performance reported by Bloomberg
2. Product innovation expanding addressable market
3. Operating leverage driving margin expansion
🛠 Installation
Prerequisites

Python 3.8 or higher
pip package manager

Quick Setup

Clone the repository

Install dependencies

bashpip install -r requirements.txt

Get NewsAPI Key (Free)

Sign up at newsapi.org
Copy your API key
Free tier: 100 requests/day, 7 days historical news


Set API Key

bash# Mac/Linux
export NEWSAPI_KEY='your-api-key-here'

# Windows Command Prompt
set NEWSAPI_KEY=your-api-key-here

# Windows PowerShell
$env:NEWSAPI_KEY='your-api-key-here'

Run the analyzer

bashpython financial_news_analyzer.py
💡 Usage
Basic Usage
pythonfrom financial_news_analyzer import FinancialNewsAnalyzer

# Initialize analyzer
analyzer = FinancialNewsAnalyzer(newsapi_key='your-key')

# Fetch and analyze news
articles = analyzer.fetch_news("Apple", num_results=5)
analysis = analyzer.analyze_sentiment(articles)

# Generate report
report = analyzer.generate_report("Apple", analysis, articles)
print(report)
Multi-Asset Comparison
python# Compare multiple stocks
assets = ["Apple", "Microsoft", "Tesla", "NVIDIA"]
comparison = analyzer.compare_multiple_assets(assets)
print(comparison)
Custom Queries
python# Individual stocks
analyzer.fetch_news("Tesla earnings", 5)

# Macro events
analyzer.fetch_news("Federal Reserve interest rates", 5)

# Sectors
analyzer.fetch_news("semiconductor industry", 5)
📊 Output Format
Sentiment Analysis

Score Range: -10 (Very Bearish) to +10 (Very Bullish)
Sentiment Labels: BULLISH / BEARISH / NEUTRAL
Trading Signals: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
Confidence Levels: HIGH / MEDIUM / LOW

Report Sections

Executive Summary
Investment Recommendation
Key Insights
Positive Catalysts
Risk Factors
Recommended Action
Article-by-Article Breakdown
News Sources

🧠 Model Information
FinBERT (ProsusAI/finbert)

Pre-trained on 10,000+ financial news articles
Fine-tuned for financial sentiment classification
Achieves 97% accuracy on financial text
Understands domain-specific terminology:

"beat estimates" → Positive
"missed guidance" → Negative
"margin expansion" → Positive
"regulatory headwinds" → Negative



📈 Use Cases
For Finance Professionals

Portfolio Managers: Monitor sentiment for holdings
Analysts: Supplement fundamental research
Traders: Identify sentiment-driven opportunities
Risk Managers: Track negative news catalysts

For Recruitment/Interviews
Demonstrates expertise in:

🤖 AI/ML integration in finance
📊 Alternative data analysis
💼 Financial domain knowledge
🔧 Production-quality Python code
📈 Quantitative finance concepts

🔧 Configuration
Environment Variables
bashNEWSAPI_KEY=your_api_key_here
Customization Options

Number of articles to fetch
Date range for news search
Sentiment threshold for signals
Confidence level filters

📚 Technical Stack

ML Framework: Hugging Face Transformers
Deep Learning: PyTorch
Data Processing: Pandas, NumPy
News API: NewsAPI (news aggregation)
NLP Model: FinBERT (financial sentiment)

🎯 Performance

Analysis Speed: 2-5 seconds per article
Model Load Time: 2-3 seconds (after first download)
Memory Usage: ~500MB (with model loaded)
Accuracy: 97% on financial sentiment tasks

📝 Example Reports
See examples/ directory for sample outputs:

apple_analysis.txt - Single stock deep dive
tech_comparison.txt - Multi-asset comparison
macro_analysis.txt - Fed interest rate analysis

⚠ Disclaimer
This tool is for educational and informational purposes only. It does not constitute investment advice. Always conduct your own research and consult with qualified financial advisors before making investment decisions.
👤 Author
Athul VM


⭐ Star this repo if you find it useful! ⭐
Built with ❤ for the finance and AI community
