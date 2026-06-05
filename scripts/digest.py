"""
Daily Tech & AI News Digest
Fetches top tech/AI headlines from Reuters, CNBC, FT RSS feeds,
filters for major company names and market-moving events,
summarises via Google Gemini API, and writes a clean HTML page for GitHub Pages.
"""

import os
import time
import feedparser
import requests
import json
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_KEY   = os.environ["GEMINI_API_KEY"]
GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

SAST  = timezone(timedelta(hours=2))
TODAY = datetime.now(SAST).strftime("%A, %d %B %Y")

# ── RSS feeds ─────────────────────────────────────────────────────────────────
FEEDS = [
    ("Reuters", "https://feeds.reuters.com/reuters/technologyNews"),
    ("Reuters", "https://feeds.reuters.com/reuters/businessNews"),
    ("CNBC",    "https://www.cnbc.com/id/19854910/device/rss/rss.html"),
    ("CNBC",    "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
    ("FT",      "https://www.ft.com/rss/home/technology"),
]

# ── Filter keywords ───────────────────────────────────────────────────────────
COMPANY_KEYWORDS = [
    "nvidia", "apple", "microsoft", "google", "alphabet", "meta", "amazon",
    "tesla", "openai", "anthropic", "intel", "amd", "qualcomm", "broadcom",
    "tsmc", "samsung", "asml", "arm", "palantir", "salesforce", "oracle",
    "ibm", "sap", "adobe", "snowflake", "databricks", "huawei",
    "aws", "azure", "gcp", "cloudflare", "datadog",
    "netflix", "spotify", "uber", "lyft", "airbnb", "x.com", "twitter",
    "micron", "sk hynix", "western digital", "seagate",
]

EVENT_KEYWORDS = [
    "earnings", "results", "revenue", "profit", "guidance", "forecast",
    "layoffs", "cuts jobs", "acquisition", "merger", "ipo", "buyback",
    "dividend", "capex", "ai", "chip", "model", "launch", "regulation",
    "antitrust", "fine", "lawsuit", "ceo", "partnership", "deal",
]

MAX_ARTICLES  = 12
MAX_AGE_HOURS = 26


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_relevant(title: str, summary: str) -> bool:
    text = (title + " " + summary).lower()
    return any(k in text for k in COMPANY_KEYWORDS) and \
           any(k in text for k in EVENT_KEYWORDS)


def fetch_full_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsDigestBot/1.0)"}
        resp    = requests.get(url, headers=headers, timeout=10)
        soup    = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = " ".join(p.get_text(separator=" ") for p in soup.find_all("p"))
        return text[:3000].strip()
    except Exception:
        return ""


def fetch_articles() -> list[dict]:
    articles = []
    cutoff   = datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)

    for source, url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if published and published < cutoff:
                continue

            title   = entry.get("title", "")
            summary = entry.get("summary", "")
            link    = entry.get("link", "")

            if not is_relevant(title, summary):
                continue
            if any(a["title"].lower()[:60] == title.lower()[:60] for a in articles):
                continue

            articles.append({"source": source, "title": title,
                              "summary": summary, "link": link,
                              "published": published})

    articles.sort(
        key=lambda x: x["published"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return articles[:MAX_ARTICLES]


# ── Summarisation ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an institutional equity research analyst writing a daily tech and AI news digest for a sophisticated investor audience.

For each article, write a concise summary in the style of the Economist and institutional equity research.

Writing rules:
- Laconic, direct, analytical. No fluff, no vague wording, no metaphors, no promotional language.
- Active voice only.
- Prioritise market-moving information and the "why it matters".
- Every sentence must contain: financial metrics, strategic implications, competitive positioning, valuation implications, industry impact, operational changes, guidance changes, regulatory impact, AI implications, capital allocation, or market reaction.

Structure (maximum two paragraphs):
- Para 1: headline event, financial results, guidance, margins, revenue, EPS, bookings, capex, share price reaction, key operational metrics.
- Para 2: strategic implications, competition, AI positioning, regulatory implications, industry context, valuation implications, risks, investor concerns, or long-term significance. Must contain factual evidence and metrics, not generic commentary.

Formatting:
- Sentence case only.
- Double space after each sentence.
- Use "US" not "U.S".
- bn = billion, m = million, tn = trillion, k = thousand.
- Spaces for thousands: "10 000" not "10,000".
- ISO currency codes: USD, EUR, GBP, KRW, JPY, CNY.
- y/y, q/q, FY '26, Q1 '26.
- Use "Est:" for consensus expectations.
- Use "c." for approximate values.
- No bullet points. No repeated information.
- No filler phrases like "management remains optimistic" unless backed by data.

If the article lacks enough data for a full two-paragraph summary, write one strong paragraph.
If the article is behind a paywall with no extractable content, write one sentence noting this."""


def summarise(article: dict) -> str:
    body    = fetch_full_text(article["link"]) if article["link"] else ""
    content = f"Title: {article['title']}\nSource: {article['source']}\n\n"
    content += f"Article text:\n{body}" if body else f"Summary/excerpt:\n{article['summary']}"

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": content}]}],
        "generationConfig": {"maxOutputTokens": 600, "temperature": 0.3},
    }

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_KEY}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"[Summary unavailable: {e}]"


# ── HTML page builder ─────────────────────────────────────────────────────────

def build_html(articles: list[dict], summaries: list[str]) -> str:
    cards = ""
    for article, summary in zip(articles, summaries):
        pub_str = ""
        if article["published"]:
            pub_str = article["published"].astimezone(SAST).strftime("%H:%M SAST")

        safe_summary = summary.replace("\n\n", "</p><p>").replace("\n", " ")

        cards += f"""
        <article>
          <div class="meta">{article['source']} &nbsp;·&nbsp; {pub_str}</div>
          <h2><a href="{article['link']}" target="_blank" rel="noopener">{article['title']}</a></h2>
          <p>{safe_summary}</p>
        </article>"""

    if not cards:
        cards = '<p class="empty">No major tech/AI articles matched today\'s filters. Check back tomorrow.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tech & AI Digest — {TODAY}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Georgia', serif;
      background: #0f0f0f;
      color: #e8e2d9;
      min-height: 100vh;
      padding: 48px 24px 80px;
    }}

    .container {{
      max-width: 760px;
      margin: 0 auto;
    }}

    header {{
      border-bottom: 1px solid #2a2a2a;
      padding-bottom: 24px;
      margin-bottom: 40px;
    }}

    .label {{
      font-family: 'Helvetica Neue', sans-serif;
      font-size: 10px;
      letter-spacing: 2.5px;
      text-transform: uppercase;
      color: #c8a96e;
      margin-bottom: 10px;
    }}

    h1 {{
      font-size: 22px;
      font-weight: normal;
      color: #f0ebe3;
      line-height: 1.3;
    }}

    .datestamp {{
      font-family: 'Helvetica Neue', sans-serif;
      font-size: 12px;
      color: #555;
      margin-top: 6px;
    }}

    article {{
      border-bottom: 1px solid #1e1e1e;
      padding: 32px 0;
    }}

    article:last-child {{
      border-bottom: none;
    }}

    .meta {{
      font-family: 'Helvetica Neue', sans-serif;
      font-size: 10px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: #555;
      margin-bottom: 10px;
    }}

    h2 {{
      font-size: 16px;
      font-weight: bold;
      line-height: 1.45;
      margin-bottom: 14px;
      color: #f0ebe3;
    }}

    h2 a {{
      color: inherit;
      text-decoration: none;
    }}

    h2 a:hover {{
      color: #c8a96e;
    }}

    p {{
      font-size: 14px;
      line-height: 1.8;
      color: #a89f94;
    }}

    .empty {{
      color: #444;
      font-style: italic;
      padding: 40px 0;
    }}

    footer {{
      margin-top: 60px;
      padding-top: 24px;
      border-top: 1px solid #1e1e1e;
      font-family: 'Helvetica Neue', sans-serif;
      font-size: 11px;
      color: #333;
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="label">Daily Briefing</div>
      <h1>Tech &amp; AI Digest</h1>
      <div class="datestamp">{TODAY}</div>
    </header>

    {cards}

    <footer>
      Generated at 02:00 SAST &nbsp;·&nbsp; Sources: Reuters, CNBC, FT &nbsp;·&nbsp; Summaries via Gemini
    </footer>
  </div>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[{TODAY}] Fetching articles...")
    articles = fetch_articles()
    print(f"  Found {len(articles)} relevant articles.")

    summaries = []
    for i, article in enumerate(articles, 1):
        print(f"  Summarising {i}/{len(articles)}: {article['title'][:60]}...")
        summaries.append(summarise(article))
        if i < len(articles):
            time.sleep(15)  # avoid Gemini free tier rate limit

    html = build_html(articles, summaries)

    out = Path("docs")
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    print("  Written to docs/index.html")


if __name__ == "__main__":
    main()
