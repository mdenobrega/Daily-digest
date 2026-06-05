"""
Daily Tech & AI News Digest
Fetches top tech/AI headlines from Reuters, CNBC, FT RSS feeds,
splits into company-specific summaries and thematic further reading,
summarises via Google Gemini API, writes a clean HTML page for GitHub Pages.
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
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

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

# ── Keywords ──────────────────────────────────────────────────────────────────

# Specific company names — article must mention one of these to be summarised
COMPANY_KEYWORDS = [
    "nvidia", "apple", "microsoft", "google", "alphabet", "meta", "amazon",
    "tesla", "openai", "anthropic", "intel", "amd", "qualcomm", "broadcom",
    "tsmc", "samsung", "asml", "arm", "palantir", "salesforce", "oracle",
    "ibm", "sap", "adobe", "snowflake", "databricks", "huawei",
    "aws", "azure", "gcp", "cloudflare", "datadog",
    "netflix", "spotify", "uber", "lyft", "airbnb", "x.com", "twitter",
    "micron", "sk hynix", "western digital", "seagate", "crowdstrike",
    "spacex", "bluesky", "xai", "deepmind", "gemini", "mistral",
]

# Hard financial/operational events — required for a full summary
FINANCIAL_EVENTS = [
    "earnings", "results", "revenue", "profit", "loss", "guidance", "forecast",
    "layoffs", "cuts jobs", "job cuts", "acquisition", "acquires", "merger",
    "ipo", "buyback", "share repurchase", "dividend", "capex", "valuation",
    "raises", "funding", "investment", "stake", "ceo", "cfo", "coo",
    "antitrust", "fine", "lawsuit", "penalty", "regulation", "ban",
    "partnership", "deal", "contract", "stock", "shares", "market cap",
]

# Broader tech/AI signals — enough for further reading but not a full summary
THEMATIC_KEYWORDS = [
    "ai", "chip", "semiconductor", "model", "launch", "data centre",
    "cloud", "compute", "llm", "robot", "autonomous", "regulation",
    "talent", "research", "benchmark", "open source",
]

MAX_SUMMARIES = 10   # articles that get full summaries
MAX_FURTHER   = 8    # articles in further reading


def get_cutoff() -> datetime:
    """
    Returns start of previous business day in UTC.
    Monday 2am SAST  → Friday 00:00 SAST (3 days back)
    Tue–Fri 2am SAST → previous day 00:00 SAST (1 day back)
    """
    now_sast    = datetime.now(SAST)
    weekday     = now_sast.weekday()          # 0=Mon ... 4=Fri
    days_back   = 3 if weekday == 0 else 1
    cutoff_date = now_sast.date() - timedelta(days=days_back)
    cutoff_sast = datetime(cutoff_date.year, cutoff_date.month,
                           cutoff_date.day, 0, 0, tzinfo=SAST)
    return cutoff_sast.astimezone(timezone.utc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def has_company(text: str) -> bool:
    return any(k in text for k in COMPANY_KEYWORDS)

def has_financial_event(text: str) -> bool:
    return any(k in text for k in FINANCIAL_EVENTS)

def has_thematic(text: str) -> bool:
    return any(k in text for k in THEMATIC_KEYWORDS)

def is_duplicate(title: str, existing: list[dict]) -> bool:
    return any(a["title"].lower()[:60] == title.lower()[:60] for a in existing)


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


def fetch_articles() -> tuple[list[dict], list[dict]]:
    """Returns (summary_articles, further_reading_articles)."""
    summaries_pool = []
    further_pool   = []
    cutoff         = get_cutoff()

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
            text    = (title + " " + summary).lower()

            article = {"source": source, "title": title,
                       "summary": summary, "link": link,
                       "published": published}

            # Tier 1: company-specific + financial event → full summary
            if has_company(text) and has_financial_event(text):
                if not is_duplicate(title, summaries_pool):
                    summaries_pool.append(article)

            # Tier 2: company or thematic signal → further reading
            elif (has_company(text) or has_thematic(text)) and link:
                if not is_duplicate(title, further_pool) and \
                   not is_duplicate(title, summaries_pool):
                    further_pool.append(article)

    def sort_key(x):
        return x["published"] or datetime.min.replace(tzinfo=timezone.utc)

    summaries_pool.sort(key=sort_key, reverse=True)
    further_pool.sort(key=sort_key, reverse=True)

    return summaries_pool[:MAX_SUMMARIES], further_pool[:MAX_FURTHER]


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


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(articles: list[dict], summaries: list[str],
               further: list[dict]) -> str:

    # ── Main article cards ────────────────────────────────────────────────────
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
        cards = '<p class="empty">No company-specific articles matched today\'s filters. Check back tomorrow.</p>'

    # ── Further reading links ─────────────────────────────────────────────────
    further_html = ""
    if further:
        links = ""
        for a in further:
            pub_str = ""
            if a["published"]:
                pub_str = a["published"].astimezone(SAST).strftime("%H:%M")
            links += f"""
            <div class="fr-item">
              <span class="fr-source">{a['source']} {pub_str}</span>
              <a href="{a['link']}" target="_blank" rel="noopener">{a['title']}</a>
            </div>"""

        further_html = f"""
        <section class="further">
          <div class="further-label">Further reading</div>
          {links}
        </section>"""

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

    h2 a:hover {{ color: #c8a96e; }}

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

    /* Further reading */
    .further {{
      margin-top: 48px;
      padding-top: 32px;
      border-top: 1px solid #2a2a2a;
    }}

    .further-label {{
      font-family: 'Helvetica Neue', sans-serif;
      font-size: 10px;
      letter-spacing: 2.5px;
      text-transform: uppercase;
      color: #c8a96e;
      margin-bottom: 20px;
    }}

    .fr-item {{
      display: flex;
      gap: 16px;
      align-items: baseline;
      padding: 10px 0;
      border-bottom: 1px solid #1a1a1a;
    }}

    .fr-item:last-child {{ border-bottom: none; }}

    .fr-source {{
      font-family: 'Helvetica Neue', sans-serif;
      font-size: 10px;
      color: #444;
      white-space: nowrap;
      min-width: 80px;
      letter-spacing: 0.5px;
    }}

    .fr-item a {{
      font-size: 13px;
      color: #7a7068;
      text-decoration: none;
      line-height: 1.5;
    }}

    .fr-item a:hover {{ color: #c8a96e; }}

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
    {further_html}

    <footer>
      Generated at 02:00 SAST &nbsp;·&nbsp; Sources: Reuters, CNBC, FT &nbsp;·&nbsp; Summaries via Gemini
    </footer>
  </div>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[{TODAY}] Fetching articles...")
    articles, further = fetch_articles()
    print(f"  {len(articles)} articles to summarise, {len(further)} for further reading.")

    summaries = []
    for i, article in enumerate(articles, 1):
        print(f"  Summarising {i}/{len(articles)}: {article['title'][:60]}...")
        summaries.append(summarise(article))
        if i < len(articles):
            time.sleep(15)  # avoid Gemini free tier rate limit

    html = build_html(articles, summaries, further)

    out = Path("docs")
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    print("  Written to docs/index.html")


if __name__ == "__main__":
    main()
