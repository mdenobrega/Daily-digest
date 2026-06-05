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
GROQ_KEY = os.environ["GROQ_API_KEY"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SAST  = timezone(timedelta(hours=2))
TODAY = datetime.now(SAST).strftime("%A, %d %B %Y")

# ── RSS feeds ─────────────────────────────────────────────────────────────────
FEEDS = [
    # Reuters
    ("Reuters",     "https://feeds.reuters.com/reuters/technologyNews"),
    ("Reuters",     "https://feeds.reuters.com/reuters/businessNews"),
    # CNBC
    ("CNBC",        "https://www.cnbc.com/id/19854910/device/rss/rss.html"),
    ("CNBC",        "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
    # FT
    ("FT",          "https://www.ft.com/rss/home/technology"),
    # Bloomberg Technology
    ("Bloomberg",   "https://feeds.bloomberg.com/technology/news.rss"),
    # TechCrunch
    ("TechCrunch",  "https://techcrunch.com/feed/"),
    # The Verge
    ("The Verge",   "https://www.theverge.com/rss/index.xml"),
    # WSJ Tech
    ("WSJ",         "https://feeds.a.dj.com/rss/RSSWSJD.xml"),
]

# ── Keywords ──────────────────────────────────────────────────────────────────

# Broad company signal — catches any named company in a tech/AI/finance context.
# Kept intentionally wide; the AI summariser and FINANCIAL_EVENTS filter do the
# precision work. Add names here only if they are consistently missed.
COMPANY_KEYWORDS = [
    # Big Tech & cloud
    "nvidia", "apple", "microsoft", "google", "alphabet", "meta", "amazon",
    "tesla", "openai", "anthropic", "intel", "amd", "qualcomm", "broadcom",
    "tsmc", "samsung", "asml", "arm holdings", "palantir", "salesforce",
    "oracle", "ibm", "sap", "adobe", "snowflake", "databricks", "huawei",
    "aws", "azure", "google cloud", "cloudflare", "datadog", "servicenow",
    "workday", "veeva", "zendesk", "twilio", "okta", "crowdstrike", "palo alto",
    # Consumer & social
    "netflix", "spotify", "uber", "lyft", "airbnb", "twitter", "x.com",
    "pinterest", "snap", "tiktok", "bytedance", "reddit", "linkedin",
    "booking.com", "expedia", "tripadvisor", "doordash", "instacart",
    # Semiconductors & hardware
    "micron", "sk hynix", "western digital", "seagate", "marvell", "monolithic",
    "foxconn", "hon hai", "pegatron", "flex", "jabil", "corning", "keysight",
    # Data centres & infrastructure
    "equinix", "digital realty", "iron mountain", "switch", "ntt data",
    # AI & emerging tech
    "spacex", "bluesky", "xai", "deepmind", "gemini", "mistral", "cohere",
    "perplexity", "stability ai", "inflection", "runway", "scale ai",
    "hugging face", "together ai", "anyscale", "weights & biases",
    # Fintech
    "stripe", "klarna", "revolut", "nubank", "affirm", "robinhood", "coinbase",
    "paypal", "square", "block", "visa", "mastercard", "adyen",
    # Enterprise & SaaS
    "atlassian", "freshworks", "hubspot", "monday.com", "asana", "notion",
    "figma", "canva", "gitlab", "github", "hashicorp", "confluent",
    # Telecoms & hardware
    "ericsson", "nokia", "cisco", "juniper", "arista", "pure storage",
    "netapp", "dell", "hp", "lenovo", "asus",
    # EV & robotics
    "rivian", "lucid", "waymo", "cruise", "nuro", "aurora", "mobileye",
]

# Hard financial/operational events — required for a full summary
FINANCIAL_EVENTS = [
    "earnings", "results", "revenue", "profit", "loss", "guidance", "forecast",
    "layoffs", "cuts jobs", "job cuts", "acquisition", "acquires", "merger",
    "ipo", "buyback", "share repurchase", "dividend", "capex", "valuation",
    "raises", "funding", "investment", "stake", "ceo", "cfo", "coo",
    "antitrust", "fine", "lawsuit", "penalty", "regulation", "ban",
    "partnership", "deal", "contract", "stock", "shares", "market cap",
    # Product/operational launches with financial scale
    "unveils", "launches", "deploys", "expands", "opens", "signs",
    "plans", "announces", "cuts", "closes", "sells", "buys",
    "billion", "million", "USD", "EUR", "GBP",
]

# Broader tech/AI signals — enough for further reading but not a full summary
THEMATIC_KEYWORDS = [
    "ai", "chip", "semiconductor", "model", "launch", "data centre",
    "cloud", "compute", "llm", "robot", "autonomous", "regulation",
    "talent", "research", "benchmark", "open source",
]

# Fully blocked — junk with no value anywhere on the page
EXCLUDE_ALWAYS = [
    # Roundups and newsletters
    "and more", "5 things", "morning squawk", "what to know", "week ahead",
    "roundup", "wrap", "recap", "things to watch", "what's happening",
    "need to know", "in charts", "in numbers", "top stories", "highlights",
    "morning brief", "evening brief", "daily briefing", "this week",
    "market open", "before the bell", "after the bell", "premarket",
    # Sponsored content
    "sponsored", "partner content", "presented by", "paid post",
    # Pure listicles
    "obsessing over", "everything you need", "all you need",
    # Named newsletter columns
    "tech download", "tech wrap", "tech briefing", "morning download",
    "daily download", "weekly download", "the download:",
    "squawk newsletter", "pro newsletter",
    # CEO personal wealth
    "net worth", "billionaire", "richest person", "wealthiest",
    "sail past", "surpasses", "fortune grows",
    # Crypto / financial instruments on tech stocks
    "perpetual futures", "coinbase", "pre-ipo", "tokenised",
    "crypto", "bitcoin", "blockchain", "nft",
]

# Summary-only exclusions — too soft for a full summary but fine as further reading
EXCLUDE_SUMMARY = [
    # Opinion / recommendations
    "is a buy", "is a sell", "is a hold", "here's why", "why you should",
    "the case for", "the case against", "opinion:", "commentary:",
    "analyst says", "analysts say", "wall street says", "should you buy",
    "should you sell", "time to buy", "time to sell", "worth buying",
    "is it worth", "overrated", "underrated", "undervalued", "overvalued",
    "price target", "rating", "upgrade", "downgrade",
    # Soft explainers
    "what is", "who is", "how to", "guide to", "look at", "explained",
    # Interviews and profiles
    "sits down with", "in conversation with", "talks to", "speaks to",
    "interview:", "interview with", "q&a", "in his own words",
    # Predictions and outlooks
    "what to expect", "predictions for", "outlook for", "forecast for",
    "what lies ahead", "what's next for", "the future of", "looking ahead",
    # Retrospectives
    "lessons from", "history of", "look back at", "years ago", "founded",
    "the rise of", "the story of", "how it started",
    # Awards and rankings
    "best companies", "top 10", "top 5", "most valuable", "richest",
    "ranked:", "best and worst", "winners and losers",
    # Personal finance crossover
    "how this affects you", "what it means for your", "investors should",
    "retail investors", "for your portfolio",
    # Event and conference coverage
    "at davos", "at ces", "at sxsw", "keynote:", "speaks at", "appearance at",
    # Career and culture
    "culture at", "what it's like to work", "employees say", "workers say",
    "best employer", "great place to work",
    # Political/regulatory meetings without operational news
    "meets with lawmakers", "meets with senators", "meets with congress",
    "senate hearing", "invites", "meets trump", "white house meeting",
    "meets with officials", "washington visit", "dc visit",
    "meets with lawmakers", "capitol hill",
]

MAX_SUMMARIES = 10   # articles that get full summaries
MAX_FURTHER   = 8    # articles in further reading


def get_cutoff() -> datetime:
    """
    Returns 07:00 SAST of the previous business day in UTC.
    Monday 6am SAST  → Friday 07:00 SAST (3 days back)
    Tue–Fri 6am SAST → previous day 07:00 SAST (1 day back)
    """
    now_sast    = datetime.now(SAST)
    weekday     = now_sast.weekday()          # 0=Mon ... 4=Fri
    days_back   = 3 if weekday == 0 else 1
    cutoff_date = now_sast.date() - timedelta(days=days_back)
    cutoff_sast = datetime(cutoff_date.year, cutoff_date.month,
                           cutoff_date.day, 7, 0, tzinfo=SAST)
    return cutoff_sast.astimezone(timezone.utc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def has_company(text: str) -> bool:
    return any(k in text for k in COMPANY_KEYWORDS)

def has_financial_event(text: str) -> bool:
    return any(k in text for k in FINANCIAL_EVENTS)

def has_thematic(text: str) -> bool:
    return any(k in text for k in THEMATIC_KEYWORDS)

def is_excluded_always(title: str) -> bool:
    """Fully block junk — never appears anywhere on the page."""
    t = title.lower()
    return any(p in t for p in EXCLUDE_ALWAYS)

def is_excluded_summary(title: str) -> bool:
    """Too soft for a full summary but fine as further reading."""
    t = title.lower()
    return any(p in t for p in EXCLUDE_SUMMARY)

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

            # Fully block junk — skip entirely
            if is_excluded_always(title):
                continue

            # Tier 1: company-specific + financial event + not soft → full summary
            if has_company(text) and has_financial_event(text) and                not is_excluded_summary(title):
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
SYSTEM_PROMPT = """You are Avior's institutional equity research analyst writing a daily tech and AI news digest for a sophisticated investor audience.

Summarise the article in a concise, institutional investor style similar to equity research commentary. Follow all rules below exactly.

---

WRITING STYLE
- Use a laconic writing style inspired by the Economist Style Guide.
- Be direct, specific, and analytical.
- Avoid fluff, vague wording, metaphors, and promotional language.
- Use active voice only.
- Prioritise market-moving information.
- Focus on the "why it matters".
- Every sentence must contain one of: financial metrics, strategic implications, competitive positioning, valuation implications, industry impact, operational changes, guidance changes, regulatory impact, AI implications, capital allocation, or market reaction.
- Do not quote analysts, commentators, or third parties. Report only company facts.
- No filler: "management remains optimistic", "strong positioning", "continues to execute" — unless supported by data.

---

STRUCTURE — maximum two paragraphs:

Paragraph 1: headline event, financial results, guidance, margins, revenue, EPS, bookings, capex, share price reaction, key operational metrics.

Paragraph 2: strategic implications, competition, AI positioning, regulatory implications, industry context, valuation implications, risks, investor concerns, or long-term significance. Must contain factual evidence and metrics — no generic commentary.

---

IMPORTANT REQUIREMENTS
- Always compare results against expectations where possible.
- Include share price reactions if available.
- Include valuation metrics if relevant.
- Highlight contradictions: strong earnings but weak guidance; strong revenue but margin pressure; AI investment but worsening cash flow.
- Explain why the stock moved.
- Explain what investors are focused on next.
- If discussing AI: quantify capex, compute demand, infrastructure spending, customer concentration, backlog, power usage, chip shortages, monetisation, or competitive positioning.
- If discussing M&A: cover premiums, strategic rationale, synergies, shareholder structure, regulatory risk, and competitive implications.
- If discussing layoffs: explain the strategic reason — AI automation, restructuring, margin preservation, or compute reallocation.

---

FORMATTING — follow exactly:
- Sentence case. Every sentence starts with a capital letter. All proper nouns and company names capitalised (Nvidia, Meta, Amazon, AI, CEO).
- Double space after every sentence.
- Use "US" not "U.S".
- bn = billion, m = million, tn = trillion, k = thousand.
- Spaces in thousands: "10 000" not "10,000".
- ISO currency codes: USD, EUR, GBP, KRW, JPY, CNY, ZAR.
- Est: used inline for consensus (e.g. "revenue of USD4.2bn (Est: USD3.9bn)").
- Use "c." for approximate values.
- Dates: 15 Sep '26. FY '26. CY '26. YTD. y/y. q/q.
- Ratings in full capitals: OUTPERFORM, UNDERPERFORM, MARKET PERFORM.
- No bullet points. No repeated information.
- Replace "increased" with "rose". Replace "decreased" with "fell".
- Do not use "while the" — start a new sentence instead.
- Maximum 20 words per sentence.

---

REFERENCE EXAMPLES — match this style exactly:

Example 1 — Nvidia:
Nvidia forecast Q2 revenue of USD91bn (Est: USD86.8bn) and announced a USD80bn share buyback alongside a dividend increase to USD0.25 per share, as AI infrastructure demand continued accelerating globally.  Q1 revenue rose 85% y/y to USD81.6bn (Est: USD78.9bn), while data centre revenue rose 91% y/y to USD75.2bn (Est: USD72.8bn) and adjusted EPS came in at USD1.87 (Est: USD1.76).  Despite the beat, shares fell 1.6% after-hours as investors questioned whether AI spending can sustain current growth rates into 2027–2028.

Management stated hyperscale AI capex could exceed USD700bn this year versus c.USD400bn in 2025, while the new Vera CPU platform could unlock an additional USD200bn addressable market.  CEO Jensen Huang acknowledged Nvidia has largely conceded China's AI chip market to Huawei following tightening US export restrictions, with China previously representing at least 20% of data centre revenue.  Nvidia flagged ongoing memory shortages and supply constraints across the Vera Rubin cycle, with competition from AMD, Intel and internally developed hyperscaler chips intensifying.

Example 2 — Uber:
Uber forecast Q2 gross bookings of USD56.25–57.75bn (Est: USD56.1bn) despite geopolitical and fuel-cost headwinds, sending shares 8% higher.  Q1 revenue rose 14% y/y to USD13.2bn, while gross bookings rose 25% y/y to USD53.7bn (Est: USD52.8bn).  Delivery revenue rose 34% y/y to USD5.1bn, materially outperforming mobility growth of 5% y/y, as the company benefited from strong international demand and Uber One membership growth beyond 50m users.

Uber's results highlight increasing exposure to higher-frequency delivery and platform monetisation rather than solely ride-hailing.  AI is helping moderate hiring growth, with 95% of engineers now using AI coding tools monthly and over 10% of code written autonomously.  Uber is expanding its autonomous vehicle ecosystem through partnerships with Waymo, WeRide and Wayve, targeting robotaxi deployment in 15 cities by end-2026.

---

If the article lacks sufficient data for two paragraphs, write one strong paragraph.
If the article is paywalled with no extractable content, write: "Article paywalled — insufficient data to summarise." """


def summarise(article: dict) -> str:
    body    = fetch_full_text(article["link"]) if article["link"] else ""
    content = f"Title: {article['title']}\nSource: {article['source']}\n\n"
    content += f"Article text:\n{body}" if body else f"Summary/excerpt:\n{article['summary']}"

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        "max_tokens": 600,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_KEY}",
            },
            data=json.dumps(payload),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
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
        safe_summary = summary.replace("\n\n", '</p><p style="margin-top:14px;">').replace("\n", " ")
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

    /* Avior brand colours */
    :root {{
      --dark-blue:    #232c3f;
      --seafoam:      #87ccd1;
      --gold:         #A09680;
      --cyan:         #01a0c6;
      --light-grey:   #e6e6e6;
      --white:        #ffffff;
      --text-primary: #232c3f;
      --text-muted:   #6b7280;
      --border:       #e6e6e6;
    }}

    body {{
      font-family: 'News Gothic MT', 'News Gothic', 'Century Gothic', sans-serif;
      background: #f5f6f7;
      color: var(--text-primary);
      min-height: 100vh;
      padding: 48px 24px 80px;
    }}

    .container {{
      max-width: 760px;
      margin: 0 auto;
    }}

    header {{
      background: var(--dark-blue);
      padding: 28px 32px;
      margin-bottom: 32px;
      border-left: 4px solid var(--seafoam);
    }}

    .label {{
      font-size: 10px;
      letter-spacing: 2.5px;
      text-transform: uppercase;
      color: var(--seafoam);
      margin-bottom: 8px;
    }}

    h1 {{
      font-size: 20px;
      font-weight: bold;
      color: var(--white);
      line-height: 1.3;
    }}

    .datestamp {{
      font-size: 11px;
      color: var(--gold);
      margin-top: 6px;
      letter-spacing: 0.5px;
    }}

    article {{
      background: var(--white);
      border-left: 3px solid var(--seafoam);
      padding: 24px 28px;
      margin-bottom: 16px;
    }}

    .meta {{
      font-size: 10px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 8px;
    }}

    h2 {{
      font-size: 15px;
      font-weight: bold;
      line-height: 1.45;
      margin-bottom: 12px;
      color: var(--dark-blue);
    }}

    h2 a {{
      color: inherit;
      text-decoration: none;
    }}

    h2 a:hover {{ color: var(--cyan); }}

    p {{
      font-size: 13.5px;
      line-height: 1.8;
      color: #3a4255;
    }}

    .empty {{
      color: var(--text-muted);
      font-style: italic;
      padding: 40px 0;
    }}

    /* Further reading */
    .further {{
      margin-top: 32px;
      background: var(--white);
      padding: 24px 28px;
      border-left: 3px solid var(--gold);
    }}

    .further-label {{
      font-size: 10px;
      letter-spacing: 2.5px;
      text-transform: uppercase;
      color: var(--gold);
      margin-bottom: 16px;
      font-weight: bold;
    }}

    .fr-item {{
      display: flex;
      gap: 16px;
      align-items: baseline;
      padding: 9px 0;
      border-bottom: 1px solid var(--border);
    }}

    .fr-item:last-child {{ border-bottom: none; }}

    .fr-source {{
      font-size: 10px;
      color: var(--text-muted);
      white-space: nowrap;
      min-width: 80px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }}

    .fr-item a {{
      font-size: 13px;
      color: var(--text-primary);
      text-decoration: none;
      line-height: 1.5;
    }}

    .fr-item a:hover {{ color: var(--cyan); }}

    footer {{
      margin-top: 32px;
      padding: 16px 0;
      font-size: 11px;
      color: var(--text-muted);
      text-align: center;
      border-top: 1px solid var(--border);
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
      Generated at 02:00 SAST &nbsp;·&nbsp; Sources: Reuters, CNBC, FT &nbsp;·&nbsp; Summaries via Groq
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
