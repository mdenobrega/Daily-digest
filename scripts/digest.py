"""
Daily Tech & AI News Digest
Fetches top tech/AI headlines from Reuters, CNBC, FT RSS feeds,
filters for major company names and market-moving events,
summarises via Claude API, and emails a clean digest.
"""

import os
import re
import smtplib
import feedparser
import anthropic
import requests
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
RECIPIENT_EMAIL = "mdenobrega19@gmail.com"
SENDER_EMAIL    = os.environ["GMAIL_ADDRESS"]      # set in GitHub secrets
GMAIL_APP_PASS  = os.environ["GMAIL_APP_PASSWORD"] # set in GitHub secrets
ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]  # set in GitHub secrets

# SAST is UTC+2
SAST = timezone(timedelta(hours=2))
TODAY = datetime.now(SAST).strftime("%A, %d %B %Y")

# ── RSS feeds ─────────────────────────────────────────────────────────────────
FEEDS = [
    # Reuters
    ("Reuters", "https://feeds.reuters.com/reuters/technologyNews"),
    ("Reuters", "https://feeds.reuters.com/reuters/businessNews"),
    # CNBC
    ("CNBC",    "https://www.cnbc.com/id/19854910/device/rss/rss.html"),  # tech
    ("CNBC",    "https://www.cnbc.com/id/10000664/device/rss/rss.html"),  # earnings
    # FT (public feed — headlines only, no paywall required)
    ("FT",      "https://www.ft.com/rss/home/technology"),
]

# ── Filter keywords ───────────────────────────────────────────────────────────
COMPANY_KEYWORDS = [
    # Big Tech
    "nvidia", "apple", "microsoft", "google", "alphabet", "meta", "amazon",
    "tesla", "openai", "anthropic", "intel", "amd", "qualcomm", "broadcom",
    "tsmc", "samsung", "asml", "arm", "palantir", "salesforce", "oracle",
    "ibm", "sap", "adobe", "snowflake", "databricks", "huawei",
    # Hyperscalers / cloud
    "aws", "azure", "gcp", "cloudflare", "datadog",
    # Social / consumer tech
    "netflix", "spotify", "uber", "lyft", "airbnb", "x.com", "twitter",
    # Semiconductors
    "micron", "sk hynix", "western digital", "seagate",
]

EVENT_KEYWORDS = [
    "earnings", "results", "revenue", "profit", "guidance", "forecast",
    "layoffs", "cuts jobs", "acquisition", "merger", "ipo", "buyback",
    "dividend", "capex", "ai", "chip", "model", "launch", "regulation",
    "antitrust", "fine", "lawsuit", "ceo", "partnership", "deal",
]

MAX_ARTICLES = 12   # cap to control API cost
MAX_AGE_HOURS = 26  # articles published within the last ~26 hours


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_relevant(title: str, summary: str) -> bool:
    text = (title + " " + summary).lower()
    has_company = any(k in text for k in COMPANY_KEYWORDS)
    has_event   = any(k in text for k in EVENT_KEYWORDS)
    return has_company and has_event


def fetch_full_text(url: str) -> str:
    """Best-effort extraction of article body text."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsDigestBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove nav, ads, scripts
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(separator=" ") for p in paragraphs)
        # Trim to ~3 000 chars to stay within token budget
        return text[:3000].strip()
    except Exception:
        return ""


def fetch_articles() -> list[dict]:
    articles = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)

    for source, url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            # Parse published date
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

            # Avoid duplicates by title similarity
            if any(a["title"].lower()[:60] == title.lower()[:60] for a in articles):
                continue

            articles.append({
                "source":    source,
                "title":     title,
                "summary":   summary,
                "link":      link,
                "published": published,
            })

    # Sort by recency, cap at MAX_ARTICLES
    articles.sort(key=lambda x: x["published"] or datetime.min.replace(tzinfo=timezone.utc),
                  reverse=True)
    return articles[:MAX_ARTICLES]


# ── Summarisation prompt ──────────────────────────────────────────────────────
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
- No bullet points.
- No repeated information.
- No filler phrases like "management remains optimistic" unless backed by data.

If the article lacks enough data for a full two-paragraph summary, write one strong paragraph. If the article is behind a paywall with no extractable content, write a one-sentence note and skip.
"""


def summarise(client: anthropic.Anthropic, article: dict) -> str:
    body = fetch_full_text(article["link"]) if article["link"] else ""
    content = f"Title: {article['title']}\n\nSource: {article['source']}\n\n"
    if body:
        content += f"Article text:\n{body}"
    else:
        content += f"Summary/excerpt:\n{article['summary']}"

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"[Summary unavailable: {e}]"


# ── Email builder ─────────────────────────────────────────────────────────────

def build_email(articles: list[dict], summaries: list[str]) -> tuple[str, str]:
    """Returns (subject, html_body)."""
    subject = f"Daily Tech & AI Digest — {TODAY}"

    html_parts = [f"""
    <html><body style="font-family: Georgia, serif; max-width: 700px; margin: auto;
                       background: #ffffff; color: #1a1a1a; padding: 32px;">
    <h2 style="font-size:18px; border-bottom:2px solid #1a1a1a; padding-bottom:8px;
               margin-bottom:24px; letter-spacing:0.5px;">
        DAILY TECH &amp; AI DIGEST &mdash; {TODAY.upper()}
    </h2>
    """]

    for article, summary in zip(articles, summaries):
        pub_str = ""
        if article["published"]:
            pub_sast = article["published"].astimezone(SAST)
            pub_str  = pub_sast.strftime("%H:%M SAST")

        html_parts.append(f"""
        <div style="margin-bottom:32px; padding-bottom:24px;
                    border-bottom:1px solid #e0e0e0;">
            <p style="font-size:11px; color:#888; margin:0 0 6px 0;
                      letter-spacing:0.8px; text-transform:uppercase;">
                {article['source']} &nbsp;|&nbsp; {pub_str}
            </p>
            <h3 style="font-size:15px; font-weight:bold; margin:0 0 10px 0;
                       line-height:1.4;">
                <a href="{article['link']}" style="color:#1a1a1a; text-decoration:none;">
                    {article['title']}
                </a>
            </h3>
            <p style="font-size:13.5px; line-height:1.75; margin:0; color:#2a2a2a;">
                {summary.replace(chr(10), '<br>')}
            </p>
        </div>
        """)

    if not articles:
        html_parts.append("""
        <p style="color:#888; font-size:13px;">
            No major tech/AI articles matched today's filter criteria.
            Check back tomorrow.
        </p>
        """)

    html_parts.append("</body></html>")
    return subject, "".join(html_parts)


# ── Email sender ──────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, GMAIL_APP_PASS)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[{TODAY}] Fetching articles...")
    articles = fetch_articles()
    print(f"  Found {len(articles)} relevant articles.")

    if not articles:
        subject, html = build_email([], [])
        send_email(subject, html)
        print("  Sent empty digest.")
        return

    client   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    summaries = []

    for i, article in enumerate(articles, 1):
        print(f"  Summarising {i}/{len(articles)}: {article['title'][:60]}...")
        summaries.append(summarise(client, article))

    subject, html = build_email(articles, summaries)
    send_email(subject, html)
    print(f"  Digest sent to {RECIPIENT_EMAIL}.")


if __name__ == "__main__":
    main()
