"""
Daily Tech & AI News Digest
Fetches top tech/AI headlines from Reuters, CNBC, FT RSS feeds,
splits into company-specific summaries and thematic further reading,
summarises via Groq API, writes a clean HTML page for GitHub Pages.
"""

import os
import time
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_KEY          = os.environ["GROQ_API_KEY"]
GROQ_URL          = "https://api.groq.com/openai/v1/chat/completions"
NEWS_API_KEY      = os.environ.get("NEWS_API_KEY", "")       # optional — NewsAPI free tier
MEDIASTACK_KEY    = os.environ.get("MEDIASTACK_API_KEY", "") # optional — Mediastack free tier
RAPIDAPI_KEY      = os.environ.get("RAPIDAPI_KEY", "")       # optional — Reuters RapidAPI

SAST  = timezone(timedelta(hours=2))
TODAY = datetime.now(SAST).strftime("%A, %d %B %Y")

# ── RSS feeds ─────────────────────────────────────────────────────────────────
FEEDS = [
    # Reuters
    ("Reuters",     "https://feeds.reuters.com/reuters/technologyNews"),
    ("Reuters",     "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters",     "https://feeds.reuters.com/reuters/companyNews"),
    ("Reuters",     "https://feeds.reuters.com/reuters/financialNews"),
    ("Reuters",     "https://feeds.reuters.com/reuters/mergersNews"),
    # CNBC
    ("CNBC",        "https://www.cnbc.com/id/19854910/device/rss/rss.html"),
    ("CNBC",        "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
    # FT
    ("FT",          "https://www.ft.com/rss/home/technology"),

    # TechCrunch
    ("TechCrunch",  "https://techcrunch.com/feed/"),
    # The Verge
    ("The Verge",   "https://www.theverge.com/rss/index.xml"),
    # WSJ Tech
    ("WSJ",         "https://feeds.a.dj.com/rss/RSSWSJD.xml"),
    # MyBroadband — SA tech news
    ("MyBroadband", "https://mybroadband.co.za/news/feed"),
]

# ── Keywords ──────────────────────────────────────────────────────────────────

# Broad company signal — catches any named company in a tech/AI/finance context.
# Kept intentionally wide; the AI summariser and FINANCIAL_EVENTS filter do the
# precision work. Add names here only if they are consistently missed.
# Sources routed straight to Further Reading — never full summaries (paywalled)
FURTHER_READING_ONLY_SOURCES = {"FT", "WSJ"}

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
    # Startup/event content
    "startup battlefield", "techcrunch disrupt", "demo day",
    "pitch competition", "accelerator", "hackathon",
    # Travel/visit pieces without operational news
    "food watch", "tracking website",
    # Entertainment / gaming — not relevant to TMT financials
    "game launch", "video game", "early access", "xbox game", "xbox series",
    "playstation", "nintendo", "forza", "fable", "halo", "call of duty",
    "grand theft", "fifa", "ea sports", "gaming studio", "game studio",
    "film", "movie", "season 2", "season 3", "tv show", "box office",
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
    "ceo tells", "tells cnbc", "tells reuters", "tells ft",
    "ceo says", "exec says", "chief says",
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
    # CEO visit/travel pieces
    "obsessing over", "ceo visit", "ceo tour", "ceo trip",
    "tracks jensen", "follows jensen", "follows ceo",
]

MAX_SUMMARIES = 6    # articles that get full summaries
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


def relevance_score(title: str, summary: str) -> int:
    """Score articles by market relevance — higher = more important."""
    text  = (title + " " + summary).lower()
    score = 0

    # Tier 1 companies — highest investor interest
    tier1 = {"nvidia", "apple", "microsoft", "google", "alphabet", "meta",
             "amazon", "tesla", "openai", "anthropic", "tsmc", "samsung",
             "asml", "broadcom", "qualcomm", "spacex"}
    if any(c in text for c in tier1):
        score += 2

    # Direct financial events — most market-moving
    financial = {"earnings", "results", "revenue", "eps", "guidance",
                 "profit", "loss", "margin", "raised", "beat", "miss"}
    if any(f in text for f in financial):
        score += 3

    # Major corporate events
    corporate = {"acquisition", "acquires", "merger", "ipo", "layoffs",
                 "cuts jobs", "funding", "valuation", "buyback", "dividend"}
    if any(c in text for c in corporate):
        score += 2

    # AI and chips — high relevance for TMT
    if any(k in text for k in {"ai", "chip", "semiconductor", "llm", "gpu"}):
        score += 1

    # Share price reaction — confirms market significance
    if any(k in text for k in {"shares", "stock", "rose", "fell", "surged", "plunged"}):
        score += 1

    return score


def fetch_full_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsDigestBot/1.0)"}
        resp    = requests.get(url, headers=headers, timeout=5)
        soup    = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = " ".join(p.get_text(separator=" ") for p in soup.find_all("p"))
        return text[:3000].strip()
    except Exception:
        return ""


def fetch_newsapi(cutoff: datetime) -> list[dict]:
    """Fetch Reuters articles via NewsAPI as a supplement to RSS."""
    if not NEWS_API_KEY:
        return []
    try:
        params = {
            "q":        "nvidia OR apple OR microsoft OR google OR meta OR amazon OR openai OR anthropic OR AI earnings",
            "apiKey":   NEWS_API_KEY,
            "pageSize": 50,
            "language": "en",
            "sortBy":   "publishedAt",
            "from":     cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        articles = []
        for a in resp.json().get("articles", []):
            published = None
            if a.get("publishedAt"):
                try:
                    published = datetime.strptime(
                        a["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=timezone.utc)
                except Exception:
                    pass
            articles.append({
                "source":    "Reuters",
                "title":     a.get("title", ""),
                "summary":   a.get("description", ""),
                "link":      a.get("url", ""),
                "published": published,
            })
        return articles
    except Exception as e:
        print(f"  NewsAPI error: {e}")
        return []


def fetch_gdelt(cutoff: datetime) -> list[dict]:
    """Fetch tech/AI news from GDELT — free, no API key required.
    Queries the GDELT DOC 2.0 API for recent Reuters and major tech news."""
    try:
        # GDELT DOC API — queries last 24h of news
        params = {
            "query":      "reuters technology AI earnings acquisition layoffs",
            "mode":       "artlist",
            "maxrecords": 75,
            "format":     "json",
            "timespan":   "1d",
            "sort":       "datedesc",
        }
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for a in data.get("articles", []):
            published = None
            if a.get("seendate"):
                try:
                    published = datetime.strptime(
                        a["seendate"], "%Y%m%dT%H%M%SZ"
                    ).replace(tzinfo=timezone.utc)
                except Exception:
                    pass
            if published and published < cutoff:
                continue
            articles.append({
                "source":    a.get("domain", "GDELT"),
                "title":     a.get("title", ""),
                "summary":   "",
                "link":      a.get("url", ""),
                "published": published,
            })
        print(f"  GDELT: found {len(articles)} articles.")
        return articles
    except Exception as e:
        print(f"  GDELT error: {e}")
        return []


def fetch_mediastack(cutoff: datetime) -> list[dict]:
    """Fetch Reuters and tech news via Mediastack API (free tier)."""
    if not MEDIASTACK_KEY:
        return []
    try:
        params = {
            "access_key": MEDIASTACK_KEY,
            "sources":    "reuters,cnbc,techcrunch,the-verge,wsj",
            "categories": "technology,business",
            "languages":  "en",
            "limit":      50,
            "sort":       "published_desc",
        }
        resp = requests.get(
            "http://api.mediastack.com/v1/news",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        articles = []
        for a in resp.json().get("data", []):
            published = None
            if a.get("published_at"):
                try:
                    published = datetime.strptime(
                        a["published_at"], "%Y-%m-%dT%H:%M:%S+%f"
                    ).replace(tzinfo=timezone.utc)
                except Exception:
                    try:
                        published = datetime.fromisoformat(
                            a["published_at"].replace("Z", "+00:00")
                        )
                    except Exception:
                        pass
            if published and published < cutoff:
                continue
            articles.append({
                "source":    a.get("source", "Mediastack"),
                "title":     a.get("title", ""),
                "summary":   a.get("description", ""),
                "link":      a.get("url", ""),
                "published": published,
            })
        print(f"  Mediastack: found {len(articles)} articles.")
        return articles
    except Exception as e:
        print(f"  Mediastack error: {e}")
        return []


def fetch_rapidapi_reuters(cutoff: datetime) -> list[dict]:
    """Fetch Reuters articles via RapidAPI Reuters Business and Financial News.
    Uses date-range endpoint to pull articles from the cutoff date to today."""
    if not RAPIDAPI_KEY:
        return []
    # Disabled — RapidAPI free tier only 30 calls/month (insufficient for daily use)
    # Re-enable by removing the return [] below once on a paid plan
    return []
    try:
        now_sast    = datetime.now(SAST)
        cutoff_sast = cutoff.astimezone(SAST)
        date_from   = cutoff_sast.strftime("%Y-%m-%d")
        date_to     = now_sast.strftime("%Y-%m-%d")
        # Ensure at least a 2-day window — API needs overlap to find articles
        from_dt = datetime.strptime(date_from, "%Y-%m-%d")
        to_dt   = datetime.strptime(date_to,   "%Y-%m-%d")
        if (to_dt - from_dt).days < 2:
            date_from = (from_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        # Ensure at least a 2-day window so API returns results
        from datetime import date as _date
        if date_from == date_to:
            from datetime import timedelta as _td
            date_from = (now_sast - _td(days=1)).strftime("%Y-%m-%d")
        print(f"  RapidAPI date range: {date_from} to {date_to}")

        headers = {
            "Content-Type":  "application/json",
            "x-rapidapi-host": "reuters-business-and-financial-news.p.rapidapi.com",
            "x-rapidapi-key":  RAPIDAPI_KEY,
        }

        articles = []
        # Fetch from tech/business category endpoints specifically
        # Category IDs from API: 238=Business, 243=Technology, 260=Media&Telecom,
        # 374=China, 254=Retail&Consumer, 381=Aerospace&Defense
        category_ids = [243, 238, 260, 374]

        def parse_article(a, cutoff):
            pub_obj = a.get("publishedAt") or {}
            pub_str = pub_obj.get("date", "") if isinstance(pub_obj, dict) else str(pub_obj)
            published = None
            if pub_str:
                for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]:
                    try:
                        published = datetime.strptime(
                            pub_str[:26].strip(), fmt
                        ).replace(tzinfo=timezone.utc)
                        break
                    except Exception:
                        continue
            if published and published < cutoff:
                return None
            title = (a.get("articlesName") or a.get("title") or "")
            if not title:
                return None
            summary = (a.get("articlesShortDescription") or a.get("description") or "")
            link = a.get("url") or a.get("link") or ""
            if not link:
                supplier = a.get("urlSupplier") or a.get("canonicalSupplier", "")
                if supplier:
                    link = "https://www.reuters.com" + supplier
            return {"source": "Reuters", "title": title,
                    "summary": summary, "link": link, "published": published}

        seen = set()
        for cat_id in category_ids:
            for offset in [0, 20]:
                url = f"https://reuters-business-and-financial-news.p.rapidapi.com/get-articles-by-category-id/{cat_id}/{offset}/20"
                print(f"  RapidAPI request: {url}")
                resp = requests.get(url, headers=headers, timeout=15)
                print(f"  RapidAPI status: {resp.status_code}")
                if resp.status_code == 429:
                    print("  RapidAPI rate limit — stopping.")
                    break
                if resp.status_code != 200:
                    continue
                data  = resp.json()
                batch = data.get("articles") or data.get("items") or []
                if isinstance(data, list):
                    batch = data
                for a in batch:
                    art = parse_article(a, cutoff)
                    if art and art["title"] not in seen:
                        seen.add(art["title"])
                        articles.append(art)
                time.sleep(0.5)

        print(f"  RapidAPI Reuters: found {len(articles)} articles.")
        if len(articles) > 0:
            sample = [a["title"][:60] for a in articles[:5]]
            print(f"  RapidAPI sample titles: {sample}")
        return articles
    except Exception as e:
        print(f"  RapidAPI Reuters error: {e}")
        return []


def scrape_reuters(cutoff: datetime) -> list[dict]:
    """Scrape Reuters technology and business section pages directly.
    Catches articles that miss the RSS feed or NewsAPI delay."""
    pages = [
        "https://www.reuters.com/technology/",
        "https://www.reuters.com/business/",
        "https://www.reuters.com/technology/artificial-intelligence/",
        "https://www.reuters.com/markets/companies/",
    ]
    articles = []
    seen_links = set()

    for page_url in pages:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; NewsDigestBot/1.0)",
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = requests.get(page_url, headers=headers, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Reuters article links follow /YYYY/MM/DD/ pattern
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if not href.startswith("/"):
                    continue
                # Must look like a Reuters article path with date
                import re
                if not re.search(r"/20[0-9]{2}/[0-9]{2}/[0-9]{2}/", href):
                    continue
                # Skip non-article paths
                skip_paths = ["/video/", "/graphics/", "/pictures/", "/author/",
                              "/tag/", "/section/", "/markets/companies/"]
                if any(p in href for p in skip_paths):
                    continue

                full_url = "https://www.reuters.com" + href.split("?")[0]
                if full_url in seen_links:
                    continue
                seen_links.add(full_url)

                title = a_tag.get_text(separator=" ").strip()
                # Skip nav links, short strings, and non-article text
                if len(title) < 20 or len(title) > 250:
                    continue

                articles.append({
                    "source":    "Reuters",
                    "title":     title,
                    "summary":   "",
                    "link":      full_url,
                    "published": None,  # timestamp fetched later if needed
                })

        except Exception as e:
            print(f"  Reuters scrape error ({page_url}): {e}")

    print(f"  Reuters scrape: found {len(articles)} candidate links.")
    return articles


def fetch_articles() -> tuple[list[dict], list[dict]]:
    """Returns (summary_articles, further_reading_articles)."""
    summaries_pool = []
    further_pool   = []
    cutoff         = get_cutoff()

    # Fetch all supplementary sources in parallel
    fetch_tasks = {
        "newsapi":    lambda: fetch_newsapi(cutoff),
        "gdelt":      lambda: fetch_gdelt(cutoff),
        "mediastack": lambda: fetch_mediastack(cutoff),
        "rapidapi":   lambda: fetch_rapidapi_reuters(cutoff),
        "scrape":     lambda: scrape_reuters(cutoff),
    }
    supplementary = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fn): name for name, fn in fetch_tasks.items()}
        for future in as_completed(futures):
            try:
                supplementary.extend(future.result())
            except Exception as e:
                print(f"  Source error ({futures[future]}): {e}")

    all_entries = list(supplementary)

    # Fetch all RSS feeds in parallel then process sequentially
    with ThreadPoolExecutor(max_workers=min(len(FEEDS), 8)) as _ex:
        _feeds = list(_ex.map(lambda su: (su[0], feedparser.parse(su[1])), FEEDS))

    for source, feed in _feeds:
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

            # Tier 1: company-specific + financial event + not soft → full summary
            # Paywalled sources go straight to further reading
            if source in FURTHER_READING_ONLY_SOURCES:
                if (has_company(text) or has_thematic(text)) and article["link"]:
                    if not is_duplicate(title, further_pool) and                        not is_duplicate(title, summaries_pool):
                        further_pool.append(article)
                continue

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

ABSOLUTE RULE — READ FIRST: Write in proper English sentence case. This means: first word of each sentence is capitalised, proper nouns (company names, people, places, acronyms) are capitalised, and ALL OTHER WORDS are lowercase. Example of CORRECT output: "Nvidia reported Q2 revenue of USD81bn, beating estimates."  Example of WRONG output: "Nvidia Reported Q2 Revenue Of USD81bn, Beating Estimates." If your output has most words capitalised, you have failed this instruction.

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
- Sentence case ONLY. Only the first word of each sentence and proper nouns get capitals. NEVER capitalise every word like title case — "Openai Is In Talks" is wrong; "OpenAI is in talks" is correct.
- Double space after every sentence.
- Maximum 20 words per sentence. Count carefully and split any sentence over 20 words.
- Do not speculate. Only report what the article explicitly states. Never write "may", "could", "might", "potential" unless directly quoting the article.
- Do not pad. One tight paragraph is better than two vague ones.
- Use "US" not "U.S".
- bn = billion, m = million, tn = trillion, k = thousand.
- Spaces in thousands: "10 000" not "10,000".
- ISO currency codes: USD, EUR, GBP, KRW, JPY, CNY, ZAR.
- NEVER use $ or £ or € symbols. Always write USD, GBP, EUR in full.
- Correct: "USD4.2bn" — Wrong: "$4.2bn" or "$4.2 billion".
- Est: used inline for consensus (e.g. "revenue of USD4.2bn (Est: USD3.9bn)").
- Use "c." for approximate values.
- Dates: 15 Sep '26. FY '26. CY '26. YTD. y/y. q/q.
- Ratings in full capitals: OUTPERFORM, UNDERPERFORM, MARKET PERFORM.
- No bullet points. No repeated information. Never start consecutive sentences with the same subject. Never repeat the company name more than twice per paragraph.
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


def group_articles(articles: list[dict]) -> list[dict]:
    """Group articles about the same company/event into merged entries.
    Returns a list of grouped articles, each with combined sources and text."""
    if not articles:
        return []

    groups = []
    used = set()

    for i, article in enumerate(articles):
        if i in used:
            continue

        group = {
            "titles":    [article["title"]],
            "sources":   [article["source"]],
            "links":     [article["link"]],
            "summary":   article["summary"],
            "published": article["published"],
            # Keep the most informative title as the display title
            "title":     article["title"],
            "link":      article["link"],
            "source":    article["source"],
        }
        used.add(i)

        # Find other articles covering the same company/event
        title_a = article["title"].lower()
        words_a = set(w for w in title_a.split() if len(w) > 4)

        for j, other in enumerate(articles):
            if j in used or j == i:
                continue
            title_b = other["title"].lower()
            words_b = set(w for w in title_b.split() if len(w) > 4)
            # Match if 3+ meaningful words overlap (same story, different source)
            overlap = words_a & words_b
            if len(overlap) >= 3:
                group["titles"].append(other["title"])
                group["sources"].append(other["source"])
                group["links"].append(other["link"])
                group["summary"] += " " + other["summary"]
                used.add(j)

        groups.append(group)

    return groups


def fix_capitalisation(text: str) -> str:
    """Fix Title Case output from Groq — convert to proper sentence case."""
    import re
    words = text.split()
    if len(words) < 10:
        return text
    # Count title-cased words (capitalised but not all-caps, longer than 4 chars)
    title_cased = sum(
        1 for w in words
        if len(w) > 4 and w[0].isupper() and w[1:].islower() and not w.isupper()
    )
    if title_cased / len(words) < 0.35:
        return text  # Already fine

    # Known proper nouns and acronyms to preserve capitalisation
    proper = {
        "nvidia", "apple", "microsoft", "google", "alphabet", "meta", "amazon",
        "tesla", "openai", "anthropic", "intel", "amd", "qualcomm", "broadcom",
        "tsmc", "samsung", "asml", "arm", "spacex", "waymo", "uber", "netflix",
        "spotify", "airbnb", "palantir", "salesforce", "oracle", "ibm", "adobe",
        "stripe", "marvell", "crowdstrike", "bluesky", "helion", "manus", "flex",
        "reuters", "cnbc", "techcrunch", "bloomberg", "wsj", "softbank",
        "ai", "ceo", "cfo", "coo", "cto", "us", "eu", "uk", "usd", "eur",
        "gbp", "zar", "cny", "krw", "jpy", "q1", "q2", "q3", "q4", "fy",
        "cy", "ytd", "arr", "gpu", "ipo", "aws", "azure", "gcp", "llm",
        "nasdaq", "s&p", "etf", "ebitda", "eps", "r&d", "m&a",
    }
    # Split on double-space sentence boundaries
    sentences = re.split(r"  +", text)
    fixed = []
    for sent in sentences:
        if not sent.strip():
            continue
        twords = sent.split()
        out = []
        for i, w in enumerate(twords):
            # Strip punctuation for matching
            clean = w.strip('.,;:!?()[]').strip('"').strip("'")
            low   = clean.lower()
            if low in proper:
                # Restore proper casing
                if low in {"ai", "ceo", "cfo", "coo", "cto", "us", "eu", "uk",
                           "usd", "eur", "gbp", "zar", "cny", "krw", "jpy",
                           "q1", "q2", "q3", "q4", "fy", "cy", "ytd", "arr",
                           "gpu", "ipo", "aws", "azure", "gcp", "llm", "etf",
                           "ebitda", "eps", "nasdaq", "s&p"}:
                    out.append(w.replace(clean, clean.upper()))
                else:
                    out.append(w.replace(clean, clean.capitalize()))
            elif i == 0:
                out.append(w[0].upper() + w[1:].lower() if len(w) > 1 else w.upper())
            else:
                out.append(w.lower())
        fixed.append(" ".join(out))
    return "  ".join(fixed)


def summarise(article: dict) -> str:
    """Summarise a single article or a merged group of articles."""
    links  = article.get("links", [article["link"]])
    titles = article.get("titles", [article["title"]])

    # Use prefetched content if available, otherwise fetch now
    if article.get("_prefetched"):
        combined_text = article["_prefetched"]
    else:
        combined_text = ""
        for title, link in zip(titles, links):
            body = fetch_full_text(link) if link else ""
            combined_text += "\n\n--- Source: {} | Title: {} ---\n".format(
                article.get("source", ""), title
            )
            combined_text += body if body else article.get("summary", "")

    prompt_content = "Company/topic: {}\n".format(article["title"])
    if len(titles) > 1:
        prompt_content += "Note: {} sources cover this story — synthesise into one summary.\n".format(len(titles))
    prompt_content += combined_text.strip()

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_content},
        ],
        "max_tokens": 600,
        "temperature": 0.3,
    }

    for attempt in range(2):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(GROQ_KEY),
                },
                data=json.dumps(payload),
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print("    Groq rate limit hit — waiting {}s before retry...".format(wait))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # Enforce ISO currency — replace any $ with USD
            import re as _re
            text = _re.sub(r"[$]([0-9])", r"USD", text)
            text = _re.sub(r"[$]([a-zA-Z])", r"USD ", text)
            # Fix Title Case if Groq ignores capitalisation rules
            text = fix_capitalisation(text)
            return text
        except Exception as e:
            if attempt == 0:
                print("    Groq error, retrying in 60s: {}".format(e))
                time.sleep(60)
            else:
                return "[Summary unavailable: {}]".format(e)
    return "[Summary unavailable: max retries exceeded]"


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(articles: list[dict], summaries: list[str],
               further: list[dict]) -> str:

    # ── Main article cards ────────────────────────────────────────────────────
    cards = ""
    for article, summary in zip(articles, summaries):
        pub_str = ""
        if article["published"]:
            pub_str = article["published"].astimezone(SAST).strftime("%H:%M SAST")

        # Normalise line breaks then add spacing between paragraphs
        summary_clean = summary.replace("\r\n", "\n").replace("\r", "\n")
        import re as _re
        summary_clean = _re.sub(r"\n{2,}", "\n\n", summary_clean).strip()
        safe_summary = summary_clean.replace("\n\n", '</p><p style="margin-top:16px;">').replace("\n", " ")

        # Build source badges — one per source if grouped
        sources  = article.get("sources", [article["source"]])
        links    = article.get("links",   [article["link"]])
        src_html = " &nbsp;·&nbsp; ".join(
            f'<a href="{l}" target="_blank" rel="noopener" style="color:inherit;">{s}</a>'
            for s, l in zip(sources, links)
        )

        cards += f"""
        <article>
          <div class="meta">{src_html} &nbsp;·&nbsp; {pub_str}</div>
          <h2>{article['title']}</h2>
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
  <title>TMT Snapshot — {TODAY}</title>
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
      <h1>TMT Snapshot</h1>
      <div class="datestamp">{TODAY}</div>
    </header>

    {cards}
    {further_html}

    <footer>
      Generated at 06:00 SAST &nbsp;·&nbsp; Sources: Reuters, CNBC, FT, TechCrunch, The Verge, WSJ &nbsp;·&nbsp; Summaries via Groq
    </footer>
  </div>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[{TODAY}] Fetching articles...")
    articles, further = fetch_articles()
    print(f"  {len(articles)} articles to summarise, {len(further)} for further reading.")

    # Group articles covering the same story before summarising
    grouped = group_articles(articles)
    saved   = len(articles) - len(grouped)
    if saved > 0:
        print(f"  Grouped into {len(grouped)} stories (saved {saved} API calls).")

    # Pre-fetch full article text in parallel before Groq calls
    print("  Pre-fetching article content...")
    def _prefetch(article):
        links  = article.get("links", [article["link"]])
        titles = article.get("titles", [article["title"]])
        texts  = []
        with ThreadPoolExecutor(max_workers=len(links)) as ex:
            fetched = list(ex.map(lambda l: fetch_full_text(l) if l else "", links))
        for title, body in zip(titles, fetched):
            texts.append("\n\n--- Source: {} | Title: {} ---\n{}".format(
                article.get("source", ""), title, body or article.get("summary", "")
            ))
        article["_prefetched"] = "\n".join(texts)
        return article

    with ThreadPoolExecutor(max_workers=len(grouped)) as ex:
        grouped = list(ex.map(_prefetch, grouped))

    summaries = []
    for i, article in enumerate(grouped, 1):
        src_list = ", ".join(article.get("sources", [article["source"]]))
        print(f"  Summarising {i}/{len(grouped)} [{src_list}]: {article['title'][:50]}...")
        summaries.append(summarise(article))
        if i < len(grouped):
            time.sleep(4)   # avoid Groq free tier rate limit

    html = build_html(grouped, summaries, further)

    out = Path("docs")
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    print("  Written to docs/index.html")


if __name__ == "__main__":
    main()
