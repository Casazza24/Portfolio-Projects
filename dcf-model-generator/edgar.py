import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Matthew mattcasazza14@gmail.com"
}

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# The SEC asks for no more than 10 requests/second.
REQUEST_DELAY = 0.1

MIN_SECTION_LENGTH = 5000  # TOC entries sit only a few hundred chars apart

# Path for the local CIK cache file (next to this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CIK_CACHE_PATH = os.path.join(_SCRIPT_DIR, "cik_cache.json")
CIK_CACHE_MAX_AGE = 86400  # 24 hours in seconds

# ---------------------------------------------------------------------------
# Word lists for tone scoring (backward compatible)
# ---------------------------------------------------------------------------

positive_words = [
    "growth", "strong", "increased", "improved", "momentum",
    "expanding", "record", "exceeded", "favorable", "opportunity",
    "confident", "robust", "accelerat"
]

negative_words = [
    "decline", "decreased", "headwind", "challenging", "risk",
    "uncertain", "slowdown", "weakness", "pressure", "adverse",
    "difficult", "impair", "restructur"
]

# Stands in for a section we could not locate, so callers can read the same
# keys either way.
EMPTY_TONE = {
    "positive_count": 0,
    "negative_count": 0,
    "positive_hits": {},
    "negative_hits": {},
    "score": None,
}

guidance_keywords = [
    "expect", "anticipate", "guidance", "outlook", "forecast",
    "project", "target", "plan", "believe", "continue",
    "invest", "committed", "remain", "intend", "estimate",
    "approximately", "range of", "on track"
]

# --- Topic classification keyword sets ---

TOPIC_KEYWORDS = {
    "revenue": [
        "revenue", "sales", "top line", "top-line", "bookings", "backlog",
        "order", "volume",
    ],
    "demand": [
        "demand", "customer", "consumer", "end market", "end-market",
        "adoption", "penetration",
    ],
    "margins": [
        "margin", "gross profit", "operating income", "cost of goods",
        "cost of revenue", "profitability", "operating expense",
    ],
    "pricing": [
        "pricing", "price", "average selling price", "asp", "discount",
    ],
    "supply_chain": [
        "supply chain", "supply-chain", "logistics", "inventory",
        "shortage", "procurement", "supplier", "lead time",
    ],
    "macro": [
        "macroeconomic", "macro", "inflation", "interest rate",
        "recession", "economic environment", "geopolitical", "tariff",
        "trade", "regulatory",
    ],
    "fx": [
        "foreign exchange", "currency", "fx", "exchange rate",
        "foreign currency", "translation",
    ],
    "capex": [
        "capital expenditure", "capex", "cap ex", "capital spending",
        "investment in infrastructure",
    ],
    "liquidity": [
        "liquidity", "cash flow", "cash position", "debt", "credit facility",
        "borrowing", "dividend", "share repurchase", "buyback",
    ],
    "products": [
        "product", "launch", "innovation", "pipeline", "r&d",
        "research and development", "new platform", "next generation",
    ],
    "strategy": [
        "strategy", "strategic", "transformation", "restructur",
        "acquisition", "divestiture", "partnership", "joint venture",
        "long-term", "long term",
    ],
}

# ---------------------------------------------------------------------------
# Precompiled regex patterns (requirement #3)
# ---------------------------------------------------------------------------

ANY_ITEM_PATTERN = re.compile(r"^item\s*\d", re.IGNORECASE | re.MULTILINE)
TOC_CLUSTER_GAP = 3000
TITLE_LOOKAHEAD = 200

FORWARD_PATTERN = re.compile(
    r"\b(expect|expects|anticipat\w*|may|could|will|intend|plan|believe"
    r"|continue to|forecast|project|outlook|guidance|looking ahead"
    r"|going forward|in the future|over the coming|in the coming"
    r"|likely|potential)\b",
    re.IGNORECASE,
)

HISTORICAL_PATTERN = re.compile(
    r"\b(increased|decreased|was driven by|resulted in"
    r"|was primarily due|compared to the prior|compared to the"
    r"|year-over-year|year over year|quarter-over-quarter"
    r"|compared with|during the period|during the quarter"
    r"|during the year|in the prior|last year|last quarter"
    r"|previously|declined|grew)\b",
    re.IGNORECASE,
)

DOLLAR_PATTERN = re.compile(r"\$\s*[\d,.]+\s*(million|billion|thousand)?", re.IGNORECASE)
PERCENT_PATTERN = re.compile(r"[\d,.]+\s*%")
MATERIAL_PATTERN = re.compile(r"\bmaterial(ly)?\b", re.IGNORECASE)
SIGNIFICANT_PATTERN = re.compile(r"\bsignificant(ly)?\b", re.IGNORECASE)
INCREASE_DECREASE_PATTERN = re.compile(r"\b(increase|decrease)\b", re.IGNORECASE)

# Boilerplate phrases to deprioritize
BOILERPLATE_PATTERNS = [
    re.compile(r"forward-looking statements are not guarantees", re.IGNORECASE),
    re.compile(r"actual results may differ", re.IGNORECASE),
    re.compile(r"private securities litigation reform act", re.IGNORECASE),
    re.compile(r"safe harbor", re.IGNORECASE),
    re.compile(r"cautionary statement", re.IGNORECASE),
]

# Block-level HTML elements that should produce paragraph breaks
BLOCK_ELEMENTS = {
    "p", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "blockquote", "section", "article", "header", "footer",
    "table", "thead", "tbody", "tfoot",
}


# ---------------------------------------------------------------------------
# CIK cache helpers (requirement #1)
# ---------------------------------------------------------------------------

def _load_cik_cache():
    """Load the cached ticker-to-CIK mapping if it exists and is fresh."""
    if not os.path.exists(CIK_CACHE_PATH):
        return None
    age = time.time() - os.path.getmtime(CIK_CACHE_PATH)
    if age > CIK_CACHE_MAX_AGE:
        return None
    with open(CIK_CACHE_PATH, "r") as f:
        return json.load(f)


def _save_cik_cache(data):
    """Write the ticker-to-CIK mapping to disk."""
    with open(CIK_CACHE_PATH, "w") as f:
        json.dump(data, f)


def _get_ticker_map():
    """
    Return a dict mapping uppercase ticker -> {"cik": str, "name": str}.

    Uses a local cache file so we don't re-download the full list every run.
    Refreshes the cache if it is older than 24 hours or missing.
    """
    cached = _load_cik_cache()
    if cached is not None:
        return cached

    raw = sec_get(TICKERS_URL).json()
    ticker_map = {}
    for entry in raw.values():
        ticker_map[entry["ticker"].upper()] = {
            "cik": str(entry["cik_str"]).zfill(10),
            "name": entry["title"],
        }
    _save_cik_cache(ticker_map)
    return ticker_map


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def sec_get(url):
    """GET a SEC URL, then pause so we stay inside their rate limit."""
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return response


def lookup_cik(ticker):
    """Return (cik, company_name) for a ticker, or raise if it isn't listed."""
    ticker_map = _get_ticker_map()
    upper = ticker.upper()
    if upper not in ticker_map:
        raise ValueError(f"Ticker {ticker!r} not found in the SEC company list")
    entry = ticker_map[upper]
    return entry["cik"], entry["name"]


def find_latest_filing(cik, filing_type):
    """Return metadata for the most recent filing of filing_type, or raise."""
    filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    recent = sec_get(filings_url).json()["filings"]["recent"]

    for i in range(len(recent["form"])):
        if recent["form"][i] == filing_type:
            accession = recent["accessionNumber"][i].replace("-", "")
            primary_doc = recent["primaryDocument"][i]
            return {
                "filing_date": recent["filingDate"][i],
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{accession}/{primary_doc}"
                ),
            }

    raise ValueError(f"No recent {filing_type} found for CIK {cik}")


# ---------------------------------------------------------------------------
# HTML cleaning with paragraph preservation (requirement #2)
# ---------------------------------------------------------------------------

def fetch_clean_text(doc_url):
    """
    Download a filing and convert it to plain text, preserving paragraph
    structure by inserting double-newlines before block-level HTML elements.
    """
    filing_text = sec_get(doc_url).text
    soup = BeautifulSoup(filing_text, "html.parser")

    # Insert paragraph breaks before block-level elements so they survive
    # get_text(). This keeps the filing from collapsing into one huge string.
    for tag in soup.find_all(BLOCK_ELEMENTS):
        tag.insert_before("\n\n")

    raw = soup.get_text(separator="\n")
    lines = [line.strip() for line in raw.split("\n") if line.strip()]

    # Rejoin with single newlines; paragraph breaks are double-newlines
    # that we inserted above, which collapse into blank-line gaps.
    result_lines = []
    prev_blank = False
    for line in lines:
        if not line:
            prev_blank = True
            continue
        if prev_blank and result_lines:
            result_lines.append("")  # blank line = paragraph break
        result_lines.append(line)
        prev_blank = False

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Section extraction (unchanged logic)
# ---------------------------------------------------------------------------

def body_start_offset(text):
    """
    Offset just past the table of contents.

    Every filing opens with its item headings packed into one dense run, then
    repeats them spread across the document. Skipping that run is more reliable
    than trying to judge each heading on its own.
    """
    headings = list(ANY_ITEM_PATTERN.finditer(text))
    if not headings:
        return 0

    last = headings[0]
    clustered = 1
    for heading in headings[1:]:
        if heading.start() - last.start() > TOC_CLUSTER_GAP:
            break
        last = heading
        clustered += 1

    # A couple of nearby headings is just a short section, not a contents page.
    return last.end() if clustered >= 3 else 0


def extract_section(text, title_pattern, item, next_item, min_length=MIN_SECTION_LENGTH):
    """Pull the body of a section, skipping the contents page and cross-references."""
    start_pattern = re.compile(title_pattern, re.IGNORECASE)
    next_pattern = re.compile(
        rf"^item\s*{next_item}[.:\s\xa0—–-]", re.IGNORECASE | re.MULTILINE
    )
    heading_prefix = re.compile(
        rf"^(item\s*{item}[.:\s\xa0—–-]*)?$", re.IGNORECASE
    )
    item_pattern = re.compile(
        rf"^item\s*{item}[.:\s\xa0—–-]", re.IGNORECASE | re.MULTILINE
    )

    body = body_start_offset(text)

    item_anchors = list(item_pattern.finditer(text, body))
    title_anchors = [
        m for m in start_pattern.finditer(text, body)
        if heading_prefix.match(text[text.rfind("\n", 0, m.start()) + 1:m.start()])
    ]

    titled = [
        m for m in item_anchors
        if start_pattern.search(text, m.end(), m.end() + TITLE_LOOKAHEAD)
    ]
    untitled = [m for m in item_anchors if m not in titled]
    matches_end = list(next_pattern.finditer(text, body))

    for tier in (titled, untitled, title_anchors):
        for start_match in sorted(tier, key=lambda m: m.start()):
            for end_match in matches_end:
                if end_match.start() <= start_match.end():
                    continue
                if end_match.start() - start_match.end() < min_length:
                    continue
                return text[start_match.start():end_match.start()]

    return None


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def split_into_paragraphs(text):
    """
    Split section text into paragraphs, preserving subsection headings.

    Returns a list of dicts: {"text": str, "is_heading": bool}

    A heading is a short line (under 120 chars) that is either all-caps or
    does not end with a period.
    """
    if not text:
        return []

    raw_chunks = re.split(r"\n{2,}", text)
    paragraphs = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        is_heading = (
            len(chunk) < 120
            and (not chunk.endswith(".") or chunk == chunk.upper())
        )
        paragraphs.append({"text": chunk, "is_heading": is_heading})

    # If we only got one big block (common after HTML stripping), fall back to
    # splitting on lines that look like headings embedded in a single chunk.
    if len(paragraphs) == 1 and len(paragraphs[0]["text"]) > 2000:
        lines = paragraphs[0]["text"].split("\n")
        rebuilt = []
        current_lines = []
        for line in lines:
            stripped = line.strip()
            if (stripped
                    and len(stripped) < 120
                    and (not stripped.endswith(".") or stripped == stripped.upper())
                    and current_lines):
                rebuilt.append({
                    "text": "\n".join(current_lines).strip(),
                    "is_heading": False,
                })
                current_lines = [stripped]
            else:
                current_lines.append(stripped)
        if current_lines:
            text_block = "\n".join(current_lines).strip()
            if text_block:
                rebuilt.append({"text": text_block, "is_heading": False})
        if len(rebuilt) > 1:
            for p in rebuilt:
                if (len(p["text"]) < 120
                        and (not p["text"].endswith(".") or p["text"] == p["text"].upper())):
                    p["is_heading"] = True
            paragraphs = rebuilt

    return paragraphs


# ---------------------------------------------------------------------------
# Single-pass paragraph analysis (requirement #4)
# ---------------------------------------------------------------------------

def analyze_single_paragraph(text):
    """
    Analyze one paragraph in a single pass, detecting:
      - forward-looking language
      - dollar amounts and percentages
      - topic tags
      - boilerplate status
      - relevance score

    Returns a dict with all analysis results.
    """
    lowered = text.lower()

    # Forward-looking and historical markers
    has_forward = bool(FORWARD_PATTERN.search(text))
    has_historical = bool(HISTORICAL_PATTERN.search(text))
    if has_forward and has_historical:
        temporality = "mixed"
    elif has_forward:
        temporality = "forward_looking"
    elif has_historical:
        temporality = "historical"
    else:
        temporality = "neutral"

    # Dollar amounts and percentages
    has_dollar = bool(DOLLAR_PATTERN.search(text))
    has_percent = bool(PERCENT_PATTERN.search(text))

    # Topic classification
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            topics.append(topic)

    # Boilerplate detection (requirement #6)
    is_boilerplate = any(bp.search(text) for bp in BOILERPLATE_PATTERNS)

    # Relevance scoring (requirement #5)
    relevance = 0
    if has_forward:
        relevance += 2
    if has_dollar or has_percent:
        relevance += 1
    if MATERIAL_PATTERN.search(text):
        relevance += 1
    if SIGNIFICANT_PATTERN.search(text):
        relevance += 1
    if INCREASE_DECREASE_PATTERN.search(text):
        relevance += 1
    if len(topics) >= 2:
        relevance += 1

    # Deprioritize boilerplate
    if is_boilerplate:
        relevance -= 3

    # Tone scoring (for backward compatibility)
    positive_hits = {w: lowered.count(w) for w in positive_words if lowered.count(w)}
    negative_hits = {w: lowered.count(w) for w in negative_words if lowered.count(w)}
    pos_count = sum(positive_hits.values())
    neg_count = sum(negative_hits.values())
    total = pos_count + neg_count
    if total == 0:
        tone_label = "neutral"
        tone = None
    elif pos_count / total > 0.6:
        tone_label = "positive"
        tone = pos_count / total
    elif pos_count / total < 0.4:
        tone_label = "negative"
        tone = pos_count / total
    else:
        tone_label = "neutral"
        tone = pos_count / total

    return {
        "temporality": temporality,
        "has_forward": has_forward,
        "has_historical": has_historical,
        "has_dollar": has_dollar,
        "has_percent": has_percent,
        "topics": topics,
        "is_boilerplate": is_boilerplate,
        "relevance": relevance,
        "sentiment": {
            "score": tone,
            "label": tone_label,
            "positive_count": pos_count,
            "negative_count": neg_count,
        },
    }


# ---------------------------------------------------------------------------
# Section-level analysis
# ---------------------------------------------------------------------------

def tone_score(text):
    """Count positive/negative word hits and return the tone breakdown."""
    lowered = text.lower()
    positive_hits = {w: lowered.count(w) for w in positive_words if lowered.count(w)}
    negative_hits = {w: lowered.count(w) for w in negative_words if lowered.count(w)}
    positive_count = sum(positive_hits.values())
    negative_count = sum(negative_hits.values())
    total = positive_count + negative_count

    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "score": positive_count / total if total else None,
    }


def analyze_paragraphs(text):
    """
    Full paragraph-level analysis of a section using single-pass analysis.

    Returns (analyzed_paragraphs, aggregate_tone_score).
    Each paragraph dict has:
        text, is_heading, topics, temporality, sentiment, relevance,
        is_boilerplate, has_dollar, has_percent
    """
    paragraphs = split_into_paragraphs(text)
    analyzed = []
    total_pos = 0
    total_neg = 0

    for para in paragraphs:
        analysis = analyze_single_paragraph(para["text"])
        total_pos += analysis["sentiment"]["positive_count"]
        total_neg += analysis["sentiment"]["negative_count"]

        analyzed.append({
            "text": para["text"],
            "is_heading": para["is_heading"],
            "topics": analysis["topics"],
            "temporality": analysis["temporality"],
            "sentiment": analysis["sentiment"],
            "relevance": analysis["relevance"],
            "is_boilerplate": analysis["is_boilerplate"],
            "has_dollar": analysis["has_dollar"],
            "has_percent": analysis["has_percent"],
        })

    total = total_pos + total_neg
    aggregate_score = total_pos / total if total else None

    return analyzed, aggregate_score


def extract_guidance(text):
    """
    Return paragraphs that contain forward-looking language.

    Works on whole paragraphs so it does not break on decimal points or
    abbreviations.
    """
    if not text:
        return []

    paragraphs = split_into_paragraphs(text)
    guidance_paragraphs = []

    for para in paragraphs:
        if para["is_heading"]:
            continue
        if len(para["text"]) > 2000:
            continue
        para_lower = para["text"].lower()
        if any(keyword in para_lower for keyword in guidance_keywords):
            guidance_paragraphs.append(para["text"].strip())

    return guidance_paragraphs


def _build_topic_summary(paragraphs):
    """
    Summarize how many paragraphs mention each topic, plus average sentiment.
    """
    summary = {}
    for topic in TOPIC_KEYWORDS:
        relevant = [p for p in paragraphs if topic in p["topics"]]
        if not relevant:
            continue
        scored = [
            p["sentiment"]["score"] for p in relevant
            if p["sentiment"]["score"] is not None
        ]
        summary[topic] = {
            "count": len(relevant),
            "avg_sentiment": sum(scored) / len(scored) if scored else None,
        }
    return summary


def _get_relevant_paragraphs(paragraphs):
    """
    Return paragraphs sorted by relevance score (highest first).

    Boilerplate paragraphs are included but sorted to the bottom.
    Headings are excluded since they carry no analytical content.
    """
    content_paras = [p for p in paragraphs if not p["is_heading"]]
    return sorted(content_paras, key=lambda p: p["relevance"], reverse=True)


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze_filing(ticker, filing_type="10-K"):
    """
    Pull the most recent 10-K or 10-Q from EDGAR, extract key sections,
    and return a summary dict with tone analysis and guidance sentences.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        filing_type: "10-K" for annual or "10-Q" for quarterly
    """
    if filing_type not in ("10-K", "10-Q"):
        raise ValueError("filing_type must be '10-K' or '10-Q'")

    cik, company_name = lookup_cik(ticker)
    filing = find_latest_filing(cik, filing_type)
    clean_text = fetch_clean_text(filing["url"])

    if filing_type == "10-K":
        mda_item, mda_next = 7, 8
        risk_item, risk_next = "1A", "1B"
        risk_min_length = MIN_SECTION_LENGTH
    else:
        mda_item, mda_next = 2, 3
        risk_item, risk_next = "1A", 2
        risk_min_length = 50

    mda_text = extract_section(
        clean_text,
        r"Management.s Discussion and Analysis of Financial Condition",
        mda_item,
        mda_next,
    )
    risk_text = extract_section(
        clean_text, r"Risk Factors", risk_item, risk_next, min_length=risk_min_length
    )

    # Paragraph-level analysis of MD&A
    if mda_text:
        mda_paragraphs, mda_aggregate_score = analyze_paragraphs(mda_text)
    else:
        mda_paragraphs, mda_aggregate_score = [], None

    # Build the legacy tone dicts from the whole section for backward compat.
    mda_tone = tone_score(mda_text) if mda_text else EMPTY_TONE
    risk_tone = tone_score(risk_text) if risk_text else EMPTY_TONE

    # Use paragraph-aggregated score as headline, fall back to whole-section.
    effective_tone = (
        mda_aggregate_score if mda_aggregate_score is not None
        else mda_tone["score"]
    )

    # Sorted/filtered paragraph views (requirement #7)
    relevant_paragraphs = _get_relevant_paragraphs(mda_paragraphs)

    return {
        # --- Original keys (unchanged for downstream compatibility) ---
        "ticker": ticker.upper(),
        "company_name": company_name,
        "cik": cik,
        "filing_type": filing_type,
        "filing_date": filing["filing_date"],
        "url": filing["url"],
        "mda_text": mda_text,
        "risk_factors_text": risk_text,
        "tone_score": effective_tone,
        "positive_counts": mda_tone["positive_hits"],
        "negative_counts": mda_tone["negative_hits"],
        "risk_tone_score": risk_tone["score"],
        "guidance_sentences": extract_guidance(mda_text) if mda_text else [],
        # --- New keys for richer analysis ---
        "paragraphs": mda_paragraphs,
        "relevant_paragraphs": relevant_paragraphs,
        "forward_looking_paragraphs": [
            p for p in mda_paragraphs
            if p["temporality"] in ("forward_looking", "mixed")
        ],
        "historical_paragraphs": [
            p for p in mda_paragraphs
            if p["temporality"] in ("historical", "mixed")
        ],
        "topic_summary": _build_topic_summary(mda_paragraphs),
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(result):
    """Print a human-readable summary of an analyze_filing result."""
    print(f"{result['company_name']} ({result['ticker']})  CIK {result['cik']}")
    print(f"{result['filing_type']} filed {result['filing_date']}")
    print(result["url"])

    sections = (
        ("MD&A", "mda_text", "tone_score"),
        ("Risk Factors", "risk_factors_text", "risk_tone_score"),
    )
    for label, text_key, score_key in sections:
        text = result[text_key]
        score = result[score_key]
        print(f"\n=== {label} ===")
        if text is None:
            print(f"Could not locate the {label} section")
            continue
        print(f"Length: {len(text):,} characters")
        if score is None:
            print("Tone score: undefined (no sentiment words found)")
        else:
            print(f"Tone score: {score:.3f}  (1.0 = all positive)")

    print(f"\nMD&A positive hits: {result['positive_counts']}")
    print(f"MD&A negative hits: {result['negative_counts']}")

    # Paragraph-level analysis summary
    paras = result.get("paragraphs", [])
    if paras:
        content_paras = [p for p in paras if not p["is_heading"]]
        boilerplate = [p for p in content_paras if p.get("is_boilerplate")]
        fwd = [p for p in paras if p["temporality"] == "forward_looking"]
        hist = [p for p in paras if p["temporality"] == "historical"]
        mixed = [p for p in paras if p["temporality"] == "mixed"]
        print(f"\n=== Paragraph Analysis ({len(paras)} total, "
              f"{len(content_paras)} content, {len(boilerplate)} boilerplate) ===")
        print(f"  Forward-looking: {len(fwd)}  |  Historical: {len(hist)}  |  Mixed: {len(mixed)}")

        pos_paras = [p for p in paras if p["sentiment"]["label"] == "positive"]
        neg_paras = [p for p in paras if p["sentiment"]["label"] == "negative"]
        neu_paras = [p for p in paras if p["sentiment"]["label"] == "neutral"]
        print(f"  Positive: {len(pos_paras)}  |  Negative: {len(neg_paras)}  |  Neutral: {len(neu_paras)}")

    # Topic summary
    topic_summary = result.get("topic_summary", {})
    if topic_summary:
        print(f"\n=== Topics ===")
        for topic, info in sorted(
            topic_summary.items(), key=lambda x: x[1]["count"], reverse=True
        ):
            avg = info["avg_sentiment"]
            avg_str = f"{avg:.3f}" if avg is not None else "n/a"
            print(f"  {topic}: {info['count']} paragraphs, avg sentiment {avg_str}")

    # Top relevant paragraphs
    relevant = result.get("relevant_paragraphs", [])
    if relevant:
        top = [p for p in relevant if p["relevance"] > 0][:5]
        if top:
            print(f"\n=== Top Relevant Paragraphs (by relevance score) ===")
            for i, p in enumerate(top, 1):
                tags = ", ".join(p["topics"]) if p["topics"] else "none"
                bp = " [boilerplate]" if p.get("is_boilerplate") else ""
                preview = p["text"][:200] + "..." if len(p["text"]) > 200 else p["text"]
                print(f"\n  {i}. [score={p['relevance']}, {p['temporality']}, "
                      f"topics: {tags}{bp}]")
                print(f"     {preview}")

    guidance = result["guidance_sentences"]
    print(f"\n=== Guidance ({len(guidance)} paragraphs) ===")
    for para in guidance[:10]:
        preview = para[:200] + "..." if len(para) > 200 else para
        print(f"  - {preview}\n")


def empty_filing(symbol):
    """Fallback filing dict when EDGAR fetch fails."""
    return {
        "ticker": symbol.upper(),
        "company_name": symbol.upper(),
        "cik": "",
        "filing_type": "N/A",
        "filing_date": "N/A",
        "url": "N/A",
        "mda_text": None,
        "risk_factors_text": None,
        "tone_score": None,
        "positive_counts": {},
        "negative_counts": {},
        "risk_tone_score": None,
        "guidance_sentences": [],
        "paragraphs": [],
        "relevant_paragraphs": [],
        "forward_looking_paragraphs": [],
        "historical_paragraphs": [],
        "topic_summary": {},
    }


if __name__ == "__main__":
    ticker = input("Ticker: ").strip().upper()
    choice = input("Annual (10-K) or quarterly (10-Q)? [K/Q]: ").strip().upper()
    print_report(analyze_filing(ticker, "10-K" if choice == "K" else "10-Q"))
