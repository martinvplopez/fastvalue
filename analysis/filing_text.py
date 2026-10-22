"""Plain-text reading of a 10-K annual report: sections, competitors, sole-source suppliers, large customers.

Everything here is pure (no I/O). ``data/sec_filings.py`` downloads the document and calls these functions.
The 10-K text carries no points in the Competitive Position agent: it is quoted as evidence, and each extractor
errs on the side of finding nothing (a missing note is honest, a wrong one is not). Precision was checked by
hand on 24 recent filings (decision_log.md D-34).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

SECTION_CAP = 12_000  # characters kept per extracted section
EXCERPT_LIMIT = 240

# ------------------------------------------------------------------ html to text

_HIDDEN = re.compile(r"<(script|style|ix:header)\b.*?</\1\s*>", re.I | re.S)
_BLOCK = frozenset({"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "td"})


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001 - HTMLParser signature
        if tag in _BLOCK:
            self.parts.append("\n")
        if tag == "td":
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(html: str) -> str:
    """Readable text of a 10-K page: hidden XBRL header, scripts and styles dropped, one line per block."""
    parser = _TextParser()
    parser.feed(_HIDDEN.sub(" ", html))
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r" *\n[ \n]*", "\n", text).strip()


# ------------------------------------------------------------------ items and subsections

_ITEM = re.compile(r"^item\s*(\d{1,2}[a-c]?)\s*[.:\-\u2013\u2014]?\s*(.{0,80})$", re.I | re.M)


def split_items(text: str) -> dict[str, tuple[int, int]]:
    """Span (start, end) of each "Item N" in the body of the report, keyed "1", "1A", "7" and so on.

    The table of contents lists every item in a few lines; the body copy is the longest span of an item,
    so the longest span wins.
    """
    marks = [(m.group(1).upper(), m.start()) for m in _ITEM.finditer(text)]
    spans: dict[str, tuple[int, int]] = {}
    for i, (key, start) in enumerate(marks):
        end = next((s for k, s in marks[i + 1:] if k != key), len(text))
        old = spans.get(key, (0, 0))
        if end - start > old[1] - old[0]:
            spans[key] = (start, end)
    return spans


_HEADINGS: dict[str, re.Pattern[str]] = {
    "competition": re.compile(
        r"^(the\s+)?(competition|competitive\s+(environment|conditions|landscape|position|factors)|competitors"
        r"|market\s+competition)\b", re.I),
    "suppliers": re.compile(
        r"^(suppliers?|supply\s+chain|raw\s+materials?|(sources?\s+(and\s+availability\s+)?of\s+(raw\s+)?materials)"
        r"|sources\s+and\s+availability|manufacturing(\s+and\s+(supply|sourcing))?|materials|purchasing|sourcing"
        r"|product\s+supply|merchandise\s+sourcing)\b", re.I),
    "customers": re.compile(
        r"^(customers?|major\s+customers|significant\s+customers|customer\s+concentration|distribution"
        r"|sales\s+and\s+distribution|sales\s+and\s+marketing)\b", re.I),
}
_RUN_IN = re.compile(r"^(competition|competitors|competitive\s+(environment|conditions|landscape))\s*[.:]\s+\S", re.I)
_NOISE_LINE = re.compile(r"^(table\s+of\s+contents|\d{1,3}|page\s+\d+|part\s+[ivx]+)$", re.I)


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or not s[0].isupper():
        return False
    if 2 <= len(s) <= 90 and len(s.split()) <= 10 and not s.endswith((".", ",", ";", ":")):
        return True
    return len(s) <= 90 and s.endswith(".") and len(s.split()) <= 6  # "Competition." as a run-in label on its own line


def subsections(item_text: str) -> dict[str, str]:
    """Competition, suppliers and customers subsections of Item 1 (first match of each, capped at 12 KB).

    A heading is a short capitalized line without a final period. A section runs to the next heading once it
    holds some text, so a page number or a "Table of Contents" line never ends it.
    """
    lines = [ln.strip() for ln in item_text.split("\n") if ln.strip()]
    out: dict[str, str] = {}
    for topic, pattern in _HEADINGS.items():
        for i, line in enumerate(lines):
            run_in = topic == "competition" and bool(_RUN_IN.match(line))
            if not (run_in or (_is_heading(line) and pattern.match(line))):
                continue
            body: list[str] = [line] if run_in else []
            size = len(line) if run_in else 0
            for nxt in lines[i + 1:]:
                if _NOISE_LINE.match(nxt):
                    continue
                if size >= 80 and _is_heading(nxt):
                    break
                body.append(nxt)
                size += len(nxt)
                if size >= SECTION_CAP:
                    break
            text = " ".join(body).strip()
            if len(text) >= 80 and len(text) > len(out.get(topic, "")):
                out[topic] = text[:SECTION_CAP]  # the fullest match: an index of headings repeats the word first
    return out


# ------------------------------------------------------------------ sentences

_SENT = re.compile(
    r"(?<=[.!?;])(?<!U\.S\.)(?<!Inc\.)(?<!Corp\.)(?<!Co\.)(?<!Ltd\.)(?<!No\.)(?<!St\.)(?<!vs\.)\s+(?=[A-Z(\"“•])")


def sentences(text: str) -> list[str]:
    """Sentences of 30 to 1,500 characters (paragraphs are lines)."""
    out: list[str] = []
    for para in text.split("\n"):
        out += _SENT.split(para)
    return [s.strip() for s in out if 30 <= len(s.strip()) <= 1500]


def excerpt(sentence: str, limit: int = EXCERPT_LIMIT) -> str:
    """One quotable line: whitespace collapsed, em and en dashes written as " - ", cut at a word with "..."."""
    text = " ".join(str(sentence).split())
    text = re.sub(r"\s*[\u2013\u2014]\s*", " - ", text)
    if len(text) <= limit:
        return text
    cut = text[: limit - 3].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{cut}..."


# ------------------------------------------------------------------ customers

_NUM = r"\d{1,2}(?:\.\d{1,2})?\s?(?:%|percent)"
_PCTS = rf"(?P<pct>\d{{1,2}}(?:\.\d{{1,2}})?)\s?(?:%|percent)(?:\s*,\s*{_NUM})*(?:\s*,?\s*and\s+{_NUM})?"
_VERB = r"(?:accounted\s+for|represented|comprised|constituted|were|was|made\s+up|amounted\s+to)"
_QUAL = r"(?:approximately\s+|about\s+|roughly\s+|nearly\s+|over\s+|more\s+than\s+|almost\s+)?"
_OF_REV = (r"(?:,?\s*respectively,?)?\s+of\s+(?:our\s+|the\s+company['’]s\s+|its\s+|total\s+|consolidated\s+"
           r"|aggregate\s+|worldwide\s+)*(?:net\s+)?(?:sales|revenues?|net\s+revenues?|operating\s+revenues?)")
_CUST_A = re.compile(_VERB + r"\s+" + _QUAL + _PCTS + _OF_REV, re.I)
_CUST_B = re.compile(r"(?:from\s+which\s+we\s+)?(?:derived|generated|received)\s+" + _QUAL + _PCTS + _OF_REV, re.I)
_WHO = re.compile(r"\b(customers?|clients?|distributors?|wholesalers?|retailers?|government|licensees?|OEMs?|accounts?)\b", re.I)
_NEG = re.compile(r"(^|\W)(no|none\s+of|not)\s+(other\s+|single\s+|one\s+|individual\s+)?(customers?|clients?|sales\s+to)", re.I)
_GEO = re.compile(
    r"located|outside\s+(of\s+)?the|international\s+(customers|operations|sales)|countr|region|segments?\b|geograph"
    r"|operations\s+generated|sales\s+to\s+(international|foreign)|e-?commerce|channel", re.I)
_GOVERNMENT = re.compile(r"\bgovernment\b", re.I)
_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "twenty": 20}
_TOP_N = re.compile(
    r"\b(?:(?:top|largest)\s+(?P<a>two|three|four|five|six|seven|eight|nine|ten|twenty|\d{1,2})\b"
    r"|(?P<b>two|three|four|five|six|seven|eight|nine|ten|twenty|\d{1,2})\s+largest\b)", re.I)
_NO_MAJOR = re.compile(
    r"\b(?:no|not\s+any|none\s+of\s+our)\s+(?:single\s+|individual\s+|one\s+|other\s+)?(?:customers?|clients?|distributors?)\b"
    r"[^.]{0,160}?(?:\b10\s?%|\bten\s+percent)", re.I)


@dataclass(frozen=True)
class Concentration:
    """A disclosed customer concentration: ``kind`` is "single", "top_n" or "government"; ``n`` the customer count."""

    kind: str
    n: int
    pct: float
    sentence: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "n": self.n, "pct": self.pct, "sentence": self.sentence}


def _concentration_kind(subject: str) -> tuple[str, int]:
    if _GOVERNMENT.search(subject):
        return "government", 1
    m = None
    for m in _TOP_N.finditer(subject):  # the closest mention before the number
        pass
    if m:
        word = (m.group("a") or m.group("b")).lower()
        n = int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 0)
        if n >= 2:
            return "top_n", n
    return "single", 1


def customer_concentration(text: str) -> list[Concentration]:
    """Sentences that give a share of sales to one customer, a group of customers or a government (largest first).

    Shares of sales by geography, channel or segment ("customers located outside the United States") and
    negations ("no customer accounted for...") are rejected.
    """
    found: dict[str, Concentration] = {}
    for sent in sentences(text):
        m = _CUST_A.search(sent) or _CUST_B.search(sent)
        if not m:
            continue
        subject = sent[max(0, m.start() - 220): m.start()]
        proper_name = re.search(r"\b[A-Z][a-z]+", subject[-60:]) and _CUST_B.search(sent)
        if not (_WHO.search(subject) or proper_name):
            continue
        if _NEG.search(sent) or _GEO.search(subject) or _GEO.search(sent[m.end(): m.end() + 80]):
            continue
        kind, n = _concentration_kind(subject)
        found.setdefault(sent[:120], Concentration(kind, n, float(m.group("pct")), sent))
    # The same share is often repeated in several sections; keep one sentence per share, the most specific kind.
    rank = {"government": 0, "top_n": 1, "single": 2}
    unique: dict[float, Concentration] = {}
    for c in found.values():
        if c.pct not in unique or rank[c.kind] < rank[unique[c.pct].kind]:
            unique[c.pct] = c
    return sorted(unique.values(), key=lambda c: -c.pct)


def no_major_customer(text: str) -> str | None:
    """The sentence saying no customer is 10% or more of sales, when the report has one."""
    for sent in sentences(text):
        if _NO_MAJOR.search(sent):
            return sent
    return None


# ------------------------------------------------------------------ suppliers

_SOURCING = re.compile(
    r"\b(?:single|sole)[\s-]+(?:source[sd]?|suppl(?:y|iers?)|vendors?|manufacturers?|partners?)\b"
    r"|\bone\s+or\s+(?:a\s+)?(?:very\s+)?(?:limited|few)\s+(?:number\s+of\s+)?(?:suppliers?|vendors?|sources?)"
    r"|\b(?:limited|small|few)\s+number\s+of\s+(?:suppliers?|vendors?|sources?|manufacturers?)\b", re.I)
_SOURCING_NOT = re.compile(
    r"\bnot\b[^.]{0,60}\bany\s+one\s+(?:supplier|vendor)|\bno\s+(?:single|one)\s+(?:supplier|vendor)"
    r"|single[\s-]+source\s+(?:drugs?|biologics?|products?)|warehouse\s+purchasing", re.I)


def sourcing_sentences(text: str) -> list[str]:
    """Sentences about single-source, sole-source or a limited number of suppliers.

    "Single-source drugs" (a drug-pricing term, not a supply risk) and "we do not obtain a significant
    portion from any one supplier" are left out.
    """
    return [s for s in sentences(text) if _SOURCING.search(s) and not _SOURCING_NOT.search(s)]


# ------------------------------------------------------------------ competitors

_SUFFIX = (r"(?:Inc\.?|Incorporated|Corp\.?|Corporation|Co\.|Company|Ltd\.?|Limited|plc|PLC|N\.V\.|NV|AG|S\.A\.|SA|SE"
           r"|S\.p\.A\.|Holdings?|Group|LLC|L\.P\.|AB|ASA|Oyj|KGaA)")
_NAME_WORD = r"(?:(?:[A-Z]\.){1,4}|[A-Z][\w&'\u2019\-]*(?:\.[a-z]+)*)"  # "Advanced", "V.F.", "Amazon.com"
_COMPANY_NAME = re.compile(
    r"\b((?:" + _NAME_WORD + r"\s+(?:(?:of|de|&)\s+)?){0,4}" + _NAME_WORD + r"),?\s+" + _SUFFIX + r"(?=[\s,.;)])")
_STOP_WORDS = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "nv", "ag", "sa", "se", "spa",
    "holdings", "holding", "group", "llc", "lp", "ab", "asa", "oyj", "kgaa", "the", "and", "&", "de", "class", "a", "b",
    "com", "new"})
MAX_COMPETITORS = 12


def normalize_company_name(name: str) -> str:
    """Lower case name without punctuation or legal suffixes ("Advanced Micro Devices, Inc." -> "advanced micro devices")."""
    tokens = re.sub(r"[^a-z0-9& ]", " ", str(name).lower().replace(".", "")).split()
    return " ".join(t for t in tokens if t not in _STOP_WORDS)


def named_competitors(section: str, titles: dict[str, str], own_names: tuple[str, ...] = ()) -> list[dict[str, str | None]]:
    """Company names written in the competition section, with a ticker when the name matches a listed company.

    ``titles`` maps normalized company titles (``normalize_company_name``) to tickers. "The Company" and the
    filer itself (``own_names``) are skipped. Names are in the order of the text, at most 12.
    """
    own = {normalize_company_name(n) for n in own_names if n}
    out: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for m in _COMPANY_NAME.finditer(section or ""):
        name = m.group(0).strip(" ,.;")
        key = normalize_company_name(name)
        if not key or key in own or key in seen or len(name.split()) > 5:  # longer matches run into a heading
            continue
        seen.add(key)
        out.append({"name": name, "ticker": titles.get(key)})
        if len(out) >= MAX_COMPETITORS:
            break
    return out
