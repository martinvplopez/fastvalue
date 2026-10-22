"""Competitive position: figures against the closest peers, peer choice and the eight scored rules.

A company with bargaining power over its suppliers pays them later than its peers do; over its customers, it
gets paid sooner, keeps a higher and steadier gross margin, and earns more on its capital. These rules read
that from annual statement figures and compare each with the median of up to eight peers (Porter's bargaining
power, read from the balance sheet). The 10-K notes (named competitors, sole-source suppliers, large
customers) and EV/FCF against peers are listed with 0 points: the notes cannot be replayed on past filings and
valuation is scored by the Fundamentals agent (decision_log.md D-34).

Everything here is pure (no I/O). The live agent (``agents/competitive_agent.py``) and the point-in-time study
(``core/competitive_study.py``) call the same ``choose_peers``, ``company_metrics`` and ``score_competitive``,
so what is tested is what runs.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import pandas as pd

from analysis.filing_text import normalize_company_name
from analysis.fundamental_metrics import _roic_for_year
from data.financial_statements import FinancialStatements

SCORING_VERSION = 1

# Peer group: market value within a tenth to ten times the company's, widened when fewer than four qualify.
BAND = (0.1, 10.0)
WIDE_BAND = (0.04, 25.0)
MIN_PEERS = 3  # a peer median needs this many peers with the figure
GOOD_PEERS = 4  # fewer than this widens the band or the industry code
MAX_PEERS = 8
STALE_MONTHS = 15  # a peer whose latest fiscal year ended this long before the company's is left out
MAX_DAYS = 730  # receivable or payable days above this are a tagging error
MARGIN_YEARS = 5  # fiscal years behind the margin swing (four are needed)
MAX_CONFIDENCE = 85.0

FIELDS = ("gross_margin", "gross_margin_std", "operating_margin", "dso", "dpo", "dio", "roic", "revenue_cagr_3y")

SCORED_RULES = ("S1", "C1", "C2", "C3", "P1", "P2", "P3", "P4")
RULE_LABELS = {
    "S1": "days to pay suppliers",
    "C1": "days to get paid",
    "C2": "gross margin",
    "C3": "gross margin stability",
    "P1": "ROIC",
    "P2": "operating margin",
    "P3": "3-year sales growth",
    "P4": "share of peer-group sales",
}
RULE_COMPONENT = {"S1": "suppliers", "C1": "customers", "C2": "customers", "C3": "customers",
                  "P1": "standing", "P2": "standing", "P3": "standing", "P4": "standing"}
COMPONENTS = ("suppliers", "customers", "standing")


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ------------------------------------------------------------------ figures per company


@dataclass
class CompanyMetrics:
    """Ratios of one company's latest fiscal year (fractions; days; points for the margin swing)."""

    ticker: str = ""
    fy_end: str | None = None
    gross_margin: float | None = None
    gross_margin_std: float | None = None  # population standard deviation over up to five years, in points
    operating_margin: float | None = None
    dso: float | None = None  # days sales outstanding: receivables / revenue x 365
    dpo: float | None = None  # days payable outstanding: payables / cost of revenue x 365
    dio: float | None = None  # days inventory outstanding: inventory / cost of revenue x 365
    roic: float | None = None
    revenue_cagr_3y: float | None = None
    revenue_usd: float | None = None
    revenue_3y_ago_usd: float | None = None
    ev_to_fcf: float | None = None  # Yahoo figures, information only
    has_cost_of_revenue: bool = False
    issues: list[str] = field(default_factory=list)  # tagging errors found and left out

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("ticker", "fy_end", *FIELDS, "revenue_usd", "revenue_3y_ago_usd", "ev_to_fcf")}


def _filled(year: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    """(revenue, cost of revenue, gross profit) with the missing one of the last two derived from the other."""
    revenue, cogs, gross = _num(year.get("revenue")), _num(year.get("cogs")), _num(year.get("gross_profit"))
    if revenue is not None:
        if cogs is None and gross is not None:
            cogs = revenue - gross
        elif gross is None and cogs is not None:
            gross = revenue - cogs
    return revenue, cogs, gross


def _gross_margin(year: dict[str, Any]) -> float | None:
    revenue, _, gross = _filled(year)
    if revenue is None or revenue <= 0 or gross is None:
        return None
    margin = gross / revenue
    return margin if -1.0 <= margin <= 1.0 else None


def company_metrics(
    years: list[dict[str, Any]], fx_to_usd: float | None = None, ev_to_fcf: float | None = None, ticker: str = ""
) -> CompanyMetrics:
    """Ratios from a card's fiscal years (newest first, ``None`` where the filing has no line).

    Revenue of zero or less gives no ratios. A gross margin outside -100% to 100%, or receivable or payable days
    above 730, is a tagging error: the value is left out and the issue is listed.
    """
    m = CompanyMetrics(ticker=ticker, ev_to_fcf=ev_to_fcf)
    if not years:
        return m
    latest = years[0]
    m.fy_end = latest.get("fy_end")
    revenue, cogs, gross = _filled(latest)
    m.has_cost_of_revenue = cogs is not None
    m.roic = _num(latest.get("roic"))
    if revenue is None or revenue <= 0:
        m.issues.append("no revenue reported")
        return m

    if gross is not None:
        margin = gross / revenue
        if -1.0 <= margin <= 1.0:
            m.gross_margin = margin
        else:
            m.issues.append("gross margin outside -100% to 100% (tagging error)")
    operating = _num(latest.get("operating_income"))
    if operating is not None:
        m.operating_margin = operating / revenue
    receivables = _num(latest.get("receivables"))
    if receivables is not None and receivables >= 0:
        days = receivables / revenue * 365.0
        if days <= MAX_DAYS:
            m.dso = days
        else:
            m.issues.append("receivable days above 730 (tagging error)")
    if cogs is not None and cogs > 0:
        payables, inventory = _num(latest.get("payables")), _num(latest.get("inventory"))
        if payables is not None and payables >= 0:
            days = payables / cogs * 365.0
            if days <= MAX_DAYS:
                m.dpo = days
            else:
                m.issues.append("payable days above 730 (tagging error)")
        if inventory is not None and inventory >= 0:
            m.dio = inventory / cogs * 365.0

    margins = [g for g in (_gross_margin(y) for y in years[:MARGIN_YEARS]) if g is not None]
    if len(margins) >= 4:
        m.gross_margin_std = statistics.pstdev(margins) * 100.0

    if len(years) > 3:
        old_revenue = _num(years[3].get("revenue"))
        gap = _years_between(years[3].get("fy_end"), m.fy_end)
        if old_revenue is not None and old_revenue > 0 and gap is not None and 2.7 <= gap <= 3.3:
            m.revenue_cagr_3y = (revenue / old_revenue) ** (1.0 / 3.0) - 1.0
            if fx_to_usd is not None:
                m.revenue_3y_ago_usd = old_revenue * fx_to_usd
    if fx_to_usd is not None:
        m.revenue_usd = revenue * fx_to_usd
    return m


def _years_between(start: Any, end: Any) -> float | None:
    try:
        return (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    except (TypeError, ValueError):
        return None


def is_stale(peer: CompanyMetrics, subject: CompanyMetrics) -> bool:
    """True when the peer's latest fiscal year ended more than 15 months before the company's (or is unknown)."""
    gap = _years_between(peer.fy_end, subject.fy_end)
    return gap is None or gap * 12.0 > STALE_MONTHS


def usable_peers(subject: CompanyMetrics, peers: list[CompanyMetrics]) -> list[CompanyMetrics]:
    return [p for p in peers if p.fy_end and not is_stale(p, subject)]


def peer_medians(peers: list[CompanyMetrics]) -> dict[str, tuple[float | None, int]]:
    """Median and count per field; a median needs three or more peers that report the figure."""
    out: dict[str, tuple[float | None, int]] = {}
    for name in FIELDS + ("ev_to_fcf",):
        values = [v for v in (getattr(p, name) for p in peers) if v is not None]
        out[name] = (float(statistics.median(values)) if len(values) >= MIN_PEERS else None, len(values))
    return out


def peer_group_share(subject: CompanyMetrics, peers: list[CompanyMetrics]) -> tuple[float, float] | None:
    """(share of peer-group sales now, three years ago): the company's sales over its own plus its peers'.

    Only peers with revenue now and three years ago count, all at today's exchange rate so that currency swings
    do not move the share. None with fewer than three such peers.
    """
    s_now, s_then = subject.revenue_usd, subject.revenue_3y_ago_usd
    if not s_now or not s_then or s_now <= 0 or s_then <= 0:
        return None
    pairs = [(p.revenue_usd, p.revenue_3y_ago_usd) for p in peers
             if p.revenue_usd and p.revenue_3y_ago_usd and p.revenue_usd > 0 and p.revenue_3y_ago_usd > 0]
    if len(pairs) < MIN_PEERS:
        return None
    return s_now / (s_now + sum(a for a, _ in pairs)), s_then / (s_then + sum(b for _, b in pairs))


def share_change(subject: CompanyMetrics, peers: list[CompanyMetrics]) -> float | None:
    """Relative change of the company's share of peer-group sales over three years (0.10 = share up 10%)."""
    share = peer_group_share(subject, peers)
    return None if share is None else share[0] / share[1] - 1.0


def card_years(
    st: FinancialStatements, margins: dict[pd.Timestamp, dict[str, float]] | None = None,
    max_years: int = 10, roic_years: int | None = None,
) -> list[dict[str, Any]]:
    """A card's fiscal years (newest first) from annual statements; cost of revenue and gross profit come from
    ``margins`` (SEC filings, matched by fiscal year end within 7 days) or from the statements (Yahoo)."""
    if not st.available:
        return []
    out: list[dict[str, Any]] = []
    for i, year in enumerate(st.years()[:max_years]):
        extra = next((v for k, v in (margins or {}).items() if abs((pd.Timestamp(k) - year).days) <= 7), {})
        cogs = extra.get("cogs")
        gross = extra.get("gross_profit")
        if cogs is None:
            cogs = st.value("income", ("Cost Of Revenue", "Reconciled Cost Of Revenue"), year)
        if gross is None:
            gross = st.value("income", "Gross Profit", year)
        roic = None
        if roic_years is None or i < roic_years:
            roic = _roic_for_year(st, year).roic
        out.append({
            "fy_end": year.strftime("%Y-%m-%d"),
            "revenue": st.value("income", "Total Revenue", year),
            "cogs": cogs, "gross_profit": gross,
            "operating_income": st.value("income", ("Operating Income", "EBIT"), year),
            "receivables": st.value("balance", ("Accounts Receivable", "Receivables"), year),
            "payables": st.value("balance", ("Accounts Payable", "Payables"), year),
            "inventory": st.value("balance", "Inventory", year),
            "roic": roic,
        })
    return out


# ------------------------------------------------------------------ peer choice


class Candidate(NamedTuple):
    """A possible peer. ``market_cap`` is in US dollars; ``sic`` and ``cik`` are None for Yahoo-only names."""

    ticker: str
    name: str | None
    market_cap: float | None
    sic: int | None = None
    cik: int | None = None


@dataclass
class PeerChoice:
    """The peers picked (nearest in market value first) and where they came from."""

    tickers: list[str] = field(default_factory=list)
    source: str = ""  # "yahoo_industry", "sic4", "sic3" or "sic2"
    label: str = ""
    wide_band: bool = False


def choose_peers(
    subject_cap: float | None, candidates: list[Candidate], max_peers: int = MAX_PEERS, *,
    subject_ticker: str | None = None, subject_name: str | None = None, subject_cik: int | None = None,
    allow_wide: bool = True,
) -> PeerChoice:
    """The up-to-eight candidates nearest in market value, within a tenth to ten times the company's.

    The company and its other share classes (same CIK or same name) are dropped, as are rows without a name or
    market value and a second class of the same peer. With fewer than four in the band the band widens to 0.04x
    to 25x (``wide_band``), unless ``allow_wide`` is False.
    """
    cap = _num(subject_cap)
    if cap is None or cap <= 0:
        return PeerChoice()
    own_name = normalize_company_name(subject_name or "")
    pool: dict[str, Candidate] = {}
    for c in candidates:
        c_cap, name = _num(c.market_cap), (c.name or "").strip()
        if not name or name.lower() == "nan" or c_cap is None or c_cap <= 0:
            continue
        key = normalize_company_name(name)
        if c.ticker == subject_ticker or (subject_cik is not None and c.cik == subject_cik) or (own_name and key == own_name):
            continue
        group = f"cik:{c.cik}" if c.cik is not None else f"name:{key or c.ticker}"
        if group not in pool or c_cap > (pool[group].market_cap or 0):
            pool[group] = c
    wide = False

    def within(band: tuple[float, float]) -> list[Candidate]:
        return [c for c in pool.values() if band[0] <= c.market_cap / cap <= band[1]]

    picked = within(BAND)
    if len(picked) < GOOD_PEERS and allow_wide:
        picked, wide = within(WIDE_BAND), True
        if len(picked) <= len(within(BAND)):
            picked, wide = within(BAND), False
    picked.sort(key=lambda c: (abs(math.log(c.market_cap / cap)), c.ticker))
    return PeerChoice(tickers=[c.ticker for c in picked[:max_peers]], wide_band=wide)


class SicIndex:
    """Candidates grouped by the first 4, 3 and 2 digits of their SEC industry code (built once per date)."""

    def __init__(self, directory: list[Candidate]) -> None:
        self.groups: dict[tuple[int, str], list[Candidate]] = {}
        for c in directory:
            if c.sic is None:
                continue
            code = f"{int(c.sic):04d}"
            for digits in (4, 3, 2):
                self.groups.setdefault((digits, code[:digits]), []).append(c)

    def same(self, digits: int, code: str) -> list[Candidate]:
        return self.groups.get((digits, code[:digits]), [])


def choose_peers_by_sic(
    subject_cap: float | None, subject_sic: int | None, directory: "list[Candidate] | SicIndex", max_peers: int = MAX_PEERS,
    allow_wide: bool = True, **identity: Any,
) -> PeerChoice:
    """Peers by SEC industry code, the closest code first.

    In order, the first with enough peers wins: the same 4-digit code, then 3 digits (four or more in the size
    band), then the same two in the wide band, then 2 digits (four or more in the band) and last 2 digits in the
    wide band with three or more. A close industry in a wide band beats a broad industry in a narrow one: for a
    company worth trillions, "machinery and computers" is a worse peer group than a few computer makers worth a
    tenth as much. ``directory`` may be a ``SicIndex`` when many companies are scored against the same list;
    ``allow_wide=False`` keeps to the strict band (a robustness check in the study).
    """
    if subject_sic is None:
        return PeerChoice()
    index = directory if isinstance(directory, SicIndex) else SicIndex(directory)
    code = f"{int(subject_sic):04d}"
    for digits, wide in ((4, False), (3, False), (4, True), (3, True), (2, False), (2, True)):
        if wide and not allow_wide:
            continue
        same = index.same(digits, code)
        choice = (choose_peers(subject_cap, same, max_peers, **identity) if wide
                  else choose_peers(subject_cap, same, max_peers, allow_wide=False, **identity))
        if digits == 2 and wide:  # last resort: three peers are enough, whether or not widening added any
            if len(choice.tickers) >= MIN_PEERS:
                return replace(choice, source="sic2")
        elif len(choice.tickers) >= GOOD_PEERS and choice.wide_band == wide:
            return replace(choice, source=f"sic{digits}")
    return PeerChoice()


# ------------------------------------------------------------------ scoring


def _first_at_least(value: float, steps: tuple[tuple[float, float], ...], otherwise: float) -> float:
    for threshold, points in steps:
        if value >= threshold:
            return points
    return otherwise


def _first_at_most(value: float, steps: tuple[tuple[float, float], ...], otherwise: float) -> float:
    for threshold, points in steps:
        if value <= threshold:
            return points
    return otherwise


# Points per rule; thresholds are checked top down and the first that holds gives the points.
def points_dpo(ratio: float) -> float:  # S1: company / peer median days payable
    ratio = round(ratio, 6)
    return _first_at_least(ratio, ((1.40, 4), (1.15, 2)), 0.0) if ratio > 0.85 else (-2.0 if ratio > 0.70 else -4.0)


def points_dso(ratio: float) -> float:  # C1: company / peer median days receivable
    ratio = round(ratio, 6)
    return _first_at_most(ratio, ((0.60, 4), (0.85, 2)), 0.0) if ratio < 1.15 else (-2.0 if ratio < 1.40 else -4.0)


def points_gross_margin(gap: float) -> float:  # C2: gap to the peer median, in points
    gap = round(gap, 6)
    return _first_at_least(gap, ((10, 6), (4, 3)), 0.0) if gap > -4 else (-3.0 if gap > -10 else -6.0)


def points_margin_swing(ratio: float) -> float:  # C3: swing / max(peer median swing, 0.5 points)
    ratio = round(ratio, 6)
    return _first_at_most(ratio, ((0.50, 3), (0.80, 1)), 0.0) if ratio < 1.25 else (-1.0 if ratio < 2.0 else -3.0)


def points_roic(gap: float) -> float:  # P1: gap to the peer median, in points
    gap = round(gap, 6)
    return _first_at_least(gap, ((10, 6), (4, 3)), 0.0) if gap > -4 else (-3.0 if gap > -10 else -6.0)


def points_operating_margin(gap: float) -> float:  # P2
    gap = round(gap, 6)
    return _first_at_least(gap, ((8, 4), (3, 2)), 0.0) if gap > -3 else (-2.0 if gap > -8 else -4.0)


def points_growth(gap: float) -> float:  # P3: yearly sales growth gap over three years, in points
    gap = round(gap, 6)
    return _first_at_least(gap, ((6, 3), (2, 1)), 0.0) if gap > -2 else (-1.0 if gap > -6 else -3.0)


def points_share(change: float) -> float:  # P4: relative change of the share of peer-group sales
    change = round(change, 6)
    return _first_at_least(change, ((0.15, 3), (0.05, 1)), 0.0) if change > -0.05 else (-1.0 if change > -0.15 else -3.0)


@dataclass
class CompetitiveScore:
    """The scored rules, the notes shown with 0 points, and what could not be scored."""

    rules: list[dict[str, Any]] = field(default_factory=list)  # id, component, factor, points, info
    components: dict[str, float | None] = field(default_factory=dict)
    total: float = 0.0
    rules_scored: int = 0
    peers: int = 0
    not_scored: list[str] = field(default_factory=list)
    confidence: float = 0.0
    quality: str = "UNAVAILABLE"  # LIVE, PARTIAL or UNAVAILABLE
    medians: dict[str, tuple[float | None, int]] = field(default_factory=dict)
    share_change: float | None = None
    diagnostics: dict[str, float | None] = field(default_factory=dict)  # signed so that higher reads as better


def _days(value: float) -> str:
    return f"{value:.0f}"


def _pct(value: float) -> str:
    return f"{round(value * 100, 1) + 0.0:.1f}%"  # + 0.0 turns -0.0 into 0.0


def score_competitive(
    subject: CompanyMetrics, peers: list[CompanyMetrics], notes: dict[str, Any] | None = None,
    weak_peer_group: bool = False,
) -> CompetitiveScore:
    """Score the company against its peers (``peers`` are already usable: not stale, not the company itself).

    ``weak_peer_group`` marks peers from a 2-digit industry code or a widened market value band (confidence -10).
    ``notes`` holds the 10-K findings (``competitors``, ``sourcing``, ``raw_materials``, ``concentration``,
    ``no_major_customer``), listed with 0 points.
    """
    medians = peer_medians(peers)
    change = share_change(subject, peers)
    out = CompetitiveScore(peers=len(peers), medians=medians, share_change=change)
    diag: dict[str, float | None] = dict.fromkeys(
        ("dpo_ratio", "dso_ratio", "gm_gap", "gm_std_ratio", "roic_gap", "opm_gap", "growth_gap", "share_change"))

    def median(name: str) -> float | None:
        return medians[name][0]

    def add(rule: str, factor: str, points: float) -> None:
        out.rules.append({"id": rule, "component": RULE_COMPONENT[rule], "factor": factor, "points": float(points), "info": False})

    def skip(rule: str, reason: str) -> None:
        out.not_scored.append(f"{RULE_LABELS[rule]} ({reason})")

    def why(name: str, subject_value: float | None, no_cost: bool = False, need: str = "") -> str:
        if subject_value is None:
            if no_cost and not subject.has_cost_of_revenue:
                return "no cost of revenue reported"
            return need or "not reported for the company"
        return f"only {medians[name][1]} of {len(peers)} peers report it"

    # S1 suppliers: days to pay suppliers
    mp = median("dpo")
    if subject.dpo is not None and mp is not None and mp > 0:
        ratio = subject.dpo / mp
        diag["dpo_ratio"] = ratio
        add("S1", f"Pays suppliers in {_days(subject.dpo)} days against a peer median of {_days(mp)} (DPO)", points_dpo(ratio))
    else:
        skip("S1", why("dpo", subject.dpo, no_cost=True))

    # C1 customers: days to get paid
    mp = median("dso")
    if subject.dso is not None and mp is not None and mp > 0:
        ratio = subject.dso / mp
        diag["dso_ratio"] = -ratio
        add("C1", f"Gets paid in {_days(subject.dso)} days against a peer median of {_days(mp)} (DSO)", points_dso(ratio))
    else:
        skip("C1", why("dso", subject.dso))

    # C2 customers: gross margin against the peer median
    mp = median("gross_margin")
    if subject.gross_margin is not None and mp is not None:
        gap = (subject.gross_margin - mp) * 100.0
        diag["gm_gap"] = gap
        add("C2", f"Gross margin {_pct(subject.gross_margin)} against a peer median of {_pct(mp)}", points_gross_margin(gap))
    else:
        skip("C2", why("gross_margin", subject.gross_margin, no_cost=True))

    # C3 customers: how much the gross margin moves from year to year
    mp = median("gross_margin_std")
    if subject.gross_margin_std is not None and mp is not None:
        ratio = subject.gross_margin_std / max(mp, 0.5)
        diag["gm_std_ratio"] = -ratio
        add("C3", f"Gross margin moved {subject.gross_margin_std:.1f} points a year against {mp:.1f} for peers", points_margin_swing(ratio))
    else:
        skip("C3", why("gross_margin_std", subject.gross_margin_std, no_cost=True, need="needs four years of gross margin"))

    # P1 standing: return on invested capital
    mp = median("roic")
    if subject.roic is not None and mp is not None:
        gap = (subject.roic - mp) * 100.0
        diag["roic_gap"] = gap
        add("P1", f"ROIC {_pct(subject.roic)} against a peer median of {_pct(mp)}", points_roic(gap))
    else:
        skip("P1", why("roic", subject.roic))

    # P2 standing: operating margin
    mp = median("operating_margin")
    if subject.operating_margin is not None and mp is not None:
        gap = (subject.operating_margin - mp) * 100.0
        diag["opm_gap"] = gap
        add("P2", f"Operating margin {_pct(subject.operating_margin)} against a peer median of {_pct(mp)}", points_operating_margin(gap))
    else:
        skip("P2", why("operating_margin", subject.operating_margin))

    # P3 standing: sales growth over three years
    mp = median("revenue_cagr_3y")
    if subject.revenue_cagr_3y is not None and mp is not None:
        gap = (subject.revenue_cagr_3y - mp) * 100.0
        diag["growth_gap"] = gap
        add("P3", f"Sales growth {_pct(subject.revenue_cagr_3y)} a year over 3 years against a peer median of {_pct(mp)}", points_growth(gap))
    else:
        skip("P3", why("revenue_cagr_3y", subject.revenue_cagr_3y, need="needs four fiscal years of sales"))

    # P4 standing: share of peer-group sales
    if change is not None:
        diag["share_change"] = change
        verb = "rose" if change >= 0 else "fell"
        add("P4", f"Share of peer-group sales {verb} {abs(change) * 100:.0f}% over 3 years", points_share(change))
    else:
        skip("P4", "needs sales three years ago for the company and 3 peers"
             if subject.revenue_usd is not None else "no exchange rate to compare sales")

    _add_info_rules(out, subject, medians, notes or {})

    scored = [r for r in out.rules if not r["info"]]
    out.rules_scored = len(scored)
    out.total = sum(r["points"] for r in scored)
    out.components = {c: (sum(r["points"] for r in scored if r["component"] == c) if any(r["component"] == c for r in scored) else None)
                      for c in COMPONENTS}
    out.diagnostics = diag
    if not scored or len(peers) < MIN_PEERS:
        return out
    base = min(MAX_CONFIDENCE, 20.0 + 60.0 * out.rules_scored / len(SCORED_RULES))
    out.confidence = max(0.0, base * (1.0 if len(peers) >= 6 else 0.9 if len(peers) >= 4 else 0.75) - (10.0 if weak_peer_group else 0.0))
    out.quality = "LIVE" if out.rules_scored == len(SCORED_RULES) and len(peers) >= GOOD_PEERS else "PARTIAL"
    return out


def _add_info_rules(out: CompetitiveScore, subject: CompanyMetrics, medians: dict[str, tuple[float | None, int]],
                    notes: dict[str, Any]) -> None:
    def info(rule: str, component: str, factor: str) -> None:
        out.rules.append({"id": rule, "component": component, "factor": factor, "points": 0.0, "info": True})

    mp = medians["ev_to_fcf"][0]
    if subject.ev_to_fcf is not None and mp is not None:
        info("I1", "standing", f"EV/FCF {subject.ev_to_fcf:.1f}x against a peer median of {mp:.1f}x (not scored: the Fundamentals agent scores valuation)")
    competitors = notes.get("competitors") or []
    if competitors:
        names = ", ".join(str(c.get("ticker") or c.get("name")) for c in competitors[:6])
        info("I2", "standing", f"Competitors named in the 10-K: {names}")
    sourcing = notes.get("sourcing") or []
    if sourcing:
        info("I3", "suppliers", f"Single or sole-source suppliers named in the 10-K ({len(sourcing)} passage{'s' if len(sourcing) != 1 else ''})")
    if notes.get("raw_materials"):
        info("I4", "suppliers", f"Suppliers and materials in the 10-K: {notes['raw_materials']}")
    concentration = notes.get("concentration") or []
    if concentration:
        top = concentration[0]
        what = {"government": "A government", "top_n": f"The top {top.get('n')} customers"}.get(top.get("kind"), "The largest customer")
        info("I5", "customers", f"{what} is {float(top['pct']):g}% of sales in the 10-K")
    elif notes.get("no_major_customer"):
        info("I5", "customers", "No customer above 10% of sales is disclosed in the 10-K")
