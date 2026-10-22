"""Competitive Position Agent: suppliers, customers and competitors, against the company's closest peers.

Eight rules compare the company with the median of up to eight peers (the same Yahoo industry, else the same SEC
industry code, worth a tenth to ten times as much): days to pay suppliers, days to get paid, gross margin and its
stability, ROIC, operating margin, three-year sales growth and the share of peer-group sales. The latest 10-K is
quoted as evidence (named competitors, single or sole-source suppliers, large customers) and EV/FCF against peers is
shown, both with 0 points.

The agent sits in the owner layer with weight 0, like Insider: the point-in-time study in
``core/competitive_study.py`` found no edge (decision_log.md D-34), so it has no vote. Weight-0 agents are dropped
before the owner score is formed, and none of the keys the framework checks (``excluded_sector``,
``is_financial``, ``ev_to_fcf``, ``fcf``) appear at the top of ``raw_data``, so no existing score changes.

Peers load in a shared pool under a 12 second budget (the agent timeout is 30). With a warm cache the answer
takes a fraction of a second; a cold one returns PARTIAL with the peers still loading, and the fetches that miss the
budget keep running and fill the cache for the next run.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import wait
from typing import Any

import pandas as pd

from agents.base import LIVE, OWNER, PARTIAL, AgentAssessment, BaseAgent
from agents.registry import register_agent
from analysis.competitive_metrics import (
    SCORING_VERSION,
    CompanyMetrics,
    Candidate,
    PeerChoice,
    choose_peers,
    choose_peers_by_sic,
    company_metrics,
    peer_group_share,
    score_competitive,
    usable_peers,
)
from analysis.filing_text import excerpt
from analysis.fundamental_metrics import is_excluded_sector, is_financial_company
from data import fundamentals as _fundamentals
from data import peers as _peers
from data import sec_filings as _filings
from data.fx import convert_latest
from data.sec_edgar import has_sec_access

logger = logging.getLogger(__name__)

BUDGET_SECONDS = 12.0
MIN_PEERS = 3
ENOUGH_PEERS = 4

# One plain-language definition per measure, shown in the "Against peers" table and the terms table.
TERMS = {
    "Gross margin": "Share of each sale left after the cost of making it: pricing power.",
    "Margin swing": "How much the gross margin moves from year to year (standard deviation over up to five years, in points). A steady margin points to stable pricing.",
    "Operating margin": "Share of sales left after running the business: operating income / revenue (pretax income plus interest when a filing has no operating income line).",
    "ROIC": "Return on invested capital: operating profit after tax divided by the capital invested in the business.",
    "Sales growth": "Yearly sales growth over three years.",
    "DSO": "Days sales outstanding: receivables / revenue x 365. How many days customers take to pay; fewer than peers means the company sets the terms.",
    "DPO": "Days payable outstanding: payables / cost of revenue x 365. How many days the company takes to pay suppliers; more than peers means it sets the terms (a very high figure can also mean a stretched company).",
    "DIO": "Days inventory outstanding: inventory / cost of revenue x 365. How long goods sit in stock. Shown, not scored.",
    "Share of peer-group sales": "The company's sales divided by its own plus its peers', now against three years ago, at today's exchange rates.",
    "EV / FCF": "Enterprise value / free cash flow from Yahoo, for context and not scored: the Fundamentals agent scores valuation.",
    "Peer median": "The middle value of the peers' figures, so one outlier cannot move it.",
    "SIC code": "The SEC's industry code for the company's main business.",
    "Evidence only": "Weight 0: the point-in-time study (decision_log.md D-34) found no edge, so this view has no vote; any retest needs a new test written down first. If it makes reviews too slow, untick it in the Weights tab.",
}
LEAD_TIP = ("Company against the median of its peers, latest fiscal year. Peers are in the same Yahoo industry "
            "(or the same SEC industry code) and worth between a tenth and ten times the company.")


# ---------------------------------------------------------------- formatting


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{round(value * 100, 1) + 0.0:.1f}%"  # + 0.0 turns -0.0 into 0.0


def _days(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} days"


def _mult(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}x"


def _points(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} points"


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if value >= divisor:
            return f"${value / divisor:.1f}{suffix}"
    return f"${value:,.0f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# ---------------------------------------------------------------- helpers


def _usd_rate(currency: str | None, cache: dict[str, float | None]) -> float | None:
    """Today's rate from the statement currency to US dollars (None when no rate is available)."""
    code = (currency or "USD").upper()
    if code == "USD":
        return 1.0
    if code not in cache:
        try:
            cache[code] = float(convert_latest(1.0, code, "USD"))
        except Exception as exc:  # noqa: BLE001 - no rate: the share of peer-group sales is skipped and said so
            logger.info("No %s rate: %s", code, exc)
            cache[code] = None
    return cache[code]


def _metrics_of(card: dict[str, Any], rates: dict[str, float | None]) -> CompanyMetrics:
    return company_metrics(card["years"], _usd_rate(card.get("currency"), rates), card.get("ev_to_fcf"), card["ticker"])


def _links(info: dict[str, Any], cik: int | None, filing: dict[str, Any] | None) -> list[dict[str, str]]:
    links = []
    if filing and filing.get("url"):
        links.append({"label": f"10-K filed {filing.get('filed')}", "url": filing["url"]})
    if cik is not None:  # only a known CIK gives a working EDGAR page (a foreign ticker would not)
        links.append({"label": "SEC EDGAR filings",
                      "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={int(cik):010d}&type=10-K&dateb=&owner=include&count=40"})
    if info.get("sectorKey") and info.get("industryKey"):
        links.append({"label": "Yahoo industry", "url": f"https://finance.yahoo.com/sectors/{info['sectorKey']}/{info['industryKey']}/"})
    return links


def _group_label(choice: PeerChoice, info: dict[str, Any], sic: int | None, description: str | None) -> tuple[str, str]:
    """(label, where): "Semiconductors (Yahoo industry)" and "in Yahoo's Semiconductors industry"."""
    if choice.source == "yahoo_industry":
        industry = info.get("industry") or "its"
        return f"{industry} (Yahoo industry)", f"in Yahoo's {industry} industry"
    code = f"{int(sic):04d}" if sic is not None else "????"
    digits = int(choice.source[-1]) if choice.source.startswith("sic") else 4
    shown = code if digits == 4 else code[:digits] + "x" * (4 - digits)
    label = f"SIC {shown}" + (f" {description}" if description and digits == 4 else "")
    near = f" (near {description})" if description and digits < 4 else ""
    return label + near, f"with SEC industry code {shown}{near}"


@register_agent
class CompetitiveAgent(BaseAgent):
    """Scores the company against its peers on supplier power, customer power and standing."""

    key = "competitive"
    name = "Competitive Position Agent"
    description = (
        "Suppliers, customers and competitors: payment terms, gross margin and its stability, ROIC, operating margin, "
        "growth and revenue share against the peer median (Yahoo industry or SEC industry code), plus quoted 10-K notes "
        "on competitors, sole-source suppliers and large customers. Evidence only (weight 0): the point-in-time "
        "study (decision_log.md D-34) found no edge, so it has no vote."
    )
    category = "Fundamental"
    score_scale = 20.0
    default_weight = 0.0  # evidence only: the study found no edge (decision_log.md D-34)
    point_in_time = False
    role = OWNER
    slow_note = (
        "looks up peers for every company it rates, so a portfolio review or watchlist scan can take about a "
        "minute when its weekly peer data is not cached yet. It has no vote: if that is too slow, untick it in the "
        "Weights tab and nothing else changes."
    )

    def __init__(self, budget: float = BUDGET_SECONDS) -> None:
        super().__init__()
        self.budget = budget

    def evaluate(self, ticker: str, ohlcv: pd.DataFrame) -> AgentAssessment:
        symbol = ticker.strip().upper()
        try:
            return self._evaluate(symbol, time.monotonic() + self.budget)
        except Exception as exc:  # noqa: BLE001 - providers should not raise, but never crash the run
            logger.warning("Competitive position providers raised for %s: %s", symbol, exc)
            return self.unavailable("peer providers failed", error=f"{type(exc).__name__}: {exc}"[:200])

    # ---------------------------------------------------------------- flow

    def _evaluate(self, symbol: str, deadline: float) -> AgentAssessment:
        try:
            info = _fundamentals._load_info(symbol)
        except Exception as exc:  # noqa: BLE001
            return self.unavailable("Yahoo Finance quote request failed", error=f"{type(exc).__name__}: {exc}"[:200])

        sector, industry = info.get("sector"), info.get("industry")
        quote_type = str(info.get("quoteType") or "").upper()
        if quote_type and quote_type != "EQUITY":
            return self.unavailable("a fund has no suppliers or customers of its own")
        if is_financial_company(sector, industry):
            return self.unavailable("receivables, payables and gross margin do not describe a bank or insurer")
        if is_excluded_sector(sector):
            return self.unavailable(f"{sector} is outside the value framework, so no peer comparison is made")

        name = info.get("shortName") or info.get("longName") or symbol
        sec_ok = has_sec_access()
        cik = _peers._ticker_map().get(symbol) if sec_ok else None
        subject_cap = _peers.market_cap_usd(info)

        card_future = _peers.submit_once(f"card:{symbol}", _peers.company_card, symbol, info, new_filings=True)
        notes_future = _peers.submit_once(f"notes:{symbol}", _filings.tenk_notes, cik, None, name, symbol)
        industry_future = (_peers.submit_once(f"industry:{info['industryKey']}", _peers.industry_members, info["industryKey"])
                           if info.get("industryKey") else None)

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        wait([card_future], timeout=remaining())
        if not card_future.done():
            return self.unavailable("the company's figures are still loading; analyze again in a minute")
        card = card_future.result()
        if card.get("error") or not card.get("years"):
            if "no revenue" in str(card.get("error") or ""):
                return self.unavailable(f"the annual statements for {symbol} have no revenue line, so no ratios can be made", error=card.get("error"))
            return self.unavailable(f"Yahoo has no annual statements for {symbol}; funds and trackers have no suppliers or customers of their own",
                                    error=card.get("error"))
        rates: dict[str, float | None] = {}
        subject = _metrics_of(card, rates)
        if subject.fy_end is None or all(getattr(subject, f) is None for f in ("gross_margin", "operating_margin", "dso", "dpo", "roic")):
            return self.unavailable("the annual statements lack revenue and the lines the rules read")

        members: list[dict[str, Any]] = []
        if industry_future is not None:
            try:
                members = industry_future.result(timeout=remaining())
            except Exception:  # noqa: BLE001 - timed out or failed: the SEC industry code is the fallback
                members = []
        choice, candidates, group_note = self._choose(symbol, name, cik, card, subject_cap, members, info)
        if len(choice.tickers) < MIN_PEERS:
            return self.unavailable(
                "fewer than 3 comparable companies found" + (f" ({group_note})" if group_note else ""),
                raw_data={"links": _links(info, cik, None)})

        caps = {c.ticker: c.market_cap for c in candidates if c.market_cap}
        cards, pending, failed = _peers.peer_cards(choice.tickers, deadline, caps)
        peers = [m for m in (_metrics_of(cards[t], rates) for t in choice.tickers if t in cards)]
        usable = usable_peers(subject, peers)
        stale = len(peers) - len(usable)

        notes = self._notes(notes_future, remaining())
        score = score_competitive(subject, usable, self._score_notes(notes),
                                  weak_peer_group=choice.wide_band or choice.source == "sic2")
        if len(usable) < MIN_PEERS or score.rules_scored == 0:
            waiting = f"; {_plural(len(pending), 'peer')} still loading, analyze again in a minute" if pending else ""
            return self.unavailable(f"fewer than 3 peers with figures ({len(usable)} found{waiting})",
                                    raw_data={"links": _links(info, cik, notes.ref)})

        quality = LIVE if score.quality == LIVE and not pending else PARTIAL
        raw = self._raw_data(symbol, name, info, card, subject, usable, cards, choice, candidates, score, notes, pending, failed, stale)
        return self.assessment(
            score=score.total,
            confidence=score.confidence,
            summary=self._summary(name, usable, raw["peer_where"], subject, score),
            key_factors=self._factors(score, pending, failed, stale),
            raw_data=raw,
            data_quality=quality,
            data_as_of=subject.fy_end,
        )

    def _choose(self, symbol: str, name: str, cik: int | None, card: dict[str, Any], subject_cap: float | None,
                members: list[dict[str, Any]], info: dict[str, Any]) -> tuple[PeerChoice, list[Candidate], str]:
        """Yahoo industry members first; the SEC industry code when that leaves fewer than four peers."""
        identity = {"subject_ticker": symbol, "subject_name": name, "subject_cik": cik}
        yahoo_candidates = [Candidate(m["ticker"], m["name"], m["market_cap"]) for m in members]
        yahoo = choose_peers(subject_cap, yahoo_candidates, **identity)
        if len(yahoo.tickers) >= ENOUGH_PEERS:
            yahoo.source = "yahoo_industry"
            return yahoo, yahoo_candidates, ""
        directory = _peers.sic_directory()
        sic = card.get("sic") or (_peers.sic_records().get(symbol) or {}).get("sic")
        sic_choice = choose_peers_by_sic(subject_cap, sic, directory, **identity) if sic is not None else PeerChoice()
        note = "" if directory else "no SEC industry code list yet: run the value study once"
        if len(sic_choice.tickers) >= len(yahoo.tickers) and sic_choice.tickers:
            return sic_choice, directory, note
        yahoo.source = "yahoo_industry"
        return yahoo, yahoo_candidates, note

    @staticmethod
    def _source_label(card: dict[str, Any], peer_cards: list[dict[str, Any]]) -> str:
        """"SEC EDGAR 10-K figures (company and 6 peers), Yahoo statements (1 peer)"."""
        parts = []
        for source, text in (("SEC", "SEC EDGAR 10-K figures"), ("Yahoo", "Yahoo statements")):
            who = (["company"] if card.get("source") == source else [])
            n = sum(1 for c in peer_cards if c.get("source") == source)
            if n:
                who.append(_plural(n, "peer"))
            if who:
                parts.append(f"{text} ({' and '.join(who)})")
        return ", ".join(parts)

    @staticmethod
    def _notes(future, timeout: float) -> _filings.FilingNotes:
        try:
            return future.result(timeout=timeout)
        except TimeoutError:  # not read within the budget: it keeps running and fills the cache
            return _filings.FilingNotes(error=_filings.NOT_READ_YET, pending=True)
        except Exception as exc:  # noqa: BLE001 - tenk_notes never raises; say so honestly if it ever does
            logger.warning("10-K notes raised: %s: %s", type(exc).__name__, exc)
            return _filings.FilingNotes(error="The 10-K could not be read; analyze again later")

    @staticmethod
    def _score_notes(notes: _filings.FilingNotes) -> dict[str, Any]:
        return {"competitors": notes.competitors, "sourcing": notes.sourcing, "raw_materials": notes.raw_materials,
                "concentration": notes.concentration, "no_major_customer": notes.no_major_customer} if notes.found else {}

    # ---------------------------------------------------------------- text

    @staticmethod
    def _summary(name: str, peers: list[CompanyMetrics], where: str, m: CompanyMetrics, score) -> str:
        """One sentence; a clause is left out when the company's figure or the peer median is missing."""
        medians = score.medians
        clauses = []
        for value, median, text in ((m.gross_margin, medians["gross_margin"][0], "gross margin {} vs {}"),
                                    (m.roic, medians["roic"][0], "ROIC {} vs {}")):
            if value is not None and median is not None:
                clauses.append(text.format(_pct(value), _pct(median)))
        if m.dso is not None and medians["dso"][0] is not None:
            clauses.append(f"gets paid in {_days(m.dso)} vs {_days(medians['dso'][0])}")
        detail = ": " + ", ".join(clauses) if clauses else ""
        return f"{name} against {_plural(len(peers), 'peer')} {where}{detail}; competitive score {score.total:+.0f}. Evidence only."

    @staticmethod
    def _factors(score, pending: list[str], failed: list[str], stale: int) -> list[str]:
        top = sorted((r for r in score.rules if r["points"]), key=lambda r: abs(r["points"]), reverse=True)[:4]
        factors = [f"{r['factor']} ({r['points']:+.0f})" for r in top]
        if score.not_scored:
            factors.append("Not scored: " + "; ".join(score.not_scored))
        if pending:
            factors.append(f"{_plural(len(pending), 'peer')} still loading; analyze again in a minute for the full comparison")
        if failed:
            factors.append(f"No figures for {_plural(len(failed), 'peer')} ({', '.join(failed)}); they were left out")
        if stale:
            factors.append(f"{_plural(stale, 'peer')} left out: the latest annual report is more than 15 months older")
        return factors

    def _filing_notes(self, notes: _filings.FilingNotes, peer_tickers: set[str]) -> list[dict[str, Any]]:
        if not notes.found:
            return [{"topic": "10-K", "text": notes.error or _filings.NO_TENK, "quote": None, "url": None}]
        url = (notes.ref or {}).get("url")
        filed = (notes.ref or {}).get("filed")
        out: list[dict[str, Any]] = []
        if notes.concentration:
            top = notes.concentration[0]
            who = {"government": "A government", "top_n": f"Top {top['n']} customers"}.get(top["kind"], "Largest customer")
            out.append({"topic": "Customers", "text": f"{who} {float(top['pct']):g}% of sales (10-K filed {filed})",
                        "quote": excerpt(top["sentence"]), "url": url})
        elif notes.no_major_customer:
            out.append({"topic": "Customers", "text": "No customer above 10% of sales is disclosed in the 10-K",
                        "quote": excerpt(notes.no_major_customer), "url": url})
        if notes.competitors:
            names = ", ".join(str(c["name"]) for c in notes.competitors[:8])
            inside = [str(c["ticker"]) for c in notes.competitors if c.get("ticker") in peer_tickers]
            out.append({"topic": "Competitors", "text": f"Named in the 10-K: {names}" + (f" (in the peer set: {', '.join(inside)})" if inside else ""),
                        "quote": None, "url": url})
        if notes.sourcing:
            out.append({"topic": "Suppliers", "text": f"Mentions single or sole-source suppliers ({_plural(len(notes.sourcing), 'passage')})",
                        "quote": excerpt(notes.sourcing[0]), "url": url})
        if not out:
            out.append({"topic": "10-K", "text": "No named competitors, sole-source suppliers or customers above 10% of sales found in the 10-K text",
                        "quote": None, "url": url})
        return out

    # ---------------------------------------------------------------- raw_data

    def _raw_data(self, symbol: str, name: str, info: dict[str, Any], card: dict[str, Any], subject: CompanyMetrics,
                  usable: list[CompanyMetrics], cards: dict[str, dict[str, Any]], choice: PeerChoice,
                  candidates: list[Candidate], score, notes: _filings.FilingNotes, pending: list[str], failed: list[str],
                  stale: int) -> dict[str, Any]:
        medians = score.medians
        cik = card.get("cik")
        sic = card.get("sic") or (_peers.sic_records().get(symbol) or {}).get("sic")
        description = (_peers.sic_records().get(symbol) or {}).get("sic_description")
        label, where = _group_label(choice, info, sic, description)
        used = [m.ticker for m in usable]
        names = ", ".join(used)
        caption = f"Against {_plural(len(used), 'peer')} {where}: {names}" + (" (wide size band)" if choice.wide_band else "")
        if pending:
            caption += f"; {len(pending)} still loading"

        share = peer_group_share(subject, usable)

        def count(field: str) -> str:
            return str(medians[field][1])

        def num(value: float | None, digits: int = 0) -> str:
            return "n/a" if value is None else f"{value:.{digits}f}"

        sales_peers = str(len([p for p in usable if p.revenue_usd and p.revenue_3y_ago_usd]))
        rows = [
            ("Gross margin", "Gross margin", _pct(subject.gross_margin), _pct(medians["gross_margin"][0]), count("gross_margin")),
            ("Margin swing (pts)", "Margin swing", num(subject.gross_margin_std, 1), num(medians["gross_margin_std"][0], 1), count("gross_margin_std")),
            ("Operating margin", "Operating margin", _pct(subject.operating_margin), _pct(medians["operating_margin"][0]), count("operating_margin")),
            ("ROIC", "ROIC", _pct(subject.roic), _pct(medians["roic"][0]), count("roic")),
            ("Sales growth, 3 years", "Sales growth", _pct(subject.revenue_cagr_3y), _pct(medians["revenue_cagr_3y"][0]), count("revenue_cagr_3y")),
            ("Days to get paid", "DSO", num(subject.dso), num(medians["dso"][0]), count("dso")),
            ("Days to pay suppliers", "DPO", num(subject.dpo), num(medians["dpo"][0]), count("dpo")),
            ("Days of stock", "DIO", num(subject.dio), num(medians["dio"][0]), count("dio")),
            ("Share of sales, now", "Share of peer-group sales", "n/a" if share is None else _pct(share[0]), "n/a", sales_peers),
            ("Share 3 years ago", "Share of peer-group sales", "n/a" if share is None else _pct(share[1]), "n/a", sales_peers),
            ("EV / FCF", "EV / FCF", _mult(subject.ev_to_fcf), _mult(medians["ev_to_fcf"][0]), count("ev_to_fcf")),
        ]
        peer_tickers = set(used)
        filing = notes.ref if notes.found else None
        data_source = self._source_label(card, [cards[t] for t in used if t in cards])
        metrics_table = [
            {"metric": "Peer group", "value": label + (", widened size band (0.04x to 25x)" if choice.wide_band else ", a tenth to ten times the company's market value")},
            {"metric": "Figures", "value": f"{data_source}; latest fiscal year ends {subject.fy_end}"},
            {"metric": "10-K", "value": (f"filed {filing['filed']}" if filing else (notes.error or "n/a"))},
        ]
        peers_table = {
            "title": "Peers",
            "columns": ["Ticker", "Market cap", "Figures from"],
            "rows": [[t, _money(cards[t].get("market_cap")), str(cards[t].get("source"))] for t in used if t in cards]
                    + [[t, "n/a", "loading"] for t in pending],
        }
        return {
            "score_breakdown": [{"factor": r["factor"], "points": r["points"], "component": r["component"]} for r in score.rules],
            "components": score.components,
            "caption": caption,
            "lead_table": {"title": "Against peers", "tip": LEAD_TIP, "columns": ["Measure", "Company", "Peer median", "Peers"],
                           "rows": [[label_, company, median, n] for label_, _, company, median, n in rows],
                           "tips": [TERMS[term] for _, term, *_rest in rows]},
            "filing_notes": self._filing_notes(notes, peer_tickers),
            "metrics_table": metrics_table,
            "terms": [{"term": term, "text": text} for term, text in TERMS.items()],
            "extra_tables": [peers_table],
            "links": _links(info, cik, filing),
            "peers": [{"ticker": t, "name": cards[t].get("name"), "market_cap": cards[t].get("market_cap"), "source": cards[t].get("source"),
                       "named_in_10k": t in {c.get("ticker") for c in (notes.competitors if notes.found else [])}} for t in used if t in cards],
            "peers_pending": pending,
            "peers_failed": failed,
            "peer_source": choice.source,
            "peer_group_label": label,
            "peer_where": where,
            "wide_band": choice.wide_band,
            "metrics": {"company": subject.to_dict(), "peer_median": {k: v[0] for k, v in medians.items()},
                        "peer_count": {k: v[1] for k, v in medians.items()}},
            "filing": filing,
            "data_source_label": data_source,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "scoring_version": SCORING_VERSION,
        }
