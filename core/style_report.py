"""The shareable version of a track record: your style in numbers, with no amounts.

Written from the stored result of ``core.track_record.run_track_record`` to
``analysis/output/my_style.html`` (git-ignored). Everything is a share, a
count, a number of days or a return: no EUR figure of any kind leaves the
machine with it. Instrument names are included only when asked. The page is a
single file with no external requests, so it can be sent as it is.
"""

from __future__ import annotations

import base64
import html
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import config.settings as settings

REPORT_NAME = "my_style.html"
MAX_TAGS = 8


def share_report_path() -> Path:
    return settings.OUTPUT_DIR / REPORT_NAME


# ================================================================== formatting (no amounts, ever)


def _pct(v: Any, digits: int = 0, signed: bool = False) -> str:
    if v is None or not isinstance(v, (int, float)):
        return "n/a"
    return f"{v:+.{digits}%}" if signed else f"{v:.{digits}%}"


def _num(v: Any, digits: int = 0) -> str:
    if v is None or not isinstance(v, (int, float)):
        return "n/a"
    return f"{v:,.{digits}f}"


def _days(v: Any) -> str:
    return "n/a" if v is None else f"{v:.0f} days"


def _n(count: int, one: str, many: str) -> str:
    """The count with the right noun: "1 case", "2 cases"."""
    return f"{count} {one if count == 1 else many}"


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _sign(v: Any) -> str:
    return "" if not isinstance(v, (int, float)) else ("pos" if v > 0 else "neg" if v < 0 else "")


def _tile(label: str, value: str, sub: str = "", cls: str = "") -> str:
    return (f"<div class='tile'><div class='label'>{_e(label)}</div><div class='value {cls}'>{_e(value)}</div>"
            f"<div class='sub'>{_e(sub)}</div></div>")


def _bars(rows: list[tuple[str, float | None, str]], color: str = "var(--c1)") -> str:
    """Horizontal bars: (label, share 0..1, text). Values are direct-labeled, so the bar is a picture of the text."""
    top = max([abs(v) for _, v, _ in rows if isinstance(v, (int, float))] + [1e-9])
    out = []
    for label, v, text in rows:
        w = 0.0 if not isinstance(v, (int, float)) else abs(v) / top * 100.0
        out.append(f"<div class='bar-row'><span class='bar-label'>{_e(label)}</span>"
                   f"<span class='bar-track'><span class='bar' style='width:{w:.1f}%;background:{color}'></span></span>"
                   f"<span class='bar-text'>{_e(text)}</span></div>")
    return "<div class='bars'>" + "".join(out) + "</div>"


def _dbars(rows: list[tuple[str, float | None, str]]) -> str:
    """Diverging bars around zero: (label, signed share, text)."""
    top = max([abs(v) for _, v, _ in rows if isinstance(v, (int, float))] + [1e-9])
    out = []
    for label, v, text in rows:
        w = 0.0 if not isinstance(v, (int, float)) else abs(v) / top * 100.0
        pos = isinstance(v, (int, float)) and v >= 0
        bar = f"<span class='bar {'pos' if pos else 'neg'}' style='width:{w:.1f}%'></span>"
        out.append(f"<div class='bar-row'><span class='bar-label'>{_e(label)}</span><span class='dtrack'>"
                   f"<span class='half left'>{'' if pos else bar}</span><span class='half'>{bar if pos else ''}</span></span>"
                   f"<span class='bar-text'>{_e(text)}</span></div>")
    return "<div class='bars'>" + "".join(out) + "</div>"


def _table(columns: list[str], rows: list[list[Any]], num_from: int = 1, text_cols: tuple[int, ...] = ()) -> str:
    """Columns from ``num_from`` on are right-aligned numbers that do not wrap, except ``text_cols`` (a verdict)."""
    num = lambda i: i >= num_from and i not in text_cols  # noqa: E731
    head = "".join(f"<th class='{'num' if num(i) else ''}'>{_e(c)}</th>" for i, c in enumerate(columns))
    # The label of each cell is its column header: on a phone the table stacks and the cell carries it (see STYLE).
    def cell(i: int, v: Any) -> str:
        label = "" if i == 0 else f" data-label='{_e(columns[i])}'"
        return f"<td class='{'num' if num(i) else ''}'{label}>{_e(v)}</td>"

    body = "".join("<tr>" + "".join(cell(i, v) for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f"<div class='wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


# ================================================================== sections


def _header(r: dict[str, Any], names: bool) -> str:
    s = r.get("summary") or {}
    p = r.get("period") or {}
    first, end = str(p.get("first_trade") or "")[:10], str(p.get("end") or "")[:10]
    years = None
    try:
        years = (datetime.fromisoformat(end) - datetime.fromisoformat(first)).days / 365.25
    except ValueError:
        pass
    benches = s.get("benchmarks") or []  # missing in a result made before D-36
    if benches:
        names_ = " · ".join(f"{b['label']} {_pct(b.get('annual'), 1, True)}" for b in benches)
        tiles = (_tile("Invested part a year", _pct(s.get("twr_invested_annual"), 1, True),
                       f"time-weighted, cash left out · {names_}", _sign(s.get("twr_invested_annual")))
                 + _tile("Money-weighted a year", _pct(s.get("invested_xirr"), 1, True),
                         f"the same buys in {benches[0]['label']} {_pct(benches[0].get('buys_xirr'), 1, True)}")
                 + _tile("Whole account a year", _pct(s.get("twr_annual"), 1, True),
                         f"cash included · cash was {_pct(s.get('avg_cash_share'))} of the account on average"))
        against = "MSCI World, the S&P 500 and the Nasdaq 100"
        old = ""
    else:
        tiles = (_tile("Return a year", _pct(s.get("twr_annual"), 1, True), f"time-weighted · MSCI World {_pct(s.get('benchmark_annual'), 1, True)}",
                       _sign(s.get("twr_annual")))
                 + _tile("Invested part a year", _pct(s.get("twr_invested_annual"), 1, True), f"cash was {_pct(s.get('avg_cash_share'))} of the account on average"))
        against = "MSCI World"
        old = ("<p class='muted small'>Made before cash was left out of the comparison: run My track record again for the invested "
               "part against the S&P 500 and the Nasdaq 100.</p>")
    labels = {b["id"]: b["label"] for b in benches}
    growth = r.get("growth") or {}
    chart = ""
    if benches and growth.get("benchmarks") and growth.get("invested_part"):
        chart = ("<h2>Growth of 100</h2><p class='muted small'>Time-weighted from the first trade; cash left out.</p>"
                 + _growth_svg(growth, labels, end=end))
    return (f"<header><h1>My investing style, in numbers</h1>"
            f"<p class='muted small'>Shared from FastValue{_try_link()}</p>"
            f"<p class='muted'>{_e(first)} to {_e(end)}{'' if years is None else f' ({years:.1f} years)'} · one brokerage account, "
            f"every trade replayed day by day against {against} · generated {_e(datetime.now().strftime('%Y-%m-%d'))}</p>"
            f"<p class='muted small'>Shared without amounts: shares, counts, holding times and returns only. "
            f"{'Instrument names included.' if names else 'Instrument names left out.'}</p></header>"
            "<section class='tiles'>" + tiles + "</section>" + old + chart)


def _try_link() -> str:
    """A "Try FastValue" link to the public address, when the owner has set one (FASTVALUE_PUBLIC_URL); else nothing."""
    url = (getattr(settings, "PUBLIC_URL", "") or "").strip()
    if not re.match(r"https?://[^\s'\"<>]+$", url):
        return ""
    return f" <a href='{_e(url)}'>Try FastValue</a>"


def _growth_svg(growth: dict[str, Any], labels: dict[str, str], end: str = "") -> str:
    """Growth of 100 of the invested part and each index, as an inline SVG with a static legend (index levels only).

    Weekly points are labelled by the week's end, which can fall after the day the page was made from: the right axis
    label stops at ``end`` (values and point positions stay)."""
    lines = {"invested_part": growth.get("invested_part") or [], **(growth.get("benchmarks") or {})}
    colors = dict(zip(lines, ("var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)")))
    pts = {k: [(datetime.fromisoformat(p["time"]), p["value"]) for p in v if isinstance(p.get("value"), (int, float))]
           for k, v in lines.items()}
    allp = [p for v in pts.values() for p in v]
    if not allp:
        return ""
    t0, t1 = min(p[0] for p in allp), max(p[0] for p in allp)
    v0, v1 = min(p[1] for p in allp), max(p[1] for p in allp)
    w, h, pad = 900, 260, 34
    span_t, span_v = max((t1 - t0).days, 1), (v1 - v0) or 1.0
    drawn = []
    for k, v in pts.items():
        d = " ".join(f"{pad + (t - t0).days / span_t * (w - 2 * pad):.1f},{h - pad - (y - v0) / span_v * (h - 2 * pad):.1f}" for t, y in v)
        drawn.append(f"<polyline fill='none' stroke='{colors.get(k, 'var(--fg3)')}' stroke-width='{2.5 if k == 'invested_part' else 1.5}' points='{d}'/>")
    names = {"invested_part": "Invested part", **labels}
    legend = " ".join(f"<span><span class='sw' style='background:{colors.get(k, 'var(--fg3)')}' aria-hidden='true'></span><span class='muted'>{_e(names.get(k, k))}</span></span>" for k in pts)
    last = t1.date()
    try:
        last = min(last, date.fromisoformat(str(end)[:10]))
    except ValueError:
        pass  # no usable end date: the last point's own day
    axes = (f"<text x='{pad}' y='{h - 8}' class='ax'>{t0.date()}</text><text x='{w - pad}' y='{h - 8}' class='ax' text-anchor='end'>{last}</text>"
            f"<text x='4' y='{pad}' class='ax'>{v1:.0f}</text><text x='4' y='{h - pad}' class='ax'>{v0:.0f}</text>")
    against = ", ".join(names.get(k, k) for k in pts if k != "invested_part")
    label = f"Growth of 100 since the first trade: invested part against {against}" if against else "Growth of 100 since the first trade: invested part"
    return f"<p class='legend'>{legend}</p><svg viewBox='0 0 {w} {h}' role='img' aria-label='{_e(label)}'>{''.join(drawn)}{axes}</svg>"


def _style_section(st: dict[str, Any], trips: list[dict[str, Any]], names: bool) -> str:
    rt, hr, pc, b, py, td, dv = (st["round_trips"], st["hit_rate"], st["concentration_of_pnl"], st["bets"], st["pyramiding"],
                                 st["today"], st["dividends"])
    costs = st.get("costs") or {}
    plan_share = None
    if (b.get("plan_total") or 0) + (b.get("discretionary_total") or 0) > 0:
        plan_share = b["plan_total"] / (b["plan_total"] + b["discretionary_total"])
    tiles = [
        _tile("Closed round trips", _num(rt["closed"]), f"{rt['open']} still open · median {_days(rt['median_days_closed'])}"),
        _tile(f"Closed within {rt['quick_days']} days", _pct(rt["share_closed_quickly"]), "of the closed round trips"),
        _tile("Hit rate", _pct(hr["rate"]), f"{_n(hr['wins'], 'win', 'wins')}, {_n(hr['losses'], 'loss', 'losses')}"),
        _tile("Payoff", _num(hr["payoff"], 2), f"average win over average loss · profit factor {_num(hr['profit_factor'], 2)}"),
        _tile("Positions for 80% of gains", _num(pc["positions_for_80pct_of_gains"]), f"of {_n(pc['positions_with_gains'], 'winner', 'winners')} · top 8 are {_pct(pc['top8_share'])} of the P&L"),
        _tile("One-off bets", _num(b["discretionary_buys"]), f"across {_n(b['discretionary_names'], 'name', 'names')} · savings plans were {_pct(plan_share)} of the money"),
        _tile("Buys per position", _num(py["median_buys_per_trip"]), f"median · {py['trips_with_5plus_buys']} built with 5+ buys carry {_pct(py['pnl_share_5plus_buys'])} of the P&L"),
        _tile("Top 3 today", _pct(td["top3_weight"]), f"largest {_pct(td['top1_weight'])} · {_n(td['positions'], 'position', 'positions')}, {_num(td.get('effective_positions'), 1)} effective"),
        _tile("Cash", _pct(td["cash_share"]), f"now · {_pct(td.get('avg_cash_share'))} on average"),
        _tile("Still unrealised", _pct(td["unrealised_share_of_gain"]), "of the total P&L"),
        _tile("One-off dividends", _pct(dv["one_off_share"]), f"of dividend income · {_n(dv['names_recurring'], 'name', 'names')} {'pays' if dv['names_recurring'] == 1 else 'pay'} regularly"),
        _tile("Turnover", _pct(costs.get("turnover_per_year")), "of the invested value a year"),
    ]
    top5 = py.get("top5") or []
    top_rows = [[(t["name"] if names else f"Position {i + 1}"), t.get("buys"), _days(t.get("build_days")), _days(t.get("days")),
                 _pct(t.get("return_pct"), 0, True), t.get("outcome")] for i, t in enumerate(top5)]
    return ("<h2>Your style</h2><section class='tiles'>" + "".join(tiles) + "</section>"
            + "<h3>The five largest results</h3>"
            + _table(["Position", "Buys", "Built over", "Held", "Return", "Status"], top_rows, 1))


def _cuts_section(cuts: dict[str, Any], buckets: list[dict[str, Any]]) -> str:
    hold = cuts.get("holding period") or []
    total = sum(r["n"] for r in hold) or 1
    parts = ["<h2>Where the results came from</h2><div class='grid'>"]
    parts.append("<div class='card'><h3>Holding period</h3>" + _bars(
        [(r["label"], r["n"] / total, f"{_n(r['n'], 'trip', 'trips')} · {_pct(r['hit_rate'])} won · {_pct(r['pnl_share'], 0, True)} of P&L") for r in hold]) + "</div>")
    nb = cuts.get("number of buys") or []
    tot_b = sum(r["n"] for r in nb) or 1
    parts.append("<div class='card'><h3>Number of buys per position</h3>" + _bars(
        [(r["label"], r["n"] / tot_b, f"{_n(r['n'], 'trip', 'trips')} · {_pct(r['hit_rate'])} won · {_pct(r['pnl_share'], 0, True)} of P&L") for r in nb],
        "var(--c2)") + "</div>")
    parts.append("<div class='card'><h3>Core, satellite, metals, speculative</h3>" + _bars(
        [(g["bucket"], g.get("weight_now"), f"{_pct(g.get('weight_now'))} of today's holdings · {_n(g['instruments'], 'instrument', 'instruments')} · {_pct(g.get('return_pct'), 0, True)}")
         for g in buckets], "var(--c3)")
        + "<p class='muted small'>Core: broad index, bond and money-market funds. Metals: physical metal ETCs. Speculative: knock-out products. "
          "Satellite: single stocks and thematic funds.</p></div>")
    vol = cuts.get("volatility class") or []
    tot_v = sum(r["n"] for r in vol) or 1
    labels = {"low": "steady names", "mid": "average volatility", "high": "volatile names"}
    parts.append("<div class='card'><h3>Volatility of the names</h3>" + _bars(
        [(labels.get(r["label"], r["label"]), r["n"] / tot_v, f"{_n(r['n'], 'trip', 'trips')} · {_pct(r['hit_rate'])} won · median {_days(r['median_days'])}") for r in vol],
        "var(--c1)") + "</div>")
    parts.append("</div>")
    ed = cuts.get("entry drawdown") or []
    if ed:
        parts.append("<h3>Share of the P&L by how far below its 52-week high the first buy was</h3>"
                     + _dbars([(r["label"], r.get("pnl_share"), f"{_n(r['n'], 'trip', 'trips')} · {_pct(r['hit_rate'])} won · {_pct(r['pnl_share'], 0, True)}") for r in ed]))
    return "".join(parts)


def _entries_section(entries: dict[str, Any], protocol: str) -> str:
    tags = (entries.get("tags") or [])[:MAX_TAGS]
    rows = [[t["tag"], f"{t['best']} ({(next((b['n'] for b in t['buckets'] if b['bucket'] == t['best']), 0))})",
             f"{t['worst']} ({(next((b['n'] for b in t['buckets'] if b['bucket'] == t['worst']), 0))})",
             _pct(t["spread"], 1, True), _pct(t["first_half"], 1, True), _pct(t["second_half"], 1, True),
             "n/a" if t["t"] is None else f"{t['t']:.1f}", t["confidence"]] for t in tags]
    return (f"<h2>What made an entry good</h2><p class='muted'>{entries.get('n', 0)} buys tagged with the conditions at the time. "
            "The gap is the best bucket against the worst, 90 days later, in excess of MSCI World.</p>"
            + _table(["Condition", "Best bucket (buys)", "Worst bucket (buys)", "Gap", "1st half", "2nd half", "t", "Confidence"], rows, 3)
            + f"<p class='muted small'>{_e(protocol)}</p>")


def _regimes_section(review: dict[str, Any]) -> str:
    """Your buys by the market regime the day before: counts and returns only (the weights and advice stay on your machine)."""
    regs = [r for r in (review or {}).get("regimes") or [] if r.get("n")]
    if not regs:
        return ""
    rows = [[r["regime"], r["n"], _pct(r["median"], 1, True), _pct(r["hit_rate"], 0), _pct(r["first_half"], 1, True), _pct(r["second_half"], 1, True),
             (r.get("against_rest") or {}).get("confidence", "too few buys")] for r in regs]
    return ("<h2>By market regime</h2><p class='muted'>Each buy is grouped by the regime the market was in the trading day before "
            "(BULL, CORRECTION, BEAR, CRISIS, from FastValue's macro classifier). Returns are 90 days later, in excess of MSCI World.</p>"
            + _table(["Regime", "Buys", "Median", "Hit rate", "1st half", "2nd half", "Against your other regimes"], rows, 1, (6,)))


def _exits_section(exits: dict[str, Any]) -> str:
    def tile(label: str, o: dict[str, Any]) -> str:
        return _tile(label, _pct(o.get("mean_excess_180"), 1, True) if o.get("n") else "n/a",
                     f"{o.get('n', 0)} sales · 180 days later, vs MSCI World · {o.get('confidence', '')} confidence")
    return ("<h2>Exits</h2><section class='tiles'>"
            + tile("Sold into weakness", exits.get("into_weakness") or {})
            + tile("Steady names sold within 6 months", exits.get("steady_quick") or {})
            + tile("Losers cut within 90 days", exits.get("losers_cut_fast") or {})
            + "</section><p class='muted small'>A positive number means the name kept beating the index after the sale.</p>")


def _rules_section(lab: dict[str, Any]) -> str:
    rows = []
    for r in lab.get("scenarios") or []:
        if r["key"] == "actual":
            rows.append([r["label"], "reference", _pct(r["twr_annual"], 1, True), _pct(r["max_drawdown"], 1), "", "", "", "Reference."])
            continue
        halves = ("+" if r["first_half"] > 0 else "-") + " / " + ("+" if r["second_half"] > 0 else "-")
        rows.append([_strip_amounts(r["label"]), _pct(r["difference_pct"], 1, True), _pct(r["twr_annual"], 1, True), _pct(r["max_drawdown"], 1), halves,
                     "n/a" if r["t"] is None else f"{r['t']:.1f}", r["confidence"], _strip_amounts(r["verdict"])])
    return ("<h2>Rules replayed on the whole history</h2>"
            f"<p>{_e(_strip_amounts(lab.get('verdict')))}</p>"
            + _table(["Rule", "vs what I did (% of account)", "A year", "Max drawdown", "Halves", "t", "Confidence", "Verdict"], rows, 1,
                     text_cols=(6, 7))
            + f"<p class='muted small'>{_e(_strip_amounts(lab.get('protocol')))}</p>")


def _lessons_section(lessons: list[dict[str, Any]]) -> str:
    items = "".join(f"<li><strong>{_e(x['title'])}</strong> <span class='conf {_e(x['confidence'])}'>{_e(x['confidence'])}</span><br>{_e(_strip_amounts(x['text']))}</li>"
                    for x in lessons)
    return f"<h2>Lessons</h2><ul class='lessons'>{items}</ul>"


PAREN_AMOUNT_RE = re.compile(r"\s*\((?:[+\-\u2212]?\d[\d,.]*\s?EUR|[+\-\u2212]?\u20ac\d[\d,.]*)\)")


def _strip_amounts(text: str | None) -> str:
    """Dashboard texts may quote EUR figures; the shared page carries none.

    A figure in brackets goes and its sentence stays ("the strongest rule (+1,234 EUR); it does not pass"). A clause
    that still names EUR goes whole, because a figure cut out of its wording leaves a sentence that does not read
    ("Average win against an average loss of"). Clauses keep their separators, and a clause that now opens a sentence
    gets its capital back.
    """
    original = text or ""
    pieces = re.split(r"(; |\. )", PAREN_AMOUNT_RE.sub("", original))
    kept = [[c, sep] for c, sep in zip(pieces[0::2], pieces[1::2] + [""]) if c.strip() and "EUR" not in c and "\u20ac" not in c]
    if not kept:
        return ""
    kept[-1][1] = ""
    out = "".join(c + sep for c, sep in kept).strip()
    out = re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    return out + "." if original.rstrip().endswith(".") and not out.endswith(".") else out


STYLE = """
:root{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--line:#e6e5e1;--fg:#0b0b0b;--fg2:#52514e;--fg3:#6f6e68;
--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--neg:#e34948;--pos-text:#0e7643;--neg-text:#c9372f;--mid:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--card:#22221f;--line:#34332f;
--fg:#ffffff;--fg2:#c3c2b7;--fg3:#8f8e86;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--neg:#e66767;--pos-text:#199e70;--neg-text:#e66767;--mid:#383835}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--card:#22221f;--line:#34332f;--fg:#ffffff;--fg2:#c3c2b7;--fg3:#8f8e86;
--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--neg:#e66767;--pos-text:#199e70;--neg-text:#e66767;--mid:#383835}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
a{color:var(--c1);text-decoration:underline}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
main{max-width:1040px;margin:0 auto;padding:24px 16px 48px}h1{font-size:1.9rem;margin:0 0 4px;letter-spacing:-0.01em}
h2{font-size:1.2rem;margin:36px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}h3{font-size:0.95rem;margin:14px 0 8px;color:var(--fg2)}
.muted{color:var(--fg2)}.small{font-size:0.82rem}.pos{color:var(--pos-text)}.neg{color:var(--neg-text)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin:12px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tile .label{font-size:0.8rem;font-weight:500;color:var(--fg3)}
.tile .value{font-size:1.45rem;font-weight:650;margin-top:2px}.tile .sub{font-size:0.76rem;color:var(--fg3);margin-top:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px 12px}
.bars{display:flex;flex-direction:column;gap:7px;margin:6px 0}.bar-row{display:grid;grid-template-columns:minmax(110px,170px) minmax(0,1fr) minmax(150px,230px);gap:10px;align-items:center;font-size:0.82rem}
.bar-label{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.bar-track{display:block}.bar{display:block;height:8px;border-radius:0 4px 4px 0;min-width:1px}
.bar-text{color:var(--fg2);font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dtrack{position:relative;display:grid;grid-template-columns:1fr 1fr;align-items:center;min-height:14px}.dtrack::before{content:"";position:absolute;left:50%;top:0;bottom:0;border-left:1px solid var(--line)}
.half{display:flex;align-items:center}.half.left{justify-content:flex-end}.half .bar.pos{background:var(--c3);border-radius:0 4px 4px 0}.half .bar.neg{background:var(--neg);border-radius:4px 0 0 4px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:0.82rem;background:var(--card);border:1px solid var(--line);border-radius:8px}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:0.8rem;font-weight:500;color:var(--fg3)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}tr:last-child td{border-bottom:0}
.lessons{padding-left:18px}.lessons li{margin:8px 0}.conf{font-size:0.7rem;padding:1px 7px;border-radius:999px;border:1px solid var(--line);color:var(--fg3);vertical-align:middle}
svg{width:100%;height:auto}.ax{fill:var(--fg3);font-size:12px}.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:0.82rem}
footer{margin-top:40px;color:var(--fg3);font-size:0.78rem;border-top:1px solid var(--line);padding-top:10px}
@media (max-width:640px){.ax{font-size:30px}.bar-row{grid-template-columns:minmax(90px,120px) minmax(0,1fr)}.bar-text{grid-column:1 / -1;white-space:normal}
table,tbody{display:block}thead{display:none}
tr{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:6px 12px;padding:10px 9px;border-bottom:1px solid var(--line)}tr:last-child{border-bottom:0}
td{display:block;padding:0;border:0;white-space:normal;min-width:0}td.num{text-align:left;white-space:normal}td:first-child{grid-column:1 / -1;font-weight:600}
td[data-label]::before{content:attr(data-label);display:block;font-size:0.72rem;color:var(--fg3)}}
"""


INTER_FILE = Path(__file__).resolve().parents[1] / "web" / "static" / "vendor" / "inter" / "inter-v20-latin.woff2"


@lru_cache(maxsize=1)
def _font_face() -> str:
    """Inter (the latin subset the app vendors) embedded as base64, so the page stays one file and asks nobody for a font."""
    try:
        data = base64.b64encode(INTER_FILE.read_bytes()).decode("ascii")
    except OSError:
        return ""  # the system sans-serif takes over
    return ("@font-face{font-family:'Inter';font-style:normal;font-weight:400 700;font-display:swap;"
            f"src:url(data:font/woff2;base64,{data}) format('woff2')}}")


def write_share_report(result: dict[str, Any], names: bool = False, path: Path | None = None) -> Path:
    """Write the shareable page from a stored track record and return its path."""
    path = path or share_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    style = result.get("style") or {}
    st = style.get("style")
    parts = [_header(result, names)]
    if st:
        parts.append(_style_section(st, result.get("round_trips") or [], names))
        parts.append(_cuts_section(style.get("cuts") or {}, st.get("buckets") or []))
        parts.append(_entries_section(style.get("entries") or {}, style.get("protocol") or ""))
        parts.append(_regimes_section(style.get("weights") or {}))
        parts.append(_exits_section(style.get("exits") or {}))
    else:
        parts.append("<p class='muted'>The style profile did not run for this track record.</p>")
    if result.get("rules_lab"):
        parts.append(_rules_section(result["rules_lab"]))
    if style.get("lessons"):
        parts.append(_lessons_section(style["lessons"]))
    if (result.get("summary") or {}).get("benchmarks"):  # a result made before D-36 has the one index
        read_returns = ("<li>Returns are time-weighted (deposits, withdrawals and idle cash removed), in the account's currency, against MSCI World, "
                        "the S&P 500 and the Nasdaq 100 through their accumulating UCITS funds; the whole account's return, cash included, is shown beside it.</li>")
    else:
        read_returns = "<li>Returns are time-weighted (deposits and withdrawals removed), in the account's currency; the benchmark is MSCI World with the same cash flows.</li>"
    parts.append("<h2>How to read it</h2><ul class='muted small'>"
                 + read_returns +
                 "<li>Round trips run from the first buy to the day the position is back at zero. P&L includes dividends and fees and values open positions at the last close.</li>"
                 "<li>Patterns are judged on cases that are counted once per name and month, by the gap between the best and the worst bucket, "
                 "and must hold in both halves of the period to be more than chance. A thin sample is said to be thin.</li>"
                 "<li>Rules are replayed on the whole history with the same deposits and prices; a rule passes only if it wins over the whole period and in each half alone with t of at least 2. "
                 "It is all in sample on one history: a winner is a hypothesis for the next year.</li></ul>"
                 "<footer>Made with FastValue, a decision desk for value investors that runs on your own computer"
                 f"{_try_link()}. Decision support only, not financial advice. These are past results of one account, measured in sample: "
                 "they do not predict future returns. This page holds no amounts, account numbers or names of people.</footer>")
    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>My investing style</title><link rel='icon' href='data:,'><style>{_font_face()}{STYLE}</style></head><body><main>{''.join(parts)}</main></body></html>")
    path.write_text(page)
    return path
