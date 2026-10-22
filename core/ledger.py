"""Event-sourced ledger of a broker account: cash, positions, cost, realised P&L and round trips.

Input is the sanitized event list of ``data.trade_republic`` (EUR account).
Events are processed in time order into cash plus, per ISIN, a share count,
an average-cost basis and FIFO lots. Nothing here needs prices or the network,
so every number can be checked against the raw file.

Conventions (owner's calls, decision_log.md D-23):

- Cash impact of any row is amount + fee + tax. The sum over all rows is the
  account's cash.
- Realised P&L is before fees and taxes: sale proceeds minus the cost of the
  shares sold. Average cost and FIFO are both always computed; ``method`` picks
  the headline one. Over a round trip that ends at zero shares both give the
  same total; they differ in which year a gain lands. The position cost the
  broker records (and its app shows) is FIFO lots including buy fees and
  taxes: the export's migration rows carry it and the ledger checks against it.
- Stock perks are income. Tax corrections keep their exported signs.
- Shares delivered out (FREE_RECEIPT with negative shares) were moved to
  another broker: they leave at cost, as a transfer, not a sale.
- Private-market funds are left out: their cash legs count as money moved out
  of the analysed account, so the cash still reconciles and returns are not
  distorted by an asset with no price.
- External flows are deposits, withdrawals and card spending (spending is a
  withdrawal for performance, not a loss), plus the two transfers above.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from data.trade_republic import event_time

EPS = 1e-6
METHODS = ("average", "fifo")
POSITION_KINDS = ("buy", "sell", "transfer_out", "transfer_in")
EXTERNAL_KINDS = ("deposit", "withdrawal", "card")


@dataclass
class _Lot:
    shares: float
    cost: float  # purchase amount, before fees and taxes
    costs: float  # buy fee and transaction tax, as a positive amount
    date: str
    order: str
    savings_plan: bool


def _year(date: str) -> str:
    return date[:4]


def _exchange_ratios(events: list[dict[str, Any]]) -> dict[str, float]:
    """Shares received per share given up in each ISIN change (EXCHANGE pair), by timestamp; 1.0 is a rename."""
    legs: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if e["kind"] == "isin_change" and e.get("shares"):
            legs[e["ts"]].append(float(e["shares"]))
    out: dict[str, float] = {}
    for ts, qty in legs.items():
        given, got = [-q for q in qty if q < 0], [q for q in qty if q > 0]
        if len(given) == 1 and len(got) == 1 and given[0] > EPS:
            out[ts] = got[0] / given[0]
    return out


def _orders(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Buy and sell rows merged per order (the broker splits whole and fractional shares)."""
    merged: dict[str, dict[str, Any]] = {}
    for e in events:
        if e["kind"] not in ("buy", "sell"):
            continue
        o = merged.get(e["order"])
        if o is None:
            o = merged[e["order"]] = {
                "order": e["order"], "date": e["date"], "ts": e["ts"], "isin": e["isin"], "side": e["kind"],
                "asset_class": e["asset_class"], "shares": 0.0, "amount": 0.0, "fee": 0.0, "tax": 0.0,
                "savings_plan": bool(e.get("savings_plan")), "rows": 0,
            }
        o["shares"] += abs(e["shares"] or 0.0)
        o["amount"] += e["amount"]
        o["fee"] += e["fee"]
        o["tax"] += e["tax"]
        o["rows"] += 1
    out = sorted(merged.values(), key=lambda o: event_time(o["ts"]))
    for o in out:
        o["price"] = abs(o["amount"]) / o["shares"] if o["shares"] > 0 else None
        for k in ("shares", "amount", "fee", "tax"):
            o[k] = round(o[k], 6)
    return out


def build_ledger(events: list[dict[str, Any]], method: str = "average") -> dict[str, Any]:
    """Replay the events. Returns positions, realisations, lots, round trips, income, flows and a summary."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    events = sorted(events, key=lambda e: event_time(e["ts"]))
    ratios = _exchange_ratios(events)
    applied: set[str] = set()

    cash = 0.0
    shares: dict[str, float] = defaultdict(float)
    cost: dict[str, float] = defaultdict(float)  # average-cost basis
    lots: dict[str, deque[_Lot]] = defaultdict(deque)
    instruments: dict[str, dict[str, Any]] = {}
    realisations: list[dict[str, Any]] = []
    lot_records: list[dict[str, Any]] = []
    income: list[dict[str, Any]] = []
    flows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    cash_by_date: dict[str, float] = defaultdict(float)
    fees_by_type: dict[str, float] = defaultdict(float)
    taxes_by_type: dict[str, float] = defaultdict(float)
    episodes: list[dict[str, Any]] = []
    open_ep: dict[str, dict[str, Any]] = {}
    migration: list[dict[str, Any]] = []
    warnings: list[str] = []

    def instrument(e: dict[str, Any]) -> None:
        if not e["isin"]:
            return
        info = instruments.setdefault(e["isin"], {"isin": e["isin"], "name": e["name"], "asset_class": e["asset_class"],
                                                  "first_date": e["date"], "last_date": e["date"], "turbo": None,
                                                  "former_isins": []})
        if e["name"] and e["kind"] in POSITION_KINDS + ("dividend",) and e["asset_class"] != "SYNTHETIC":
            info["name"] = e["name"]  # latest name wins (names change, ISINs do not)
        if e["asset_class"] and e["asset_class"] != "SYNTHETIC":
            info["asset_class"] = e["asset_class"]
        info["last_date"] = e["date"]
        if e.get("turbo"):
            info["turbo"] = e["turbo"]
        if e.get("raw_isin") and e["raw_isin"] not in info["former_isins"]:
            info["former_isins"].append(e["raw_isin"])

    def consume_lots(isin: str, qty: float, date: str, order: str | None, proceeds: float | None, exit_kind: str) -> float:
        """Take ``qty`` shares off the FIFO lots; return their cost. Records each closed piece."""
        taken_cost = 0.0
        remaining = qty
        price = proceeds / qty if proceeds is not None and qty > 0 else None
        while remaining > EPS and lots[isin]:
            lot = lots[isin][0]
            take = min(lot.shares, remaining)
            piece_cost = lot.cost * take / lot.shares
            piece_costs = lot.costs * take / lot.shares
            lot_records.append({
                "isin": isin, "buy_date": lot.date, "buy_order": lot.order, "savings_plan": lot.savings_plan,
                "shares": round(take, 6), "cost": round(piece_cost, 6), "buy_costs": round(piece_costs, 6),
                "exit": exit_kind, "exit_date": date, "exit_order": order,
                "proceeds": round(price * take, 6) if price is not None else None,
            })
            taken_cost += piece_cost
            lot.cost -= piece_cost
            lot.costs -= piece_costs
            lot.shares -= take
            remaining -= take
            if lot.shares <= EPS:
                lots[isin].popleft()
        if remaining > EPS:
            warnings.append(f"{isin}: {remaining:.6f} shares left the account without matching buys")
        return taken_cost

    def new_episode(e: dict[str, Any]) -> dict[str, Any]:
        ep = {"isin": e["isin"], "asset_class": e["asset_class"], "start": e["date"], "end": None, "end_reason": "open",
              "buy_orders": set(), "sell_orders": set(), "bought": 0.0, "sold": 0.0, "shares_bought": 0.0,
              "shares_sold": 0.0, "realised_avg": 0.0, "realised_fifo": 0.0, "fees": 0.0, "trade_taxes": 0.0,
              "savings_plan_bought": 0.0, "peak_cost": 0.0, "transferred_cost": 0.0, "transferred_shares": 0.0}
        episodes.append(ep)
        open_ep[e["isin"]] = ep
        return ep

    for e in events:
        cash += e["cash"]
        cash_by_date[e["date"]] += e["cash"]
        if e["fee"]:
            fees_by_type[e["type"]] += e["fee"]
        if e["tax"]:
            taxes_by_type[e["type"]] += e["tax"]
        kind = e["kind"]
        isin = e["isin"]
        instrument(e)

        if kind in EXTERNAL_KINDS:
            flows.append({"date": e["date"], "kind": kind, "amount": e["cash"]})
        elif kind == "private":
            if e["cash"]:
                flows.append({"date": e["date"], "kind": "private", "amount": e["cash"], "isin": isin or None})
        elif kind in ("interest", "dividend", "perk", "bonus"):
            income.append({"date": e["date"], "isin": isin or None, "kind": kind, "type": e["type"],
                           "amount": e["amount"], "tax": e["tax"]})
        elif kind == "buy":
            q = e["shares"] or 0.0
            c = -e["amount"]
            if shares[isin] <= EPS:
                shares[isin], cost[isin] = 0.0, 0.0
                ep = new_episode(e)
            else:
                ep = open_ep.get(isin) or new_episode(e)
            shares[isin] += q
            cost[isin] += c
            lots[isin].append(_Lot(q, c, -(e["fee"] + e["tax"]), e["date"], e["order"], bool(e.get("savings_plan"))))
            ep["buy_orders"].add(e["order"])
            ep["bought"] += c
            ep["shares_bought"] += q
            ep["fees"] += e["fee"]
            ep["trade_taxes"] += e["tax"]
            if e.get("savings_plan"):
                ep["savings_plan_bought"] += c
            ep["peak_cost"] = max(ep["peak_cost"], cost[isin])
            changes.append({"date": e["date"], "isin": isin, "delta": q, "cost": c})
        elif kind == "sell":
            q = -(e["shares"] or 0.0)
            proceeds = e["amount"]
            avg = cost[isin] / shares[isin] if shares[isin] > EPS else 0.0
            cost_avg = avg * q
            cost_fifo = consume_lots(isin, q, e["date"], e["order"], proceeds, "sold")
            shares[isin] -= q
            cost[isin] -= cost_avg
            rec = {"date": e["date"], "isin": isin, "order": e["order"], "shares": round(q, 6),
                   "proceeds": round(proceeds, 6), "cost_avg": round(cost_avg, 6), "cost_fifo": round(cost_fifo, 6),
                   "pnl_avg": round(proceeds - cost_avg, 6), "pnl_fifo": round(proceeds - cost_fifo, 6),
                   "fee": e["fee"], "tax": e["tax"]}
            realisations.append(rec)
            ep = open_ep.get(isin)
            if ep is not None:
                ep["sell_orders"].add(e["order"])
                ep["sold"] += proceeds
                ep["shares_sold"] += q
                ep["realised_avg"] += rec["pnl_avg"]
                ep["realised_fifo"] += rec["pnl_fifo"]
                ep["fees"] += e["fee"]
                ep["trade_taxes"] += e["tax"]
            if shares[isin] <= EPS:
                shares[isin], cost[isin] = 0.0, 0.0
                if ep is not None:
                    ep["end"], ep["end_reason"] = e["date"], "sold"
                    open_ep.pop(isin, None)
            changes.append({"date": e["date"], "isin": isin, "delta": -q, "cost": -cost_avg})
        elif kind == "transfer_out":
            q = -(e["shares"] or 0.0)
            avg = cost[isin] / shares[isin] if shares[isin] > EPS else 0.0
            out_cost = avg * q
            consume_lots(isin, q, e["date"], None, None, "transferred")
            shares[isin] -= q
            cost[isin] -= out_cost
            flows.append({"date": e["date"], "kind": "securities_out", "amount": -round(out_cost, 6), "isin": isin,
                          "shares": round(q, 6)})
            ep = open_ep.get(isin)
            if ep is not None:
                ep["transferred_cost"] += out_cost
                ep["transferred_shares"] += q
            if shares[isin] <= EPS:
                shares[isin], cost[isin] = 0.0, 0.0
                if ep is not None:
                    ep["end"], ep["end_reason"] = e["date"], "transferred"
                    open_ep.pop(isin, None)
            changes.append({"date": e["date"], "isin": isin, "delta": -q, "cost": -out_cost})
        elif kind == "transfer_in":
            q = e["shares"] or 0.0
            basis = q * (e["price"] or 0.0)
            warnings.append(f"{isin}: {q:g} shares delivered in on {e['date']} without a cost basis in the export"
                            + ("" if basis else "; counted at zero cost"))
            if shares[isin] <= EPS:
                new_episode(e)
            shares[isin] += q
            cost[isin] += basis
            lots[isin].append(_Lot(q, basis, 0.0, e["date"], e["id"], False))
            flows.append({"date": e["date"], "kind": "securities_in", "amount": round(basis, 6), "isin": isin,
                          "shares": round(q, 6)})
            changes.append({"date": e["date"], "isin": isin, "delta": q, "cost": basis})
        elif kind == "migration" and (e["shares"] or 0) > 0 and e["price"]:
            # The broker's recorded cost per share: FIFO lots including buy fees and taxes (all 16 match in the
            # owner's export, decision_log.md D-23), so this is the cross-check of the lot accounting.
            held = sum(lot.shares for lot in lots[isin])
            ours = sum(lot.cost + lot.costs for lot in lots[isin]) / held if held > EPS else None
            migration.append({"date": e["date"], "isin": isin, "broker_avg": e["price"],
                              "our_avg": round(ours, 4) if ours else None,
                              "diff_pct": round(ours / e["price"] - 1.0, 5) if ours else None})
        elif kind == "isin_change" and e["ts"] in ratios and e["ts"] not in applied:
            # Both legs carry the new ISIN; a ratio other than 1 (reverse split, merger) rescales the holding
            # and its lots at unchanged cost.
            applied.add(e["ts"])
            ratio = ratios[e["ts"]]
            if abs(ratio - 1.0) > 1e-9 and shares[isin] > EPS:
                before = shares[isin]
                shares[isin] = before * ratio
                for lot in lots[isin]:
                    lot.shares *= ratio
                changes.append({"date": e["date"], "isin": isin, "delta": shares[isin] - before, "cost": 0.0,
                                "ratio": round(ratio, 6)})
        # tax corrections, rights, unknown rows and the cash of ISIN changes: cash only (already counted)

    for ep in episodes:
        ep["buy_orders"] = len(ep["buy_orders"])
        ep["sell_orders"] = len(ep["sell_orders"])
    _attach_income(episodes, income)
    for ep in episodes:
        ep["avg_buy"] = ep["bought"] / ep["shares_bought"] if ep["shares_bought"] > EPS else None
        ep["avg_sell"] = ep["sold"] / ep["shares_sold"] if ep["shares_sold"] > EPS else None
        ep["name"] = instruments.get(ep["isin"], {}).get("name")
        ep["savings_plan_share"] = ep["savings_plan_bought"] / ep["bought"] if ep["bought"] > 0 else 0.0
        for k in ("bought", "sold", "shares_bought", "shares_sold", "realised_avg", "realised_fifo", "fees",
                  "trade_taxes", "savings_plan_bought", "peak_cost", "transferred_cost", "transferred_shares",
                  "dividends", "dividend_tax"):
            ep[k] = round(ep[k], 6)

    for isin, q in lots.items():
        for lot in q:
            lot_records.append({"isin": isin, "buy_date": lot.date, "buy_order": lot.order,
                                "savings_plan": lot.savings_plan, "shares": round(lot.shares, 6),
                                "cost": round(lot.cost, 6), "buy_costs": round(lot.costs, 6), "exit": "open",
                                "exit_date": None, "exit_order": None, "proceeds": None})

    positions = []
    for isin, q in shares.items():
        if q <= EPS:
            continue
        ep = open_ep.get(isin, {})
        fifo_cost = sum(lot.cost for lot in lots[isin])
        broker_cost = fifo_cost + sum(lot.costs for lot in lots[isin])
        positions.append({
            "isin": isin, "name": instruments[isin]["name"], "asset_class": instruments[isin]["asset_class"],
            "shares": round(q, 6), "cost_avg": round(cost[isin], 6), "cost_fifo": round(fifo_cost, 6),
            "avg_price": round(cost[isin] / q, 6), "opened": ep.get("start"),
            # What the broker app shows as the position's cost: FIFO lots with their buy fees and taxes.
            "cost_broker": round(broker_cost, 6), "avg_price_broker": round(broker_cost / q, 6),
        })
    positions.sort(key=lambda p: -p["cost_avg"])

    orders = _orders(events)
    summary = _summary(cash, flows, income, realisations, lot_records, fees_by_type, taxes_by_type, positions,
                       episodes, events)
    return {
        "method": method,
        "cash": round(cash, 6),
        "positions": positions,
        "orders": orders,
        "realisations": realisations,
        "lots": lot_records,
        "episodes": episodes,
        "income": income,
        "flows": flows,
        "changes": changes,
        "cash_by_date": {d: round(v, 6) for d, v in sorted(cash_by_date.items())},
        "instruments": instruments,
        "fees_by_type": {k: round(v, 6) for k, v in fees_by_type.items()},
        "taxes_by_type": {k: round(v, 6) for k, v in taxes_by_type.items()},
        "migration_check": migration,
        "warnings": warnings,
        "summary": summary,
    }


def _attach_income(episodes: list[dict[str, Any]], income: list[dict[str, Any]]) -> None:
    """Dividends go to the round trip that held the shares (paid after the sale: the last one before)."""
    by_isin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        ep["dividends"] = 0.0
        ep["dividend_tax"] = 0.0
        by_isin[ep["isin"]].append(ep)
    for inc in income:
        if inc["kind"] != "dividend" or not inc["isin"]:
            continue
        candidates = [ep for ep in by_isin.get(inc["isin"], []) if ep["start"] <= inc["date"]]
        if not candidates:
            continue
        ep = candidates[-1]
        ep["dividends"] += inc["amount"]
        ep["dividend_tax"] += inc["tax"]


def _summary(cash: float, flows: list[dict[str, Any]], income: list[dict[str, Any]],
             realisations: list[dict[str, Any]], lot_records: list[dict[str, Any]],
             fees_by_type: dict[str, float], taxes_by_type: dict[str, float], positions: list[dict[str, Any]],
             episodes: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    def total(kind: str) -> float:
        return round(sum(f["amount"] for f in flows if f["kind"] == kind), 2)

    realised: dict[str, dict[str, float]] = {}
    for r in realisations:
        y = realised.setdefault(_year(r["date"]), {"average": 0.0, "fifo": 0.0})
        y["average"] += r["pnl_avg"]
    for lot in lot_records:
        if lot["exit"] == "sold":
            y = realised.setdefault(_year(lot["exit_date"]), {"average": 0.0, "fifo": 0.0})
            y["fifo"] += lot["proceeds"] - lot["cost"]
    inc = lambda types, field="amount": round(sum(i[field] for i in income if i["type"] in types), 2)  # noqa: E731
    deposits, withdrawals, card = total("deposit"), total("withdrawal"), total("card")
    closed = [ep for ep in episodes if ep["end_reason"] != "open"]
    return {
        "cash": round(cash, 2),
        "deposits": deposits,
        "withdrawals": withdrawals,
        "card_spending": card,
        "net_external_flows": round(deposits + withdrawals + card, 2),
        "securities_moved_out_at_cost": -total("securities_out"),
        "securities_moved_in": total("securities_in"),
        "private_funds_moved_out": -total("private"),
        "fees": round(sum(fees_by_type.values()), 2),
        "fees_by_type": {k: round(v, 2) for k, v in sorted(fees_by_type.items())},
        "taxes": round(sum(taxes_by_type.values()), 2),
        "taxes_by_type": {k: round(v, 2) for k, v in sorted(taxes_by_type.items())},
        "dividends_gross": inc(("DIVIDEND",)),
        "dividends_tax": inc(("DIVIDEND",), "tax"),
        "liquidation_proceeds": inc(("LIQUIDATION_PROCEEDS",)),
        "interest_gross": inc(("INTEREST_PAYMENT",)),
        "interest_tax": inc(("INTEREST_PAYMENT",), "tax"),
        "perks": inc(("STOCKPERK",)),
        "bonuses": inc(("BONUS",)),
        "realised_by_year": {y: {k: round(v, 2) for k, v in d.items()} for y, d in sorted(realised.items())},
        "realised_total": {m: round(sum(d[m] for d in realised.values()), 2) for m in METHODS},
        "open_positions": len(positions),
        "cost_of_open_positions": round(sum(p["cost_avg"] for p in positions), 2),
        # The broker's basis (FIFO lots with their buy fees and taxes): what the portfolio's average costs use.
        "broker_cost_of_open_positions": round(sum(p["cost_broker"] for p in positions), 2),
        "round_trips_closed": len(closed),
        "round_trips_open": len(episodes) - len(closed),
        "first_trade": min((e["date"] for e in events if e["kind"] in ("buy", "sell")), default=None),
        "last_event": max((e["date"] for e in events), default=None),
    }


def holdings_by_date(ledger: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Share count per ISIN after each day with a change: {date: {isin: shares}} (only changed ISINs listed)."""
    out: dict[str, dict[str, float]] = {}
    running: dict[str, float] = defaultdict(float)
    for c in sorted(ledger["changes"], key=lambda c: c["date"]):
        running[c["isin"]] += c["delta"]
        if abs(running[c["isin"]]) <= EPS:
            running[c["isin"]] = 0.0
        out.setdefault(c["date"], {})[c["isin"]] = running[c["isin"]]
    return out
