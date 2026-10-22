"""Daily check: review the portfolio and the watchlist, store the history, and alert on what changed.

Run it by hand or on a schedule (macOS launchd, weekdays after the US close):

    python -m core.monitor                 # one check now
    python -m core.monitor --install       # schedule it (writes ~/Library/LaunchAgents/com.fastvalue.daily.plist)
    python -m core.monitor --uninstall
    python -m core.monitor --show          # the latest alerts

Each check runs the owner-first review on every holding and the watchlist scan, which records every analysis
(``data/store/history.db``), the daily news-tone snapshot and the options IV snapshot. Those stored histories are
what the 30/90-day rating changes, the sentiment validation (``core/sentiment_validation.py``) and the real IV
rank need, and they only grow when something runs every day.

It then compares the result with the previous check (``data/store/monitor_state.json``) and raises an alert when:
- a holding's action changes (EXIT and REVIEW first, then TRIM and ADD; ADD back to HOLD is not worth an alert),
  the monthly red-flag monitor of decision_log.md T-02. When the review's rules change (``RULES_VERSION``), changes
  that do not involve EXIT come as one note instead of one alert each: no rating moved, the rules did;
- a watchlist company's owner score crosses one of the levels set in the Weights tab (rising to the entry level is
  the one to act on);
- a trend rule on a watched market flips between invested and cash (``settings.TREND_WATCH`` and
  ``TREND_ALERT_RULES``, ``core/systematic.py``). The owner asked for these as drawdown alerts (D-48); like every
  alert they inform and change no score.
Timing labels never raise an alert: they are not a rule (D-19). Every alert carries what to do about it.

Alerts are appended to ``data/store/alerts.jsonl`` and, on macOS, shown as a local notification. Nothing is sent
off the machine: tickers and actions only, no amounts. Both files are git-ignored.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import plistlib
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import config.settings as settings

logger = logging.getLogger(__name__)

LABEL = "com.fastvalue.daily"
SEVERITY = {"EXIT": "risk", "REVIEW": "risk", "TRIM": "risk", "ADD": "note", "HOLD": "note"}
BUYS = ("BUY", "STRONG_BUY")
SELLS = ("SELL", "STRONG_SELL")
RANK = {"STRONG_BUY": 2, "BUY": 1, "HOLD": 0, "NOT_RATED": 0, "SELL": -1, "STRONG_SELL": -2}
# Environment variables the scheduled run needs and launchd does not inherit from your shell.
PASS_ENV = ("SEC_USER_AGENT", "ALPHAVANTAGE_API_KEY", "FINNHUB_API_KEY")
DASHBOARD = "http://127.0.0.1:{port}/"


def _n(count: int, one: str, many: str) -> str:
    """The count with the right noun: "1 case", "2 cases"."""
    return f"{count} {one if count == 1 else many}"


def _state_path() -> Path:
    return settings.STORE_DIR / "monitor_state.json"


def _alerts_path() -> Path:
    return settings.STORE_DIR / "alerts.jsonl"


def _link(fragment: str) -> str:
    port = os.environ.get("FASTVALUE_PORT") or os.environ.get("QUANT_PORT") or "8501"
    return DASHBOARD.format(port=port) + fragment


def load_state() -> dict[str, Any]:
    try:
        data = json.loads(_state_path().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def snapshot(review: dict[str, Any] | None, scan: dict[str, Any] | None, at: str,
             trend: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the next check compares against: action, rule and owner decision per holding, decision per watchlist name,
    and invested or cash per watched market and trend rule."""
    from core.portfolio import RULES_VERSION

    holdings = {
        r["ticker"]: {"action": r["action"], "decision": r["decision"], "owner_score": r.get("owner_score"),
                      "reason": (r.get("reasons") or [""])[0], "rule": r.get("rule"), "class": r.get("class"),
                      "rated": r.get("rated", True), "fix": r.get("fix")}
        for r in (review or {}).get("positions") or []
    }
    watch = {
        r["ticker"]: {"decision": r["decision"], "owner_score": r.get("composite_score"),
                      "rated": bool(r.get("owner_applicable", True))}
        for r in (scan or {}).get("rows") or [] if r["ticker"] not in holdings
    }
    markets = {
        f"{m['ticker']}|{key}": {"ticker": m["ticker"], "rule": key, "label": r["label"], "in_market": r["in_market"],
                                 "reason": r.get("reason") or "", "date": r.get("date")}
        for m in (trend or {}).get("markets") or [] if not m.get("error")
        for key, r in (m.get("rules") or {}).items() if key in settings.TREND_ALERT_RULES
    }
    return {"checked_at": at, "rules": RULES_VERSION, "holdings": holdings, "watchlist": watch, "trend": markets}


def _score(v: Any) -> str:
    return f" ({v:+.0f})" if isinstance(v, (int, float)) else ""


_BANDS = ("STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY")  # NOT_RATED sits with HOLD for the ranking
# The level between band i and band i+1: its threshold key and its name in the UI.
_LEVELS = (("strong_sell", "strong exit level"), ("sell", "exit level"), ("buy", "entry level"), ("strong_buy", "strong entry level"))


def _owner_thresholds() -> dict[str, float]:
    try:
        from config.weights import load_scoring_config

        return dict(load_scoring_config().owner_thresholds)
    except Exception:  # noqa: BLE001 - the alert text still reads without the numbers
        return {"strong_buy": 45.0, "buy": 15.0, "sell": -15.0, "strong_sell": -45.0}


def _crossing(old: str, new: str, score: Any) -> str:
    """Plain sentence for the level an owner score crossed between two checks, for example
    "owner score +18 rose to or above the entry level (+15)"."""
    head = f"owner score {score:+.0f}" if isinstance(score, (int, float)) else "owner score"
    if "NOT_RATED" in (old, new):
        return f"{head}: now rated" if old == "NOT_RATED" else "no longer rated by the value framework"
    if old not in _BANDS or new not in _BANDS:
        return f"{head} changed level"
    o, n = _BANDS.index(old), _BANDS.index(new)
    index = n - 1 if n > o else n  # the level that sets the new band
    key, name = _LEVELS[index]
    value = _owner_thresholds().get(key)
    level = f"the {name}" + (f" ({value:+.0f})" if isinstance(value, (int, float)) else "")
    if n > o:
        verb = "rose to or above" if key in ("buy", "strong_buy") else "rose above"
    else:
        verb = "fell below" if key in ("buy", "strong_buy") else "fell to or below"
    return f"{head} {verb} {level}"


def compare(previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """Alerts for every change between two snapshots. The first check only records a baseline."""
    if not previous:
        return []
    at = current["checked_at"]
    alerts: list[dict[str, Any]] = []
    before_h = previous.get("holdings") or {}
    rules_changed = previous.get("rules") != current.get("rules")
    folded: list[str] = []
    for ticker, now in (current.get("holdings") or {}).items():
        old = before_h.get(ticker)
        if old is None or old.get("action") == now["action"]:
            continue
        action = now["action"]
        if old.get("action") == "ADD" and action == "HOLD":
            continue  # the chance passed; nothing to do
        if rules_changed and "EXIT" not in (old.get("action"), action):
            folded.append(ticker)  # the rules moved, not the rating: one note below
            continue
        fix = (now.get("fix") if action != "HOLD" else None) or {
            "EXIT": "Open the review and read the reasons. The owner score fell to the exit level or below, or a red flag appeared; "
                    "the exit rule says sell.",
            "REVIEW": "Open the review and re-check the thesis (share count, net dilution, FCF, ROIC) before acting; "
                      "REVIEW is not an exit signal.",
            "TRIM": "Sell down to the position cap; the review shows the size.",
            "ADD": "Run \"Should I enter?\" on it in the Portfolio tab for the size and staging.",
            "HOLD": "Nothing to do: the earlier action no longer applies.",
        }[action]
        reason = now.get("reason") or ""
        if now.get("rated", True) and reason.startswith("Rated"):
            text = reason  # the reason already states the rating
        elif now.get("rated", True):
            text = f"Owner score {now['owner_score']:+.0f}. {reason}" if isinstance(now.get("owner_score"), (int, float)) else reason
        else:
            text = now.get("reason") or "Reviewed by its class, not by the owner score."
        alerts.append({
            "at": at, "ticker": ticker, "scope": "holding", "kind": SEVERITY.get(action, "note"),
            "title": f"{ticker}: {old['action']} -> {action}", "text": text.strip(),
            "fix": fix, "link": _link(f"#portfolio={ticker}"),
        })
    if folded:
        alerts.append({
            "at": at, "ticker": "", "scope": "holding", "kind": "note", "title": "The review rules changed",
            "text": f"The review rules changed: {len(folded)} holding"
                    f"{'' if len(folded) == 1 else 's'} changed action without a new rating: {', '.join(sorted(folded))}.",
            "fix": "Nothing to do; funds and trackers are now reviewed by their class. Open the review to see each reason.",
            "link": _link("#portfolio=review"),
        })
    before_w = previous.get("watchlist") or {}
    for ticker, now in (current.get("watchlist") or {}).items():
        old = before_w.get(ticker)
        if old is None or old.get("decision") == now["decision"]:
            continue
        into_buy = now["decision"] in BUYS and old["decision"] not in BUYS
        crossing = _crossing(str(old["decision"]), str(now["decision"]), now.get("owner_score"))
        alerts.append({
            "at": at, "ticker": ticker, "scope": "watchlist", "kind": "note",
            "title": f"{ticker}: {crossing}",
            "text": f"Watchlist: {crossing}.",
            "fix": ("Run \"Should I enter?\" in the Portfolio tab: it gives the size and whether to stage the entry."
                    if into_buy else "No action needed unless you were planning to buy it."),
            "link": _link(f"#portfolio={ticker}" if into_buy else f"#analyze={ticker}"),
        })
    alerts += _trend_alerts(previous.get("trend") or {}, current.get("trend") or {}, at)
    order = {"risk": 0, "note": 1}
    alerts.sort(key=lambda a: (order.get(a["kind"], 2), a["scope"] != "holding", a["ticker"]))
    return alerts


def _trend_alerts(before: dict[str, Any], now: dict[str, Any], at: str) -> list[dict[str, Any]]:
    """A watched market whose trend rule flipped since the last check. Markets first seen today are a baseline."""
    alerts = []
    for key, cur in now.items():
        old = before.get(key)
        if old is None or old.get("in_market") == cur["in_market"]:
            continue
        ticker, out = cur["ticker"], not cur["in_market"]
        alerts.append({
            "at": at, "ticker": ticker, "scope": "market", "kind": "risk" if out else "note",
            "title": f"{ticker} {'to cash' if out else 'invested again'} ({cur['label']})",
            "text": f"At the {cur.get('date') or 'latest'} close: {cur['reason']}",
            "fix": ("If you hold a leveraged or satellite position on this market and follow this rule, it would sell at "
                    "the next close. Your core funds and the owner scores are not affected: the rule informs and does "
                    "not decide. Its backtest is in Backtest, Trend and dips." if out else
                    "If you follow this rule, it would buy back at the next close. Its backtest is in Backtest, Trend and dips."),
            "link": _link("#backtest=trend"),
        })
    return alerts


def _append_alerts(alerts: list[dict[str, Any]]) -> None:
    if not alerts:
        return
    path = _alerts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for a in alerts:
            handle.write(json.dumps(a) + "\n")


def recent_alerts(limit: int = 50) -> list[dict[str, Any]]:
    """Latest alerts, newest check first; within a check, in the order it raised them (risk first)."""
    try:
        lines = _alerts_path().read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-max(limit * 4, 200):]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(item)
    out.sort(key=lambda a: str(a.get("at", "")), reverse=True)  # stable: a check's own order survives
    return out[:limit]


def notify(alerts: list[dict[str, Any]]) -> bool:
    """One local macOS notification summarizing the alerts (tickers and actions only). False when not shown."""
    if not alerts or sys.platform != "darwin" or os.environ.get("FASTVALUE_NOTIFY") == "0":
        return False
    risk = [a for a in alerts if a["kind"] == "risk"]
    held = [a for a in risk if a.get("scope") == "holding"]
    head = (f"{_n(len(held), 'holding', 'holdings')} {'needs' if len(held) == 1 else 'need'} a look" if held
            else _n(len(risk), "trend alert", "trend alerts") if risk else _n(len(alerts), "change", "changes"))
    body = "; ".join(a["title"] for a in (risk or alerts)[:4]) + (" ..." if len(risk or alerts) > 4 else "")
    script = f"display notification {json.dumps(body)} with title \"FastValue\" subtitle {json.dumps(head)}"
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=10, capture_output=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def run_check(review_fn=None, scan_fn=None, now: datetime | None = None, trend_fn=None) -> dict[str, Any]:
    """Review the portfolio, the watchlist and the watched markets' trend rules; compare with the previous check,
    store and return the alerts."""
    from core.portfolio import review_portfolio
    from core.watchlist import scan_watchlist
    from data.history import load_watchlist
    from data.portfolio import load_portfolio

    at = (now or datetime.now()).isoformat(timespec="seconds")
    errors: list[str] = []
    review = scan = None
    portfolio = load_portfolio()
    if portfolio["positions"]:
        try:
            review = (review_fn or review_portfolio)(portfolio)
            errors += [f"{e['ticker']}: {e['error']}" for e in review.get("errors") or []]
        except Exception as exc:  # noqa: BLE001 - the watchlist still runs
            logger.exception("Portfolio review failed")
            errors.append(f"portfolio review: {exc}")
    held = {p["ticker"] for p in portfolio["positions"]}
    watch = [t for t in load_watchlist() if t not in held]
    if watch:
        try:
            scan = (scan_fn or scan_watchlist)(watch)
            errors += [f"{e['ticker']}: {e['error']}" for e in scan.get("errors") or []]
        except Exception as exc:  # noqa: BLE001
            logger.exception("Watchlist scan failed")
            errors.append(f"watchlist scan: {exc}")

    trend = None
    if settings.TREND_WATCH:
        try:
            from core.systematic import trend_check

            trend = (trend_fn or trend_check)()
            errors += [f"{m['ticker']}: {m['error']}" for m in trend.get("markets") or [] if m.get("error")]
        except Exception as exc:  # noqa: BLE001 - the review's alerts still go out
            logger.exception("Trend check failed")
            errors.append(f"trend check: {exc}")

    previous = load_state()
    current = snapshot(review, scan, at, trend)
    # A market whose prices failed today keeps its last state too.
    for key, old in (previous.get("trend") or {}).items():
        if key not in current["trend"] and old.get("ticker") in settings.TREND_WATCH:
            current["trend"][key] = old
    # A holding or watchlist name that failed today keeps its last known state, so it does not alert twice.
    for scope in ("holdings", "watchlist"):
        for ticker, old in (previous.get(scope) or {}).items():
            if ticker not in current[scope] and (ticker in held if scope == "holdings" else ticker in watch):
                current[scope][ticker] = old
    alerts = compare(previous, current)
    _append_alerts(alerts)
    _save_state({**current, "errors": errors[:50]})
    return {"checked_at": at, "baseline": not previous, "alerts": alerts, "errors": errors,
            "holdings": len(current["holdings"]), "watchlist": len(current["watchlist"]), "markets": len(current["trend"])}


# ------------------------------------------------------------------ schedule (macOS launchd)


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def default_run_time() -> time:
    """45 minutes after the New York close, in this machine's local time."""
    ny = datetime.now(ZoneInfo("America/New_York")).replace(hour=16, minute=45, second=0, microsecond=0)
    local = ny.astimezone()
    return time(local.hour, local.minute)


def build_plist(at: time, python: str | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    """The launchd job: Monday to Friday at ``at`` (a run missed while asleep starts on wake)."""
    log = str(settings.STORE_DIR / "monitor.log")
    job: dict[str, Any] = {
        "Label": LABEL,
        "ProgramArguments": [python or sys.executable, "-m", "core.monitor"],
        "WorkingDirectory": str(settings.PROJECT_ROOT),
        "StartCalendarInterval": [{"Weekday": d, "Hour": at.hour, "Minute": at.minute} for d in range(1, 6)],
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "ProcessType": "Background",
    }
    if env:
        job["EnvironmentVariables"] = dict(env)
    return job


def install(at: time | None = None) -> str:
    at = at or default_run_time()
    env = {k: os.environ[k] for k in PASS_ENV if os.environ.get(k)}
    path = _plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps(build_plist(at, env=env)))
    path.chmod(0o600)  # it may hold API keys
    subprocess.run(["launchctl", "unload", str(path)], check=False, capture_output=True)
    subprocess.run(["launchctl", "load", str(path)], check=True)
    names = ", ".join(env) if env else "none"
    return (f"Scheduled: weekdays at {at:%H:%M} local time ({path}).\n"
            f"Environment passed to the job: {names}. Run --install again after changing them.\n"
            f"Log: {settings.STORE_DIR / 'monitor.log'}")


def uninstall() -> str:
    path = _plist_path()
    if not path.exists():
        return "No schedule installed."
    subprocess.run(["launchctl", "unload", str(path)], check=False, capture_output=True)
    path.unlink()
    return f"Removed {path}."


def schedule_status() -> dict[str, Any]:
    path = _plist_path()
    if not path.exists():
        return {"installed": False}
    try:
        job = plistlib.loads(path.read_bytes())
        slot = (job.get("StartCalendarInterval") or [{}])[0]
        return {"installed": True, "time": f"{slot.get('Hour', 0):02d}:{slot.get('Minute', 0):02d}"}
    except (OSError, plistlib.InvalidFileException):
        return {"installed": True, "time": None}


def status() -> dict[str, Any]:
    """For the dashboard: last check, its errors, the schedule and the latest alerts."""
    state = load_state()
    return {"checked_at": state.get("checked_at"), "errors": state.get("errors") or [],
            "holdings": len(state.get("holdings") or {}), "watchlist": len(state.get("watchlist") or {}),
            "schedule": schedule_status() if sys.platform == "darwin" else {"installed": False, "unsupported": True},
            "install_command": "python -m core.monitor --install", "alerts": recent_alerts(30)}


def _print_alerts(alerts: list[dict[str, Any]]) -> None:
    for a in alerts:
        print(f"  [{a['kind'].upper():4}] {a['title']}  {a['text']}")
        print(f"         What to do: {a['fix']}  {a.get('link', '')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FastValue daily check: review, record history, alert on changes")
    parser.add_argument("--install", action="store_true", help="Schedule the check on weekdays (macOS launchd)")
    parser.add_argument("--at", default=None, help="Local time for --install, HH:MM (default: 45 min after the US close)")
    parser.add_argument("--uninstall", action="store_true", help="Remove the schedule")
    parser.add_argument("--show", action="store_true", help="Print the latest alerts")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.install or args.uninstall:
        if sys.platform != "darwin":
            print("The built-in schedule uses macOS launchd. Elsewhere, run `python -m core.monitor` from cron.")
            return 1
        if args.uninstall:
            print(uninstall())
            return 0
        at = time.fromisoformat(args.at) if args.at else None
        print(install(at))
        return 0
    if args.show:
        alerts = recent_alerts(20)
        print("No alerts yet." if not alerts else f"Latest {_n(len(alerts), 'alert', 'alerts')}, newest first:")
        _print_alerts(alerts)
        return 0

    result = run_check()
    print(f"FastValue daily check {result['checked_at']}: {_n(result['holdings'], 'holding', 'holdings')}, "
          f"{_n(result['watchlist'], 'watchlist name', 'watchlist names')}.")
    if result["baseline"]:
        print("First check: baseline recorded, alerts start with the next one.")
    elif not result["alerts"]:
        print("No changes since the previous check.")
    _print_alerts(result["alerts"])
    for e in result["errors"]:
        print(f"  ERROR {e}")
    notify(result["alerts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
