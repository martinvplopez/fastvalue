"""FastValue: Flask web application for the decision dashboard and backtester.

Run with ``python web/app.py`` (http://127.0.0.1:8501). Set FASTVALUE_DEBUG=1 for
Flask debug mode; never expose the debug server beyond localhost.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import logging
import math
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

# Add project root to path so absolute imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.base import AgentAssessment
from agents.master_agent import MasterDecisionAgent
from agents.registry import agent_metadata, get_agent_classes
from agents.scoring import aggregate_layers
from analysis.metrics import compute_buy_and_hold_returns, compute_metrics
from analysis.trend_indicators import pb_ema_channel
from config.settings import COMMISSION_PER_SHARE, INITIAL_CASH
from config.weights import (
    REGIMES,
    has_local_overrides,
    load_raw_config,
    load_scoring_config,
    reset_scoring_config,
    save_scoring_config,
    validate_config,
)
from config.data_sources import DATA_SOURCES, VERSION as APP_VERSION
from core import broker_import, history_lab, index_check, lab, monitor, privacy, style_report, systematic, track_record
from core.comparator import compare_strategies
from core.engine import run_backtest
from core.portfolio import assess_candidate, limits as portfolio_limits, refreshed_plan, review_portfolio
from core.score_backtest import FREQUENCIES, backtestable_agents, run_score_backtest
from core.watchlist import analyze_ticker, scan_watchlist
from data.fetcher import fetch_ohlcv
from data.history import get_history, get_latest, load_watchlist, save_watchlist
from data.portfolio import (
    ISIN_RE,
    PortfolioError,
    check_imported_unchanged,
    load_last_review,
    load_portfolio,
    save_last_review,
    remove_position,
    save_portfolio,
    set_notes,
    set_plan,
    upsert_position,
)
from data import brokers, generic_ledger, trade_republic
from data.sec_edgar import has_sec_access, save_contact, sec_access_source
from data.trade_republic import ExportFormatError
from strategies import STRATEGY_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # broker exports are well under this

TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,11}$")


class BadRequest(ValueError):
    pass


def _ticker(value: Any, default: str = "SPY") -> str:
    symbol = str(value or default).strip().upper()
    if not TICKER_RE.match(symbol):
        raise BadRequest(f"Invalid ticker symbol: {symbol!r}")
    return symbol


def _clean(obj: Any) -> Any:
    """Make results JSON safe: NaN/Inf -> None, numpy/pandas scalars -> Python."""
    kind = type(obj)  # exact types first: most of a payload is plain floats, strings, dicts and lists
    if kind is float:
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if kind is str or kind is int or kind is bool or obj is None:
        return obj
    if kind is dict:
        return {str(k): _clean(v) for k, v in obj.items()}
    if kind is list or kind is tuple:
        return [_clean(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _clean(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    return obj


def _json(payload: Any, status: int = 200):
    return jsonify(_clean(payload)), status


def _body() -> dict[str, Any]:
    return request.get_json(silent=True) or {}


@app.errorhandler(BadRequest)
def _bad_request(exc: BadRequest):
    return _json({"error": str(exc)}, 400)


@app.errorhandler(LookupError)
def _not_found(exc: LookupError):
    return _json({"error": str(exc)}, 404)


@app.errorhandler(PortfolioError)
def _bad_portfolio(exc: PortfolioError):
    return _json({"error": str(exc)}, 400)


@app.errorhandler(ExportFormatError)
def _bad_export(exc: ExportFormatError):
    body = {"error": str(exc)}
    if exc.other_broker:  # the file belongs to another broker: the page offers to import it as that one
        body["other_broker"] = exc.other_broker
        body["other_broker_label"] = brokers.label(exc.other_broker)
    return _json(body, 400)


@app.errorhandler(413)
def _too_large(_exc: Exception):
    return _json({"error": "The file is larger than 10 MB"}, 413)


LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# Scripts, styles and fonts only from this app (the chart library and Inter are vendored in web/static/vendor), so a
# page load reaches no third party; data goes nowhere but back to this app.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "font-src 'self' data:; img-src 'self' data:; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


def _hostname(netloc: str) -> str:
    """``127.0.0.1:8501`` -> ``127.0.0.1``; ``[::1]:8501`` -> ``[::1]``."""
    value = (netloc or "").strip().lower()
    if value.startswith("["):
        return value[: value.find("]") + 1]
    return value.split(":", 1)[0]


@app.before_request
def _local_only():
    """Serve only this machine, and refuse changes sent by other web pages.

    The dashboard holds your portfolio and broker ledger. Binding to 127.0.0.1 keeps other machines out,
    but a web page open in the same browser can still reach it: through DNS rebinding (its own domain
    pointed at 127.0.0.1, which a Host check stops) or with a cross-site form post such as a file upload
    to the import (which an Origin check stops; browsers always send Origin on cross-origin posts). The
    origin must match this app's own host and port: a page served by another local server (a dev server on
    localhost:3000) is a different origin.
    """
    if _hostname(request.host) not in LOCAL_HOSTS:
        return _json({"error": "FastValue answers only at http://127.0.0.1 or http://localhost."}, 403)
    if request.method not in SAFE_METHODS:
        origin = request.headers.get("Origin")
        cross_site = request.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site")
        if cross_site or (origin is not None and urlsplit(origin).netloc.lower() != request.host.lower()):
            return _json({"error": "Changes are accepted only from the FastValue page itself."}, 403)
    return None


@app.after_request
def _security_headers(response):
    response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.path.startswith(("/api/", "/report/")):
        # Portfolio, ledger and track-record answers are never reused from cache; keep them out of the
        # browser's disk cache altogether (data minimisation).
        response.headers["Cache-Control"] = "no-store"
    elif request.path.startswith("/static/vendor/") or (request.path.startswith("/static/") and request.args.get("v")):
        # Vendored files carry their version in the name (lightweight-charts-4.1.1, inter-v20), and the page asks for
        # its own files as ?v=<hash of the file>: a new version is a new address, so the browser may keep the old one
        # without asking again on every load.
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    if request.path.startswith("/static/") or request.path == "/":  # the page itself is 110 KB of markup
        _gzip_static(response)
    return response


# Text assets are sent compressed when the browser accepts it. The compressed copy is made once per file version and
# kept in memory, so nothing is written next to the source and no package is needed.
_GZIP_TYPES = ("text/css", "text/javascript", "application/javascript", "application/json", "image/svg+xml", "text/html")
_gzip_cache: dict[tuple[str, int, int], bytes] = {}


def _gzip_static(response) -> None:
    if response.status_code != 200 or response.headers.get("Content-Encoding"):
        return
    if "gzip" not in request.headers.get("Accept-Encoding", "") or not response.mimetype.startswith(_GZIP_TYPES):
        return
    response.direct_passthrough = False
    body = response.get_data()
    key = (request.path, len(body), hash(body))
    packed = _gzip_cache.get(key)
    if packed is None:
        if len(_gzip_cache) > 32:  # old versions of the files after edits
            _gzip_cache.clear()
        packed = _gzip_cache[key] = gzip.compress(body, compresslevel=9, mtime=0)
    response.set_data(packed)
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Content-Length"] = str(len(packed))
    response.headers.add("Vary", "Accept-Encoding")


_asset_versions: dict[str, tuple[float, str]] = {}


@app.context_processor
def _asset_urls():
    """``static_url('app.js')`` -> ``/static/app.js?v=<sha1 of the file>``: a new version of a file is a new address."""
    def static_url(name: str) -> str:
        path = Path(app.static_folder) / name
        mtime = path.stat().st_mtime
        known = _asset_versions.get(name)
        if known is None or known[0] != mtime:
            known = _asset_versions[name] = (mtime, hashlib.sha1(path.read_bytes()).hexdigest()[:10])
        return f"/static/{name}?v={known[1]}"
    return {"static_url": static_url}


@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------ agents & config


@app.route("/api/agents")
def api_agents():
    """Registered agents plus the effective scoring config."""
    cfg = load_scoring_config()
    return _json({
        "agents": agent_metadata(),
        "config": cfg.to_dict(),
        "regimes": list(REGIMES),
        "has_local_overrides": has_local_overrides(),
        "backtestable": backtestable_agents(cfg),
        "frequencies": list(FREQUENCIES),
        "portfolio_limits": portfolio_limits(),
        "data_sources": DATA_SOURCES,
        "version": APP_VERSION,
    })


@app.route("/api/data", methods=["DELETE"])
def api_delete_all_data():
    """Delete everything FastValue keeps about the person on this computer (not cached market data)."""
    privacy.delete_all_data()
    return "", 204


@app.route("/api/config", methods=["PUT"])
def api_save_config():
    try:
        cfg = save_scoring_config(_body())
    except (ValueError, TypeError) as exc:
        raise BadRequest(str(exc)) from exc
    return _json({"config": cfg.to_dict(), "has_local_overrides": True, "backtestable": backtestable_agents(cfg)})


@app.route("/api/config/reset", methods=["POST"])
def api_reset_config():
    cfg = reset_scoring_config()
    return _json({"config": cfg.to_dict(), "has_local_overrides": False, "backtestable": backtestable_agents(cfg)})


# ------------------------------------------------------------------- analysis


def _candles(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """OHLC rows for the chart (column lists: iterrows built a Series per row and took most of the payload time)."""
    columns = (frame[k].tolist() for k in ("open", "high", "low", "close"))
    return [
        {"time": d.strftime("%Y-%m-%d"), "open": round(float(o), 2), "high": round(float(h), 2),
         "low": round(float(lo), 2), "close": round(float(c), 2)}
        for d, o, h, lo, c in zip(frame.index, *columns)
    ]


def _chart_payload(ohlcv: pd.DataFrame, bars: int = 260) -> dict[str, Any]:
    close = ohlcv["close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    pb = pb_ema_channel(ohlcv["high"], close, 200)  # on the whole history, so the EMAs have settled
    tail = ohlcv.tail(bars)
    return {
        "candles": _candles(tail),
        "volume": [{"time": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in tail["volume"].items()],
        "sma50": _series_to_chart(sma50.loc[tail.index]),
        "sma200": _series_to_chart(sma200.loc[tail.index]),
        "pb_ema_close": _series_to_chart(pb["ema_close"].loc[tail.index]),
        "pb_ema_high": _series_to_chart(pb["ema_high"].loc[tail.index]),
    }


@app.route("/api/analyze", methods=["POST"])
@app.route("/api/multi-agent", methods=["POST"])
def api_analyze():
    """Fresh multi-agent analysis for one ticker (recorded to history)."""
    body = _body()
    ticker = _ticker(body.get("ticker"), "AAPL")
    rec, ohlcv = analyze_ticker(ticker, record=bool(body.get("record", True)))
    payload = rec.to_dict()
    payload["current_price"] = round(float(ohlcv["close"].iloc[-1]), 2)
    payload["chart"] = _chart_payload(ohlcv)
    payload["history"] = get_history(ticker, limit=120)
    payload["date_range"] = {
        "start": ohlcv.index[0].strftime("%Y-%m-%d"),
        "end": ohlcv.index[-1].strftime("%Y-%m-%d"),
    }
    return _json(payload)


def _assessment_from_dict(d: dict[str, Any]) -> AgentAssessment:
    return AgentAssessment(
        agent_name=str(d.get("agent_name", "")),
        agent_key=str(d.get("agent_key", "")),
        score=float(d.get("score", 0.0)),
        signal=str(d.get("signal", "NEUTRAL")),
        confidence=float(d.get("confidence", 0.0)),
        summary="",
        data_quality=str(d.get("data_quality", "LIVE")),
        raw_data={k: (d.get("raw_data") or {}).get(k) for k in ("regime", "ev_to_fcf", "fcf", "is_financial", "excluded_sector", "sector")},
    )


@app.route("/api/rescore", methods=["POST"])
def api_rescore():
    """
    Recompute the composite for already-fetched assessments with a draft config.

    Lets the weights editor show the effect of a change instantly, without
    refetching any data or saving the config.
    """
    body = _body()
    assessments = [_assessment_from_dict(a) for a in body.get("assessments", [])]
    if not assessments:
        raise BadRequest("No assessments to rescore")
    classes = get_agent_classes()
    roles = {k: c.role for k, c in classes.items()}
    draft = body.get("config")
    try:
        if draft:
            merged = {**load_raw_config(), **draft}
            cfg = validate_config(merged, {k: c.default_weight for k, c in classes.items()}, roles)
        else:
            cfg = load_scoring_config()
    except (ValueError, TypeError) as exc:
        raise BadRequest(str(exc)) from exc
    regime = body.get("regime")
    scales = {k: c.score_scale for k, c in classes.items()}
    return _json(aggregate_layers(assessments, cfg, regime, scales, roles).to_dict())


@app.route("/api/suggest/<query>")
def api_suggest(query: str):
    """Listings Yahoo's search finds for a symbol, name or ISIN (for a ticker that has no data)."""
    from data.instruments import suggestion_report

    q = query.strip()[:40]
    if not q or not re.match(r"^[A-Za-z0-9 .&'\-^=]+$", q):
        raise BadRequest("Nothing to search for")
    return _json({"query": q, **suggestion_report(q)})


@app.route("/api/history/<ticker>")
def api_history(ticker: str):
    return _json({"ticker": _ticker(ticker), "history": get_history(_ticker(ticker), limit=500)})


@app.route("/api/scores")
def api_scores():
    """The latest owner score of every rated company analysed in the last 90 days, recomposed on the saved weights."""
    from core.score_board import score_board

    return _json(score_board())


@app.route("/api/financials/<ticker>")
def api_financials(ticker: str):
    """Annual (ten years plus LTM) or quarterly (twelve quarters) statements and the owner's metrics, from SEC EDGAR."""
    from data.sec_reports import financial_overview

    period = (request.args.get("period") or "annual").strip().lower()
    if period not in ("annual", "quarterly"):
        raise BadRequest("period must be annual or quarterly")
    return _json(financial_overview(_ticker(ticker), period))


@app.route("/api/filings/<ticker>")
def api_filings(ticker: str):
    """The latest 10-K, 10-Q, earnings releases, presentations and proxy statement of a company, with links."""
    from data.sec_reports import recent_filings

    try:
        limit = int(request.args.get("limit") or 40)
    except ValueError:
        raise BadRequest("limit must be a number") from None
    return _json(recent_filings(_ticker(ticker), max(1, min(limit, 200))))


@app.route("/api/announcements/<ticker>")
def api_announcements(ticker: str):
    """Official announcements and press releases (SEC, MFN, ASX, RNS) with the source that answered and each source's status."""
    from data.official_sources import announcements

    try:
        limit = int(request.args.get("limit") or 20)
    except ValueError:
        raise BadRequest("limit must be a number") from None
    return _json(announcements(_ticker(ticker), max(1, min(limit, 100))))


@app.route("/api/industry/<ticker>")
def api_industry(ticker: str):
    """Industry outlook links plus the latest earnings release and presentation of the company and its peers."""
    from data.industry_context import industry_context

    return _json(industry_context(_ticker(ticker)))


@app.route("/api/regime/history")
def api_regime_history():
    """The market regime since 2000: today's inputs, the weekly label and score with SPY and QQQ, stats per regime."""
    from core.regime_history import get_regime_history

    try:
        return _json(get_regime_history(refresh=request.args.get("refresh") == "1"))
    except Exception as exc:  # noqa: BLE001 - Yahoo and FRED refuse now and then; the panel offers a retry
        return _json({"error": f"The market data could not be loaded ({exc}). Try again in a minute."}, 503)


# ------------------------------------------------------------------ watchlist


@app.route("/api/watchlist", methods=["GET"])
def api_get_watchlist():
    return _json({"tickers": load_watchlist()})


@app.route("/api/watchlist", methods=["PUT"])
def api_put_watchlist():
    tickers = [_ticker(t) for t in _body().get("tickers", []) if str(t).strip()]
    if len(tickers) > 50:
        raise BadRequest("Watchlist is limited to 50 tickers")
    return _json({"tickers": save_watchlist(tickers)})


@app.route("/api/watchlist/scan", methods=["POST"])
def api_scan_watchlist():
    body = _body()
    tickers = [_ticker(t) for t in (body.get("tickers") or load_watchlist())]
    if not tickers:
        raise BadRequest("Watchlist is empty")
    if len(tickers) > 50:
        raise BadRequest("Watchlist is limited to 50 tickers")
    result = scan_watchlist(tickers)
    result["scanned_at"] = datetime.now().isoformat(timespec="seconds")
    return _json(result)


# ------------------------------------------------------------------ portfolio


@app.route("/api/portfolio", methods=["GET"])
def api_get_portfolio():
    return _json({**load_portfolio(), "limits": portfolio_limits()})


@app.route("/api/monitor")
def api_monitor():
    """The daily check: when it last ran, whether it is scheduled, and the latest alerts (core/monitor.py)."""
    return _json(monitor.status())


@app.route("/api/monitor/install", methods=["POST"])
def api_monitor_install():
    """Schedule the daily check (macOS launchd). Where there is no scheduler the page offers only "Run a check now"."""
    if not monitor.status()["schedule"].get("unsupported"):
        try:
            monitor.install()
        except Exception as exc:  # noqa: BLE001 - launchctl refused: say what, the page keeps its button
            logger.exception("Scheduling the daily check failed")
            raise BadRequest(f"The daily check could not be scheduled ({exc}).") from exc
    return _json(monitor.status())


@app.route("/api/monitor/uninstall", methods=["POST"])
def api_monitor_uninstall():
    monitor.uninstall()
    return _json(monitor.status())


@app.route("/api/monitor/run", methods=["POST"])
def api_monitor_run():
    """Run one check now, as a background job like the import (polled at /api/lab/jobs/<id>)."""
    def check(progress=None):
        progress and progress(0, 1, "Checking your holdings and watchlist")
        monitor.run_check()
        return monitor.status()

    return _json({"job": lab.start_job("monitor_check", check)})


@app.route("/api/portfolio", methods=["PUT"])
def api_put_portfolio():
    """Replace the whole portfolio (cash and positions). Imported holdings must come back unchanged unless ``force``."""
    body = _body()
    if body.get("force") is not True:
        check_imported_unchanged(body)
    return _json({**save_portfolio(body), "limits": portfolio_limits()})


@app.route("/api/portfolio/position", methods=["POST"])
def api_upsert_position():
    """Add or replace one position. The latest recorded analysis (if any) is stored as the entry snapshot.
    An imported holding is replaced only with ``force: true`` in the body (400 otherwise)."""
    body = _body()
    position = dict(body.get("position") or body)
    position["ticker"] = _ticker(position.get("ticker"))
    if position.get("owner_score_at_entry") is None:
        latest = get_latest(position["ticker"])
        if latest is not None:
            position["owner_score_at_entry"] = latest.get("owner_score") if latest.get("owner_score") is not None else latest.get("composite_score")
            position["owner_decision_at_entry"] = latest.get("decision")
            position["timing_score_at_entry"] = latest.get("timing_score")
            position["rated_at"] = latest.get("generated_at")
    return _json({**upsert_position(position, force=body.get("force") is True), "limits": portfolio_limits()})


@app.route("/api/portfolio/plan", methods=["PUT"])
def api_put_plan():
    """Your plan: the share you want in core funds and a yearly derivatives budget (null clears one).

    With the last review's ``core_share`` and ``invested_value`` in the body, the answer also carries ``review_plan``:
    the checks and drift of the saved plan, so the page need not wait for a new review.
    """
    body = _body()
    saved = set_plan(body)
    out: dict[str, Any] = {**saved, "limits": portfolio_limits()}
    share, invested = body.get("core_share"), body.get("invested_value")
    if isinstance(invested, (int, float)) and not isinstance(invested, bool) and (share is None or isinstance(share, (int, float))):
        out["review_plan"] = refreshed_plan(share, float(invested), saved["plan"])
    return _json(out)


@app.route("/api/portfolio/position/<ticker>/notes", methods=["PUT"])
def api_put_notes(ticker: str):
    """Write the notes of one position (at most 500 characters); imported positions keep their source."""
    notes = _body().get("notes")
    if not isinstance(notes, str):
        raise BadRequest("notes must be text")
    return _json({**set_notes(_ticker(ticker), notes), "limits": portfolio_limits()})


@app.route("/api/portfolio/position/<ticker>", methods=["DELETE"])
def api_remove_position(ticker: str):
    return _json({**remove_position(_ticker(ticker)), "limits": portfolio_limits()})


@app.route("/api/portfolio/review", methods=["POST"])
def api_review_portfolio():
    """Owner-first review of every held position plus portfolio-level risk."""
    body = _body()
    review = review_portfolio(record=bool(body.get("record", True)))
    try:
        save_last_review(review)
    except (OSError, TypeError, ValueError):
        pass  # the review itself is fine; only the reload shortcut is lost
    return _json(review)


@app.route("/api/portfolio/review/last")
def api_last_review():
    """The saved review while the holdings and cash are unchanged since it ran; {} otherwise."""
    return _json(load_last_review())


@app.route("/api/portfolio/performance", methods=["POST"])
def api_portfolio_performance():
    """The imported account's value and time-weighted growth over time (background job)."""
    return _json({"job": lab.start_job("account_history", track_record.account_history)})


@app.route("/api/portfolio/import", methods=["GET"])
def api_import_status():
    """What the stored broker import contains: checks, totals and each open holding's price source."""
    return _json(broker_import.import_status())


@app.route("/api/brokers")
def api_brokers():
    """The brokers whose exports can be imported (a parser each; see data/brokers.py)."""
    return _json({"brokers": brokers.describe(), "default": brokers.DEFAULT})


@app.route("/api/brokers/template")
def api_broker_template():
    """The CSV template for any broker: the columns and a few synthetic rows."""
    from flask import Response

    return Response(generic_ledger.template_csv(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=fastvalue_broker_template.csv"})


@app.route("/api/portfolio/import", methods=["POST"])
def api_import():
    """Upload a broker export. It is read in memory with the broker's parser, personal fields are dropped, and only
    the cleaned events are stored (data/store/, git-ignored). Ticker mapping runs as a background job."""
    upload = request.files.get("file")
    if upload is None:
        raise BadRequest("Choose the transaction export (CSV) to upload")
    broker = (request.form.get("broker") or brokers.DEFAULT).strip().lower()
    if broker not in brokers.BROKERS:
        raise BadRequest("Unknown broker; choose one of: " + ", ".join(brokers.BROKERS))
    parsed = brokers.parse(broker, upload.read())  # raises ExportFormatError -> 400 before any state changes
    merge = request.form.get("merge") == "1"
    return _json({"job": lab.start_job("import", broker_import.import_export, broker=broker, parsed=parsed, merge=merge)})


@app.route("/api/portfolio/import", methods=["DELETE"])
def api_delete_import():
    return _json(broker_import.delete_import())


@app.route("/api/portfolio/import/instrument/<isin>", methods=["PUT"])
def api_set_instrument(isin: str):
    """Set (or clear, with an empty ticker) the price ticker of one imported ISIN."""
    isin = isin.strip().upper()
    if not ISIN_RE.match(isin):
        raise BadRequest("Invalid ISIN")
    ticker = str(_body().get("ticker") or "").strip().upper() or None
    if ticker is not None:
        ticker = _ticker(ticker)
    try:
        return _json(broker_import.set_ticker(isin, ticker))
    except ValueError as exc:
        raise BadRequest(str(exc)) from exc


@app.route("/api/track-record", methods=["POST"])
def api_track_record():
    """Track record of the imported account (background job; polled at /api/lab/jobs/<id>)."""
    body = _body()
    method = str(body.get("method", "fifo"))
    if method not in ("fifo", "average"):
        raise BadRequest("method must be 'fifo' or 'average'")
    return _json({"job": lab.start_job("track_record", track_record.run_track_record, method=method,
                                       fastvalue=bool(body.get("fastvalue")))})


@app.route("/report/track-record")
def report_track_record():
    """The full HTML report written by the last track-record run (local file, never committed)."""
    path = track_record.report_path()
    if not path.exists():
        raise LookupError("No report yet: run My track record in the Backtest tab first")
    return send_file(path, mimetype="text/html", max_age=0)


@app.route("/api/track-record/rules", methods=["POST"])
def api_track_record_rules():
    """Replay the imported history under the preset rules and up to six rules of your own (background job)."""
    body = _body()
    method = str(body.get("method", "fifo"))
    if method not in ("fifo", "average"):
        raise BadRequest("method must be 'fifo' or 'average'")
    custom = body.get("custom") or []
    if not isinstance(custom, list) or len(custom) > history_lab.MAX_CUSTOM:
        raise BadRequest(f"custom must be a list of at most {history_lab.MAX_CUSTOM} rules")
    try:
        rules = [history_lab.rules_from(c if isinstance(c, dict) else {}, key=f"custom_{i + 1}") for i, c in enumerate(custom)]
    except ValueError as exc:
        raise BadRequest(str(exc)) from exc
    return _json({"job": lab.start_job("rules_lab", track_record.run_rules, rules=rules,
                                       include_presets=bool(body.get("presets", True)), method=method)})


@app.route("/api/track-record/last")
def api_track_record_last():
    """The stored result of the last run, if there is one (the same local file the share page is written from)."""
    return _json(track_record.load_result() or {})


@app.route("/api/track-record/share", methods=["POST"])
def api_track_record_share():
    """Write the shareable version of the last track record: percentages and counts only, names optional."""
    result = track_record.load_result()
    if not result:
        raise LookupError("Run My track record in the Backtest tab first")
    names = bool(_body().get("names"))
    style_report.write_share_report(result, names=names)
    return _json({"url": "/report/my-style", "names": names})


@app.route("/report/my-style")
def report_my_style():
    """The shareable style report written by /api/track-record/share (local file, never committed)."""
    path = style_report.share_report_path()
    if not path.exists():
        raise LookupError("No shareable report yet: open My track record and click Share")
    if request.args.get("download") == "1":  # a file to send: the address itself works only on this computer
        return send_file(path, mimetype="text/html", max_age=0, as_attachment=True, download_name="my-investing-style.html")
    return send_file(path, mimetype="text/html", max_age=0)


# ------------------------------------------------------------------ settings


@app.route("/api/settings/sec", methods=["GET"])
def api_sec_settings():
    return _json({"configured": has_sec_access(), "source": sec_access_source()})


@app.route("/api/settings/sec", methods=["PUT"])
def api_save_sec_settings():
    """Save the contact email the SEC asks for (stored only on this machine, sent only to sec.gov)."""
    try:
        source = save_contact(str(_body().get("email") or ""))
    except ValueError as exc:
        raise BadRequest(str(exc)) from exc
    return _json({"configured": source is not None, "source": source})


@app.route("/api/portfolio/candidate", methods=["POST"])
def api_candidate():
    """Should I enter this ticker (or, if held, what is the position's risk)?"""
    body = _body()
    ticker = _ticker(body.get("ticker"), "AAPL")
    result = assess_candidate(ticker, record=bool(body.get("record", True)))
    ohlcv = fetch_ohlcv([ticker]).get(ticker)
    if ohlcv is not None:
        result["chart"] = _chart_payload(ohlcv, bars=130)
    return _json(result)


# ---------------------------------------------------------- score backtesting


# ------------------------------------------------------------------- backtest lab


def _lab_inputs(body: dict[str, Any]) -> dict[str, Any]:
    """Universe, period and trading rules shared by a lab run and a setup search."""
    kind = str(body.get("universe", "ticker"))
    if kind not in ("ticker", "portfolio"):
        raise BadRequest("universe must be 'ticker' or 'portfolio'")
    allocation = str(body.get("allocation", "current"))
    if allocation not in ("current", "equal"):
        raise BadRequest("allocation must be 'current' or 'equal'")
    universe = {"kind": kind, "allocation": allocation}
    if kind == "ticker":
        universe["ticker"] = _ticker(body.get("ticker"), "SPY")
    years = body.get("years")
    if years not in (None, "", 0):
        try:
            years = float(years)
        except (TypeError, ValueError) as exc:
            raise BadRequest("years must be a number") from exc
        if not 1 <= years <= 20:
            raise BadRequest("years must be between 1 and 20")
    else:
        years = None
    try:
        rules = lab.Rules(
            entry_gate=str(body.get("entry_gate", "none")),
            start=str(body.get("start", "invested")),
            rebalance=str(body.get("rebalance", "yearly")),
            cost_bps=float(body.get("cost_bps", lab.COST_BPS)),
        ).validate()
    except (TypeError, ValueError) as exc:
        raise BadRequest(str(exc)) from exc
    return {"universe": universe, "years": years, "rules": rules}


@app.route("/api/lab/options")
def api_lab_options():
    return _json({
        "grid": lab.GRID,
        "grid_order": list(lab.GRID),  # JSON objects arrive with sorted keys; this keeps the intended order
        "entry_gates": lab.ENTRY_GATES,
        "entry_gate_order": list(lab.ENTRY_GATES),
        "rebalance": list(lab.REBALANCE),
        "classic": lab.CLASSIC_LABELS,
        "max_setups": lab.MAX_SETUPS,
        "start": lab.PANEL_START.strftime("%Y-%m-%d"),
        "sec_access": has_sec_access(),
        "sec_source": sec_access_source(),
        "portfolio_tickers": [p["ticker"] for p in load_portfolio().get("positions", [])],
        "has_import": bool(trade_republic.load_ledger()),  # what import_status() reports, without the replay
        "setup": lab.describe_setup(load_scoring_config()),
        "replayed_agents": list(lab.AGENT_COLUMNS),
    })


@app.route("/api/lab/run", methods=["POST"])
def api_lab_run():
    body = _body()
    inputs = _lab_inputs(body)
    compare = body.get("compare")
    if compare is not None and not isinstance(compare, list):
        raise BadRequest("compare must be a list")
    return _json({"job": lab.start_job("run", lab.run_lab, compare=compare, **inputs)})


@app.route("/api/lab/search", methods=["POST"])
def api_lab_search():
    body = _body()
    inputs = _lab_inputs(body)
    try:
        grid = lab.validate_grid(body.get("grid"))
    except ValueError as exc:
        raise BadRequest(str(exc)) from exc
    return _json({"job": lab.start_job("search", lab.run_setup_search, grid=grid, **inputs)})


@app.route("/api/lab/index-check", methods=["POST"])
def api_lab_index_check():
    """Replay one search row and the saved setup on the S&P 500 panel (core/index_check.py)."""
    body = _body()
    combo = body.get("combo")
    if not isinstance(combo, dict):
        raise BadRequest("combo must be an object")
    allowed = set(lab.GRID)
    if set(combo) - allowed:
        raise BadRequest("Unknown setting: " + ", ".join(sorted(set(combo) - allowed)))
    rules = _lab_inputs(body)["rules"]
    return _json({"job": lab.start_job("index", index_check.check_setup, combo=combo, rules=rules)})


@app.route("/api/lab/jobs/<job_id>")
def api_lab_job(job_id: str):
    job = lab.get_job(job_id)
    if job is None:
        raise LookupError("Unknown or expired backtest job")
    return _json(job)


@app.route("/api/lab/jobs/<job_id>/cancel", methods=["POST"])
def api_lab_job_cancel(job_id: str):
    """Stop a running backtest job; nothing it computed is kept."""
    if not lab.cancel_job(job_id):
        raise LookupError("Unknown or expired backtest job")
    return _json({"job": job_id, "status": "cancelled"})


# ------------------------------------------------------- trend and dip rules (core/systematic.py, D-48)


@app.route("/api/systematic/options")
def api_systematic_options():
    return _json(systematic.options())


def _optional_ticker(value: Any) -> str | None:
    return _ticker(value) if str(value or "").strip() else None


def _number(body: dict, key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(body.get(key, default) if body.get(key) is not None else default)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"{key} must be a number") from exc
    if not low <= value <= high:
        raise BadRequest(f"{key} must be between {low:g} and {high:g}")
    return value


@app.route("/api/systematic/run", methods=["POST"])
def api_systematic_run():
    """Start a trend, savings or rotation study as a background job; poll /api/lab/jobs/<id>."""
    body = _body()
    study = str(body.get("study", "rule"))
    start = str(body.get("start") or "").strip() or None
    if start and start != "vehicle" and not re.match(r"^\d{4}-\d{2}-\d{2}$", start):
        raise BadRequest("start must be YYYY-MM-DD")
    market = {
        "signal": _ticker(body.get("signal"), "QQQ"),
        "leverage": int(_number(body, "leverage", 1, 1, systematic.MAX_LEVERAGE)),
        "vehicle": _optional_ticker(body.get("vehicle")),
        "extend": bool(body.get("extend")),
        "fee": _number(body, "fee", systematic.SYNTHETIC_FEE, 0.0, 0.05),
        "spread": _number(body, "spread", systematic.SYNTHETIC_SPREAD, 0.0, 0.05),
    }
    if study == "rule":
        rule = str(body.get("rule", "two_engine"))
        params = body.get("params") or {}
        if not isinstance(params, dict):
            raise BadRequest("params must be an object")
        try:
            params = systematic.validate_params(rule, params)
        except ValueError as exc:
            raise BadRequest(str(exc)) from exc
        execution = str(body.get("execution", "next_close"))
        if execution not in systematic.EXECUTION_LAG:
            raise BadRequest("Unknown execution")
        kwargs = {**market, "rule": rule, "params": params, "execution": execution, "start": start,
                  "cost_bps": _number(body, "cost_bps", systematic.COST_BPS, 0, 200)}
        return _json({"job": lab.start_job("systematic", systematic.run_rule, **kwargs)})
    if study == "savings":
        raw = body.get("thresholds") or list(systematic.SAVINGS_THRESHOLDS)
        try:
            thresholds = tuple(float(t) for t in raw)[:8]
        except (TypeError, ValueError) as exc:
            raise BadRequest("thresholds must be numbers") from exc
        return _json({"job": lab.start_job("savings", systematic.run_savings, **market, thresholds=thresholds,
                                           start=None if start == "vehicle" else start)})
    if study == "rotation":
        tickers = body.get("tickers")
        if tickers is not None:
            if not isinstance(tickers, list) or len(tickers) > 40:
                raise BadRequest("tickers must be a list of at most 40 symbols")
            tickers = [_ticker(t) for t in tickers]
        kwargs = {"universe": str(body.get("universe", "sectors")), "tickers": tickers or None,
                  "top": int(_number(body, "top", 3, 1, 20)), "review": str(body.get("review", "quarterly")),
                  "weighting": str(body.get("weighting", "score")), "buffer": bool(body.get("buffer", True)),
                  "adhoc": bool(body.get("adhoc", True)), "trend_filter": bool(body.get("trend_filter")),
                  "cost_bps": _number(body, "cost_bps", systematic.COST_BPS, 0, 200),
                  "start": None if start == "vehicle" else start}
        return _json({"job": lab.start_job("rotation", systematic.run_rotation, **kwargs)})
    raise BadRequest("study must be rule, savings or rotation")


@app.route("/api/systematic/check")
def api_systematic_check():
    """Today's state of the alert rules on the watched markets (what the daily check alerts on)."""
    return _json(systematic.trend_check())


@app.route("/api/score-backtest", methods=["POST"])
def api_score_backtest():
    body = _body()
    ticker = _ticker(body.get("ticker"), "SPY")
    data = fetch_ohlcv([ticker])
    if ticker not in data:
        raise LookupError(f"No data available for {ticker}")
    try:
        result = run_score_backtest(
            ticker,
            data[ticker],
            agent_keys=body.get("agents") if "agents" in body else None,
            frequency=str(body.get("frequency", "weekly")),
            entry_threshold=float(body["entry_threshold"]) if body.get("entry_threshold") is not None else None,
            exit_threshold=float(body.get("exit_threshold", 0.0)),
            forward_days=int(body.get("forward_days", 20)),
            years=float(body["years"]) if body.get("years") else None,
        )
    except (ValueError, TypeError) as exc:
        raise BadRequest(str(exc)) from exc
    return _json(result)


# ------------------------------------------------------- strategy backtesting


@app.route("/api/strategies")
def api_strategies():
    return _json({
        name: {"label": lab.CLASSIC_LABELS.get(name, name.replace("_", " ").capitalize()), "description": e["description"],
               "params": e.get("params", {}), "defaults": e["defaults"]}
        for name, e in STRATEGY_REGISTRY.items()
    })


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    body = _body()
    ticker = _ticker(body.get("ticker"), "SPY")
    strategy_name = body.get("strategy", "sma_crossover")
    params = body.get("params", {}) or {}

    if strategy_name not in STRATEGY_REGISTRY:
        raise BadRequest(f"Unknown strategy: {strategy_name}")

    data = fetch_ohlcv([ticker])
    if ticker not in data:
        raise LookupError(f"No data available for {ticker}")
    ohlcv = data[ticker]

    entry = STRATEGY_REGISTRY[strategy_name]
    merged = dict(entry["defaults"])
    specs = entry.get("params", {})
    for key, value in params.items():
        if key not in specs:
            continue
        ptype = specs[key]["type"]
        try:
            merged[key] = int(value) if ptype == "int" else float(value) if ptype == "float" else value
        except (TypeError, ValueError) as exc:
            raise BadRequest(f"Invalid value for {key}: {value!r}") from exc

    try:
        strategy = entry["class"](**merged)
    except (ValueError, TypeError) as exc:
        raise BadRequest(f"Invalid strategy parameters: {exc}") from exc

    signals = strategy.generate_signals(ohlcv)
    result = run_backtest(ohlcv=ohlcv, signals=signals, commission_per_share=COMMISSION_PER_SHARE, init_cash=INITIAL_CASH)
    strategy_stats = compute_metrics(result.returns, signals=signals)
    benchmark_returns = compute_buy_and_hold_returns(ohlcv).reindex(result.returns.index).fillna(0.0)
    benchmark_stats = compute_metrics(benchmark_returns)

    equity = (1 + result.returns).cumprod() * INITIAL_CASH
    benchmark_equity = (1 + benchmark_returns).cumprod() * INITIAL_CASH

    return _json({
        "ticker": ticker,
        "strategy": strategy_name,
        "strategy_stats": strategy_stats,
        "benchmark_stats": benchmark_stats,
        "chart": _build_chart_data(ohlcv, signals, strategy_name, merged),
        "equity": _series_to_chart(equity, 2),
        "benchmark_equity": _series_to_chart(benchmark_equity, 2),
        "date_range": {"start": ohlcv.index[0].strftime("%Y-%m-%d"), "end": ohlcv.index[-1].strftime("%Y-%m-%d")},
    })


@app.route("/api/compare", methods=["POST"])
def api_compare():
    body = _body()
    ticker = _ticker(body.get("ticker"), "SPY")
    data = fetch_ohlcv([ticker])
    if ticker not in data:
        raise LookupError(f"No data available for {ticker}")
    comp = compare_strategies(ticker=ticker, strategy_names=body.get("strategies"), ohlcv=data[ticker])
    return _json({
        "ticker": comp.ticker,
        "benchmark_name": comp.benchmark_name,
        "benchmark_stats": comp.benchmark_stats,
        "rankings": comp.rankings,
        "strategies": [
            {"strategy": s["strategy"], "display_name": s["display_name"], "stats": s["stats"], "equity": s["equity"]}
            for s in comp.strategy_results
        ],
        "date_range": comp.date_range,
    })


def _build_chart_data(ohlcv: pd.DataFrame, signals: pd.DataFrame, strategy_name: str, params: dict) -> dict:
    """Price candles, trade markers and indicator overlays for the strategy view."""
    candles = _candles(ohlcv)
    entries = signals["entries"].reindex(ohlcv.index).fillna(False)
    exits = signals["exits"].reindex(ohlcv.index).fillna(False)
    buy_signals = [{"time": d.strftime("%Y-%m-%d"), "position": "belowBar", "shape": "arrowUp", "text": "In"}
                   for d in ohlcv.index[entries.values]]
    sell_signals = [{"time": d.strftime("%Y-%m-%d"), "position": "aboveBar", "shape": "arrowDown", "text": "Out"}
                    for d in ohlcv.index[exits.values]]
    return {
        "candles": candles,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "overlays": _compute_overlays(ohlcv, strategy_name, params),
    }


def _compute_overlays(ohlcv: pd.DataFrame, strategy_name: str, params: dict) -> dict:
    """Indicator overlay lines and sub-plots for the strategy chart."""
    close = ohlcv["close"]
    overlays: dict = {"lines": [], "bands": [], "subplots": []}

    if strategy_name == "sma_crossover":
        fast_w, slow_w = params.get("fast_window", 50), params.get("slow_window", 200)
        overlays["lines"] += [
            {"name": f"SMA {fast_w}", "color": "series-1", "data": _series_to_chart(close.rolling(fast_w).mean())},
            {"name": f"SMA {slow_w}", "color": "series-2", "data": _series_to_chart(close.rolling(slow_w).mean())},
        ]
    elif strategy_name == "rsi":
        period = params.get("period", 14)
        delta = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, float("nan"))))
        overlays["subplots"].append({
            "name": "RSI",
            "data": _series_to_chart(rsi),
            "levels": [params.get("oversold", 30), params.get("overbought", 70)],
        })
    elif strategy_name == "macd":
        macd_line = (close.ewm(span=params.get("fast_period", 12), adjust=False).mean()
                     - close.ewm(span=params.get("slow_period", 26), adjust=False).mean())
        signal_line = macd_line.ewm(span=params.get("signal_period", 9), adjust=False).mean()
        overlays["subplots"].append({
            "name": "MACD",
            "data": _series_to_chart(macd_line),
            "signal_data": _series_to_chart(signal_line),
            "histogram": _series_to_chart(macd_line - signal_line),
        })
    elif strategy_name == "bollinger_bands":
        window, num_std = params.get("window", 20), params.get("num_std", 2.0)
        middle = close.rolling(window).mean()
        std = close.rolling(window).std()
        overlays["lines"] += [
            {"name": "Upper band", "color": "series-2", "data": _series_to_chart(middle + num_std * std)},
            {"name": "Middle band", "color": "series-3", "data": _series_to_chart(middle)},
            {"name": "Lower band", "color": "series-1", "data": _series_to_chart(middle - num_std * std)},
        ]
    elif strategy_name == "momentum":
        roc = close.pct_change(periods=params.get("lookback", 20)) * 100
        overlays["subplots"].append({
            "name": "Momentum (ROC %)",
            "data": _series_to_chart(roc),
            "levels": [params.get("entry_threshold", 0.05) * 100, params.get("exit_threshold", -0.02) * 100],
        })
    return overlays


def _series_to_chart(s: pd.Series, digits: int = 4) -> list[dict]:
    return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), digits)} for d, v in s.items() if pd.notna(v)]


if __name__ == "__main__":
    # QUANT_* are the names used before the rename to FastValue.
    debug = (os.environ.get("FASTVALUE_DEBUG") or os.environ.get("QUANT_DEBUG")) == "1"
    port = int(os.environ.get("FASTVALUE_PORT") or os.environ.get("QUANT_PORT") or "8501")
    print(f"\n  FastValue dashboard: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=debug, threaded=True)
