"""Delete everything FastValue keeps about the person on this computer ("Delete all my FastValue data").

What goes: the portfolio and the broker ledger, the ticker map, the watchlist, saved weights, the SEC contact, the
analysis history, the rules-lab history, the daily check (its schedule, state, alerts and log), the last review and
the pages written under the output folder (style page, track record). What stays: market data only
(``STORE_DIR/iv_history`` and the price cache in ``data/cache``), which holds nothing about the person.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import config.settings as settings
from config.weights import _local_path as weights_path
from core import monitor, style_report, track_record
from core.history_lab import HISTORY_FILE
from data import history, instruments, portfolio, trade_republic
from data.sec_edgar import CONTACT_FILE

logger = logging.getLogger(__name__)

# STORE_DIR files that are not reached through a module's own path function.
_STORE_FILES = (CONTACT_FILE, HISTORY_FILE, "monitor.log")


def _paths() -> list[Path]:
    db = history._db_path()
    paths = [
        portfolio._path(), trade_republic._path(), instruments._path(), history._watchlist_path(), weights_path(),
        db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm"),  # SQLite side files
        monitor._state_path(), monitor._alerts_path(), portfolio.last_review_path(),
        *(settings.STORE_DIR / name for name in _STORE_FILES),
        style_report.share_report_path(), track_record.report_path(), track_record.result_path(),
    ]
    return paths


def delete_all_data() -> dict[str, Any]:
    """Remove the daily check's schedule and every file above. Returns how many files went and whether a schedule did."""
    scheduled = bool(monitor.schedule_status().get("installed"))
    if scheduled:
        monitor.uninstall()
    removed = 0
    for path in _paths():
        if path.exists():
            path.unlink()
            removed += 1
    logger.info("Deleted %d local files%s", removed, " and the daily check schedule" if scheduled else "")
    return {"removed": removed, "schedule_removed": scheduled}
