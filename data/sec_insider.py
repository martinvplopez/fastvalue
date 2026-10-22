"""Point-in-time insider transactions from the SEC Form 3/4/5 data sets.

The SEC publishes every Form 3, 4 and 5 since 2006 as quarterly bundles of
tab-separated files (Insider Transactions Data Sets). Each filing carries its
filing date, so a backtest can use exactly the trades that were public on a
given day. ``to_yahoo_frame`` turns one issuer's rows into the shape that
``data.insider_trading.analyze_insider_transactions`` reads, so the Insider
agent runs unchanged on this history.

Each quarter is about 14 MB zipped. ``load_insider_history`` downloads a
quarter, keeps only non-derivative transactions of the requested issuers,
caches that extract as Parquet under data/cache/sec_insider/ and deletes the
zip. Requests go through the SEC User-Agent and rate limit of ``data.sec_edgar``.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

import config.settings as settings
from data.sec_edgar import SecDataError, user_agent

logger = logging.getLogger(__name__)

DATASET_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{quarter}_form345.zip"
FORMS = frozenset({"4", "4/A", "5", "5/A"})
EXTRACT_VERSION = 1

SUBMISSION_COLS = ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"]
OWNER_COLS = ["ACCESSION_NUMBER", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]
TRANS_COLS = ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
              "TRANS_ACQUIRED_DISP_CD"]


def _cache_dir() -> Path:
    path = settings.DATA_DIR / "sec_insider"
    path.mkdir(parents=True, exist_ok=True)
    return path


def quarters(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    out, p = [], pd.Period(start, freq="Q")
    while p <= pd.Period(end, freq="Q"):
        out.append(f"{p.year}q{p.quarter}")
        p += 1
    return out


def _download(quarter: str, retries: int = 3) -> bytes:
    url = DATASET_URL.format(quarter=quarter)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent()})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise SecDataError(f"Insider data set {quarter} is not published yet") from exc
            if attempt == retries - 1:
                raise SecDataError(f"Insider data set {quarter} failed ({exc.code})") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise SecDataError(f"Insider data set {quarter} failed: {exc}") from exc
        time.sleep(5 * (attempt + 1))
    raise SecDataError(f"Insider data set {quarter} failed")


def _dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%d-%b-%Y", errors="coerce")
    missing = parsed.isna() & values.notna()
    if missing.any():
        parsed[missing] = pd.to_datetime(values[missing], errors="coerce")
    return parsed


def parse_quarter(blob: bytes, issuer_ciks: set[int]) -> pd.DataFrame:
    """Non-derivative Form 4/5 transactions of ``issuer_ciks`` from one quarterly zip."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = {Path(n).name.upper(): n for n in zf.namelist()}

        def read(name: str, cols: list[str]) -> pd.DataFrame:
            with zf.open(names[name]) as fh:
                return pd.read_csv(fh, sep="\t", dtype=str, usecols=lambda c: c in cols, quoting=3,
                                   on_bad_lines="skip", encoding_errors="replace")

        subs = read("SUBMISSION.TSV", SUBMISSION_COLS)
        subs["ISSUERCIK"] = pd.to_numeric(subs["ISSUERCIK"], errors="coerce")
        subs = subs[subs["ISSUERCIK"].isin(issuer_ciks) & subs["DOCUMENT_TYPE"].isin(FORMS)]
        if subs.empty:
            return pd.DataFrame()
        owners = read("REPORTINGOWNER.TSV", OWNER_COLS).drop_duplicates("ACCESSION_NUMBER")  # first owner of joint filings
        trans = read("NONDERIV_TRANS.TSV", TRANS_COLS)
    df = trans.merge(subs, on="ACCESSION_NUMBER").merge(owners, on="ACCESSION_NUMBER", how="left")
    return pd.DataFrame({
        "issuer_cik": df["ISSUERCIK"].astype("int64"),
        "accession": df["ACCESSION_NUMBER"],
        "filed": _dates(df["FILING_DATE"]),
        "trade_date": _dates(df["TRANS_DATE"]),
        "code": df["TRANS_CODE"].fillna("").str.strip().str.upper(),
        "shares": pd.to_numeric(df["TRANS_SHARES"], errors="coerce"),
        "price": pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce"),
        "acquired": df["TRANS_ACQUIRED_DISP_CD"].fillna("").str.strip().str.upper(),
        "owner": df["RPTOWNERNAME"].fillna("Unknown insider"),
        "relationship": df["RPTOWNER_RELATIONSHIP"].fillna(""),
        "title": df["RPTOWNER_TITLE"].fillna(""),
    }).dropna(subset=["filed", "trade_date"])


def load_insider_history(issuer_ciks: set[int], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """All Form 4/5 non-derivative transactions of the issuers, filed between ``start`` and ``end``."""
    tag = f"{len(issuer_ciks)}_{hash(frozenset(issuer_ciks)) & 0xFFFFFFFF:08x}"
    frames = []
    for q in quarters(start, end):
        path = _cache_dir() / f"{q}_v{EXTRACT_VERSION}_{tag}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        try:
            blob = _download(q)
        except SecDataError as exc:
            logger.warning("%s", exc)
            continue
        part = parse_quarter(blob, issuer_ciks)
        part.to_parquet(path, index=False)
        frames.append(part)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["issuer_cik", "accession", "filed", "trade_date", "code", "shares", "price",
                                     "acquired", "owner", "relationship", "title"])
    return pd.concat(frames, ignore_index=True).sort_values("filed", kind="stable").reset_index(drop=True)


def _position(relationship: str, title: str) -> str:
    """Yahoo-style position text: officer title first, then director or 10% owner."""
    title = (title or "").strip()
    rel = (relationship or "").lower()
    if title:
        return title
    if "director" in rel:
        return "Director"
    if "tenpercentowner" in rel.replace(" ", "") or "10%" in rel:
        return "Beneficial Owner of more than 10% of a Class of Security"
    if "officer" in rel:
        return "Officer"
    return "Insider"


def to_yahoo_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows of one issuer shaped like ``yfinance.Ticker(...).insider_transactions``."""
    shares = rows["shares"].abs()
    value = (shares * rows["price"]).where(rows["price"] > 0)
    return pd.DataFrame({
        "Shares": shares.values,
        "Value": value.values,
        "URL": "",
        "Text": "",
        "Insider": rows["owner"].values,
        "Position": [_position(r, t) for r, t in zip(rows["relationship"], rows["title"])],
        "Transaction": rows["code"].values,
        "Start Date": rows["trade_date"].values,
    })
