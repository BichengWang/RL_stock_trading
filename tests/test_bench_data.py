"""Offline tests for benchmark data loading (network and Yahoo are mocked)."""

import hashlib
import io
import zipfile
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rl.bench import data as bench_data
from rl.bench.data import load_task_data, parse_klines
from rl.bench.tasks import Task

DAY_MS = 86_400_000


def make_task(source, tickers, start, test_end="2100-01-01"):
    day = pd.Timestamp(start)
    return Task(
        id="t", title="t", source=source, tickers=tuple(tickers), start=start,
        validation_start=str((day + pd.Timedelta(days=1)).date()),
        test_start=str((day + pd.Timedelta(days=2)).date()), test_end=test_end,
        transaction_cost=0.001, periods_per_year=365, description="",
    )


def epoch_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)


def kline_zip(days, price=100.0, *, micro=False, header=False, name="x.csv") -> bytes:
    rows = []
    if header:
        rows.append(",".join(bench_data.KLINE_COLUMNS))
    for i, day in enumerate(days):
        opened = epoch_ms(day) * (1000 if micro else 1)
        closed = opened + (DAY_MS * (1000 if micro else 1)) - 1
        p = price + i
        rows.append("{},{},{},{},{},5,{},{},7,1,1,0".format(opened, p, p + 1, p - 1, p + 0.5, closed, p * 5))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, "\n".join(rows) + "\n")
    return buffer.getvalue()


def month_days(year, month, count=None):
    day = date(year, month, 1)
    days = []
    while day.month == month and (count is None or len(days) < count):
        days.append(day)
        day += timedelta(days=1)
    return days


class Archive:
    """A fake data.binance.vision keyed by URL; unknown URLs are 404."""

    def __init__(self):
        self.files = {}
        self.requests = []

    def add(self, symbol, kind, label, payload, checksum=None):
        url = "{}/{}/klines/{}/1d/{}-1d-{}.zip".format(bench_data.BINANCE_BASE, kind, symbol, symbol, label)
        digest = checksum or hashlib.sha256(payload).hexdigest()
        self.files[url] = payload
        self.files[url + ".CHECKSUM"] = "{}  {}-1d-{}.zip\n".format(digest, symbol, label).encode()

    def get(self, url):
        self.requests.append(url)
        return self.files.get(url)


@pytest.fixture
def archive():
    fake = Archive()
    with patch.object(bench_data, "_http_get", fake.get):
        yield fake


def crypto_archive(archive):
    archive.add("AAAUSDT", "monthly", "2024-01", kline_zip(month_days(2024, 1)))
    archive.add("AAAUSDT", "monthly", "2024-02", kline_zip(month_days(2024, 2), price=200))
    # OLDUSDT is delisted after January; February's file does not exist.
    archive.add("OLDUSDT", "monthly", "2024-01", kline_zip(month_days(2024, 1, 20), header=True))
    return make_task("binance", ["OLDUSDT", "AAAUSDT"], "2024-01-01")


def test_checksum_mismatch_raises(archive, tmp_path):
    archive.add("AAAUSDT", "monthly", "2024-01", kline_zip(month_days(2024, 1)), checksum="0" * 64)
    task = make_task("binance", ["AAAUSDT"], "2024-01-01")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_task_data(task, end="2024-02-01", cache_dir=tmp_path)
    assert not list(tmp_path.rglob("*.zip")), "unverified downloads must not be cached"


def test_millisecond_and_microsecond_timestamps_give_the_same_dates():
    days = month_days(2025, 1, 5)
    milli, micro = kline_zip(days), kline_zip(days, micro=True)
    parsed = parse_klines(micro)
    pd.testing.assert_frame_equal(parse_klines(milli), parsed)
    assert parsed["date"].tolist() == [pd.Timestamp(d) for d in days]


def test_header_row_is_ignored_and_quote_volume_is_volume():
    days = month_days(2024, 1, 3)
    plain = parse_klines(kline_zip(days))
    pd.testing.assert_frame_equal(parse_klines(kline_zip(days, header=True)), plain)
    assert plain["volume"].tolist() == [500.0, 505.0, 510.0]  # quote volume = price * 5


def test_delisted_symbol_is_nan_afterwards_and_columns_follow_task(archive, tmp_path):
    task = crypto_archive(archive)
    panel = load_task_data(task, end="2024-03-01", cache_dir=tmp_path)
    assert panel.tickers == ("OLDUSDT", "AAAUSDT")
    assert len(panel) == 31 + 29
    old = panel.close[:, 0]
    assert np.isfinite(old[:20]).all() and np.isnan(old[20:]).all()
    assert np.isfinite(panel.close[:, 1]).all()
    assert str(panel.dates[0]) == "2024-01-01" and str(panel.dates[-1]) == "2024-02-29"
    assert panel.volume[0, 1] == 500.0


def test_end_is_exclusive_and_start_filters(archive, tmp_path):
    task = make_task("binance", ["AAAUSDT"], "2024-01-05")
    crypto_archive(archive)
    panel = load_task_data(task, end="2024-02-10", cache_dir=tmp_path)
    assert str(panel.dates[0]) == "2024-01-05"
    assert str(panel.dates[-1]) == "2024-02-09"


def test_verified_cache_is_reused_offline_and_corruption_is_caught(archive, tmp_path):
    task = crypto_archive(archive)
    first = load_task_data(task, end="2024-03-01", cache_dir=tmp_path)
    archive.requests.clear()
    with patch.object(bench_data, "_http_get", side_effect=AssertionError("network used")):
        again = load_task_data(task, end="2024-03-01", cache_dir=tmp_path)
    assert again.sha256() == first.sha256()
    archive.requests.clear()
    load_task_data(task, end="2024-03-01", cache_dir=tmp_path, refresh=True)
    assert archive.requests, "refresh must re-download"
    next(tmp_path.glob("binance/AAAUSDT/*2024-01.zip")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_task_data(task, end="2024-03-01", cache_dir=tmp_path)


def test_recent_month_without_monthly_archive_uses_daily_files(archive, tmp_path):
    today = datetime.now(timezone.utc).date()
    month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    days = month_days(month.year, month.month, 4)
    for i, day in enumerate(days):
        archive.add("AAAUSDT", "daily", day.isoformat(), kline_zip([day], price=100 + i))
    task = make_task("binance", ["AAAUSDT"], month.isoformat())
    panel = load_task_data(task, end=(month + timedelta(days=3)).isoformat(), cache_dir=tmp_path)
    assert [str(d) for d in panel.dates] == [d.isoformat() for d in days[:3]]
    assert panel.open[:, 0].tolist() == [100.0, 101.0, 102.0]


def test_http_errors_other_than_404_raise(tmp_path):
    import urllib.error

    def forbidden(url):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    task = make_task("binance", ["AAAUSDT"], "2024-01-01")
    with patch.object(bench_data, "_http_get", forbidden):
        with pytest.raises(urllib.error.HTTPError):
            load_task_data(task, end="2024-02-01", cache_dir=tmp_path)


def test_synthetic_is_deterministic_and_ordered_as_task():
    task = make_task("synthetic", ["ZZZ", "AAA", "MMM"], "2020-01-01")
    one = load_task_data(task, end="2020-03-01")
    two = load_task_data(task, end="2020-03-01")
    assert one.tickers == ("ZZZ", "AAA", "MMM")
    assert one.sha256() == two.sha256()
    assert str(one.dates[0]) == "2020-01-01" and one.dates[-1] < np.datetime64("2020-03-01")
    assert (one.high >= np.maximum(one.open, one.close)).all()
    longer = load_task_data(task, end="2020-04-01")
    assert np.array_equal(longer.rows(0, len(one)).close, one.close)


def yahoo_frame(first="2024-01-02", rows=4, split=2.0):
    index = pd.bdate_range(first, periods=rows, name="Date")
    price = np.arange(rows, dtype=float) + 100
    return pd.DataFrame(
        {
            "Open": price * split, "High": (price + 1) * split, "Low": (price - 1) * split,
            "Close": (price + 0.5) * split, "Adj Close": price + 0.5, "Volume": 1000.0 + np.arange(rows),
        },
        index=index,
    )


def test_yahoo_uses_adjusted_prices_and_caches_per_ticker(tmp_path):
    task = make_task("yahoo", ["BBB", "AAA"], "2024-01-01")
    with patch("rl.marketdata.yahoodownloader.yf.download", side_effect=lambda *a, **k: yahoo_frame()) as download:
        panel = load_task_data(task, end="2024-01-10", cache_dir=tmp_path)
        assert download.call_count == 2
        assert panel.tickers == ("BBB", "AAA")
        # open/high/low are scaled by Adj Close / Close like the close itself
        assert panel.open[0, 0] == pytest.approx(100.0)
        assert panel.close[0, 0] == 100.5
        first_hash = panel.sha256()

        assert load_task_data(task, end="2024-01-10", cache_dir=tmp_path).sha256() == first_hash
        assert download.call_count == 2, "covered range is served from the cache"

        load_task_data(task, end="2024-01-10", cache_dir=tmp_path, refresh=True)
        assert download.call_count == 4

        load_task_data(task, end="2024-02-10", cache_dir=tmp_path)
        assert download.call_count == 6, "a cache that ends before the requested end is refreshed"
    assert sorted(p.name for p in (tmp_path / "yahoo").iterdir()) == ["AAA.csv", "AAA.json", "BBB.csv", "BBB.json"]


def test_yahoo_ticker_without_data_raises(tmp_path):
    task = make_task("yahoo", ["AAA"], "2024-01-01")
    with patch("rl.marketdata.yahoodownloader.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="no data for AAA"):
            load_task_data(task, end="2024-01-10", cache_dir=tmp_path)


def test_yahoo_late_listing_is_nan_before_first_session(tmp_path):
    task = make_task("yahoo", ["AAA", "NEW"], "2024-01-01")
    frames = {"AAA": yahoo_frame(rows=5), "NEW": yahoo_frame(first="2024-01-04", rows=2)}
    with patch("rl.marketdata.yahoodownloader.yf.download", side_effect=lambda tic, **k: frames[tic]):
        panel = load_task_data(task, end="2024-01-10", cache_dir=tmp_path)
    assert len(panel) == 5
    assert np.isnan(panel.close[:2, 1]).all() and np.isfinite(panel.close[2:4, 1]).all()
    assert np.isnan(panel.close[4, 1])
