"""Offline regression tests for Yahoo Finance's historical data layouts."""

from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from rl.marketdata.yahoodownloader import YahooDownloader, total_return_factors


OUTPUT_COLUMNS = ["date", "open", "high", "low", "close", "volume", "tic", "day"]


def prices():
    # Rows arrive out of date order. The 2024-01-03 dividend is 10% of the
    # close before it, so every earlier price is scaled by 0.9.
    return pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [105.0, 115.0],
            "Low": [95.0, 105.0],
            "Close": [102.0, 112.0],
            # Yahoo recomputes this on every request; it must not be used.
            "Adj Close": [1.0, 2.0],
            "Volume": [1000, 2000],
            "Dividends": [11.2, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-02"]),
    ).rename_axis("Date")


@pytest.mark.parametrize("multi_index", [False, True])
def test_download_adjusts_for_dividends_and_normalizes_columns(multi_index):
    source = prices()
    # Modern responses use alphabetic price order rather than legacy OHLC.
    if multi_index:
        source = source[sorted(source.columns)]
        source.columns = pd.MultiIndex.from_product(
            [source.columns, ["AAA"]], names=["Price", "Ticker"]
        )
    original = source.copy(deep=True)
    downloader = YahooDownloader("2024-01-01", "2024-01-04", ["AAA"])

    with patch("rl.marketdata.yahoodownloader.yf.download", return_value=source):
        result = downloader.fetch_data()

    expected = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [99.0, 100.0],
            "high": [103.5, 105.0],
            "low": [94.5, 95.0],
            "close": [100.8, 102.0],
            "volume": [2000, 1000],
            "tic": ["AAA", "AAA"],
            "day": [1, 2],
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    pd.testing.assert_frame_equal(source, original)


def test_forward_factors_keep_earlier_rows_when_rows_are_appended():
    close = np.array([100.0, 101.0, 99.0, 103.0])
    dividends = np.array([5.0, 0.0, 10.1, 0.0])  # the first row's has no earlier close
    factors = total_return_factors(close, dividends)
    assert factors == pytest.approx([1.0, 1.0, 1.0 / 0.9, 1.0 / 0.9])
    longer = total_return_factors(np.append(close, 50.0), np.append(dividends, 5.15))
    assert np.array_equal(longer[:4], factors)
    for dividend in (100.0, 150.0):
        with pytest.raises(ValueError, match="not below the close"):
            total_return_factors(close, [0.0, dividend, 0.0, 0.0])


def test_dividend_on_a_row_without_prices_goes_ex_at_the_next_session():
    source = pd.DataFrame(
        {
            "Open": [100.0, np.nan, 100.0],
            "High": [100.0, np.nan, 100.0],
            "Low": [100.0, np.nan, 100.0],
            "Close": [100.0, np.nan, 100.0],
            "Volume": [1, np.nan, 1],
            "Dividends": [0.0, 10.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    downloader = YahooDownloader("2024-01-01", "2024-01-05", ["AAA"])
    with patch("rl.marketdata.yahoodownloader.yf.download", return_value=source):
        unadjusted = downloader.fetch_unadjusted()
        adjusted = downloader.fetch_data()
    assert unadjusted["date"].tolist() == ["2024-01-02", "2024-01-04"]
    assert unadjusted["dividends"].tolist() == [0.0, 10.0]
    assert adjusted["close"].tolist() == pytest.approx([90.0, 100.0])


def test_multiple_tickers_skip_empty_download_and_sort_rows():
    aaa = prices()
    bbb = prices() * 2
    bbb.loc[pd.Timestamp("2024-01-03"), "Close"] = float("nan")
    downloader = YahooDownloader("2024-01-01", "2024-01-04", ["BBB", "EMPTY", "AAA"])

    with patch(
        "rl.marketdata.yahoodownloader.yf.download",
        side_effect=[bbb, pd.DataFrame(), aaa],
    ):
        result = downloader.fetch_data()

    assert list(result.columns) == OUTPUT_COLUMNS
    assert result[["date", "tic"]].values.tolist() == [
        ["2024-01-02", "AAA"],
        ["2024-01-02", "BBB"],
        ["2024-01-03", "AAA"],
    ]
    # BBB's dividend went ex on its dropped last row, after its only price.
    assert result["close"].tolist() == pytest.approx([100.8, 224.0, 102.0])
    assert result.index.tolist() == [0, 1, 2]


@pytest.mark.parametrize("tickers,response", [([], None), (["AAA"], None), (["AAA"], pd.DataFrame())])
def test_no_data_returns_stable_empty_schema(tickers, response):
    with patch("rl.marketdata.yahoodownloader.yf.download", return_value=response) as download:
        result = YahooDownloader("2024-01-01", "2024-01-04", tickers).fetch_data()
    assert result.empty
    assert list(result.columns) == OUTPUT_COLUMNS
    assert download.call_count == len(tickers)


def test_missing_dividends_reports_ticker_and_column():
    with patch(
        "rl.marketdata.yahoodownloader.yf.download",
        return_value=prices().drop(columns="Dividends"),
    ):
        with pytest.raises(ValueError, match="Missing Yahoo Finance columns for AAA: Dividends"):
            YahooDownloader("2024-01-01", "2024-01-04", ["AAA"]).fetch_data()


def test_multi_index_accepts_normalized_ticker_case():
    source = prices()
    source.columns = pd.MultiIndex.from_product([source.columns, ["AAA"]])
    with patch("rl.marketdata.yahoodownloader.yf.download", return_value=source):
        result = YahooDownloader("2024-01-01", "2024-01-04", ["aaa"]).fetch_data()
    assert result["tic"].tolist() == ["aaa", "aaa"]
    assert result["close"].tolist() == pytest.approx([100.8, 102.0])


def test_unexpected_multiple_tickers_are_rejected():
    source = pd.concat([prices(), prices()], axis=1, keys=["AAA", "BBB"])
    source = source.swaplevel(axis=1)
    with patch("rl.marketdata.yahoodownloader.yf.download", return_value=source):
        with pytest.raises(ValueError, match="Expected one Yahoo Finance ticker for AAA"):
            YahooDownloader("2024-01-01", "2024-01-04", ["AAA"]).fetch_data()
