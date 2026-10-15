"""Sharadar SEP conversion on tiny synthetic files in the vendor's format."""

import json

import numpy as np
import pandas as pd
import pytest

from rl.marketdata.sharadar import OUTPUT_COLUMNS, convert_sharadar, load_ticker_map, main
from rl.preprocessing.data import load_dataset

HEADER = "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated"


def rows(ticker, dates, close, adj_factor, unadj_factor=1.0, volume=1000):
    lines = []
    for i, date in enumerate(dates):
        c = close + i
        # open/high/low are split adjusted, like close; closeadj also has dividends removed.
        lines.append(
            "{t},{d},{o},{h},{l},{c},{v},{a},{u},2026-01-01".format(
                t=ticker, d=date, o=c - 1, h=c + 2, l=c - 3, c=c, v=volume,
                a=c * adj_factor, u=c * unadj_factor,
            )
        )
    return lines


DATES = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]


@pytest.fixture
def sep(tmp_path):
    lines = [HEADER]
    lines += rows("AAA", DATES, 100.0, 0.9)
    lines += rows("RTX", DATES, 50.0, 1.0, unadj_factor=2.0)
    lines += rows("OLD", DATES[:2], 20.0, 0.8)
    lines += rows("NEW", DATES[2:], 30.0, 1.0)
    lines += rows("ZZZ", DATES, 10.0, 1.0)
    path = tmp_path / "sep.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_prices_are_put_on_the_closeadj_basis_with_bar_shape_kept(sep):
    out = convert_sharadar(sep, ["AAA"])
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.date.tolist() == DATES and (out.tic == "AAA").all()
    close = np.array([100.0, 101.0, 102.0, 103.0])
    np.testing.assert_allclose(out.close, close * 0.9)
    np.testing.assert_allclose(out.open, (close - 1) * 0.9)
    np.testing.assert_allclose(out.high, (close + 2) * 0.9)
    np.testing.assert_allclose(out.low, (close - 3) * 0.9)
    assert (out.low <= out.open).all() and (out.open <= out.high).all()
    assert (out.low <= out.close).all() and (out.close <= out.high).all()
    # The open-to-close ratio of the vendor's split-adjusted bar is unchanged.
    np.testing.assert_allclose(out.open / out.close, (close - 1) / close)
    assert (out.volume == 1000).all()


def test_ticker_map_renames_a_source_ticker(sep):
    out = convert_sharadar(sep, ["AAA", "UTX"], ticker_map={"UTX": "RTX"})
    assert sorted(out.tic.unique()) == ["AAA", "UTX"]
    utx = out[out.tic == "UTX"]
    np.testing.assert_allclose(utx.close, [50.0, 51.0, 52.0, 53.0])
    # Sorted by date then ticker, ready for the toolkit.
    assert out.equals(out.sort_values(["date", "tic"], ignore_index=True))


def test_ticker_map_stitches_segments_by_date(sep):
    spec = {"DD": [{"ticker": "OLD", "end": "2020-01-03"}, {"ticker": "NEW", "start": "2020-01-06"}]}
    out = convert_sharadar(sep, ["DD"], ticker_map=spec)
    assert out.date.tolist() == DATES and (out.tic == "DD").all()
    np.testing.assert_allclose(out.close, [20 * 0.8, 21 * 0.8, 30.0, 31.0])


def test_overlapping_segments_are_rejected(sep):
    with pytest.raises(ValueError, match="Overlapping"):
        convert_sharadar(sep, ["DD"], ticker_map={"DD": ["RTX", "AAA"]})


def test_seam_jump_is_reported(sep, caplog):
    spec = {"DD": [{"ticker": "OLD", "end": "2020-01-03"}, {"ticker": "NEW", "start": "2020-01-06"}]}
    with caplog.at_level("WARNING"):
        convert_sharadar(sep, ["DD"], ticker_map=spec)
    assert "DD moves" in caplog.text and "2020-01-06" in caplog.text


def test_date_range_is_start_inclusive_end_exclusive(sep):
    out = convert_sharadar(sep, ["AAA"], start_date="2020-01-03", end_date="2020-01-07")
    assert out.date.tolist() == ["2020-01-03", "2020-01-06"]


def test_streaming_in_chunks_matches_reading_at_once(sep):
    once = convert_sharadar(sep, ["AAA", "ZZZ"], ticker_map=None)
    chunked = convert_sharadar(sep, ["AAA", "ZZZ"], chunksize=3)
    pd.testing.assert_frame_equal(once, chunked)
    assert len(once) == 8


def test_rows_without_a_positive_price_are_dropped(tmp_path, caplog):
    path = tmp_path / "bad.csv"
    path.write_text(
        "\n".join([HEADER] + rows("AAA", DATES, 100.0, 1.0)
                  + ["AAA,2020-01-08,0,0,0,0,0,0,0,2026-01-01"]) + "\n"
    )
    with caplog.at_level("WARNING"):
        out = convert_sharadar(path, ["AAA"])
    assert out.date.tolist() == DATES
    assert "dropped 1 rows" in caplog.text


def test_missing_ticker_or_columns_raise(sep, tmp_path):
    with pytest.raises(ValueError, match="None of"):
        convert_sharadar(sep, ["NOPE"])
    narrow = tmp_path / "narrow.csv"
    narrow.write_text("ticker,date,open,close\nAAA,2020-01-02,1,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        convert_sharadar(narrow, ["AAA"])
    with pytest.raises(ValueError, match="'ticker'"):
        convert_sharadar(sep, ["DD"], ticker_map={"DD": [{"start": "2020-01-01"}]})


def test_command_line_writes_a_toolkit_csv(sep, tmp_path):
    map_file = tmp_path / "map.json"
    map_file.write_text(json.dumps({"UTX": "RTX"}))
    out_file = tmp_path / "prices.csv"
    main([str(sep), str(out_file), "--tickers", "AAA", "UTX", "--ticker-map", str(map_file),
          "--start-date", "2020-01-03"])
    loaded = load_dataset(file_name=str(out_file))
    assert list(loaded.columns) == OUTPUT_COLUMNS
    assert sorted(loaded.tic.unique()) == ["AAA", "UTX"]
    assert loaded.date.min() == "2020-01-03"
    assert load_ticker_map(map_file) == {"UTX": "RTX"}
