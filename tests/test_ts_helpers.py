# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:30:08 2023

@author: jpeacock
"""

import sys
import unittest

import numpy as np
import pandas as pd
import scipy

# =============================================================================
# Imports
# =============================================================================
from mt_metadata.common.mttime import MTime

from mt_timeseries.ts_helpers import (
    _count_decimal_sig_figs,
    _whole_ns_step_index,
    get_decimation_sample_rates,
    make_dt_coordinates,
    most_common_step,
    sample_rate_matches_step,
)

# =============================================================================


class TestGetDecimationSampleRates(unittest.TestCase):
    def test_4096_to_1(self):
        self.assertListEqual([512, 64, 8, 1], get_decimation_sample_rates(4096, 1, 8))

    def test_1000_to_1(self):
        self.assertListEqual([125, 16, 2, 1], get_decimation_sample_rates(1000, 1, 8))

    def test_1000_to_1000(self):
        self.assertListEqual([1000], get_decimation_sample_rates(1000, 1000, 8))


class TestMakeDtCoordinates(unittest.TestCase):
    def test_input_none(self):
        dt = make_dt_coordinates(None, None, 16)

        with self.subTest("start"):
            self.assertEqual(
                MTime(time_stamp=dt[0]), MTime(time_stamp="1980-01-01T00:00:00")
            )
        with self.subTest("end"):
            self.assertEqual(
                MTime(time_stamp=dt[-1]), MTime(time_stamp="1980-01-01T00:00:15")
            )
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sig_figs_ms(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.0010", 1, 16)

        with self.subTest("start"):
            self.assertEqual(
                MTime(time_stamp=dt[0]), MTime(time_stamp="1980-01-01T00:00:00.001")
            )
        with self.subTest("end"):
            self.assertEqual(
                MTime(time_stamp=dt[-1]), MTime(time_stamp="1980-01-01T00:00:15.001")
            )
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sig_figs_us(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.0000010", 1, 16)

        with self.subTest("start"):
            self.assertEqual(
                MTime(time_stamp=dt[0]), MTime(time_stamp="1980-01-01T00:00:00.000001")
            )
        with self.subTest("end"):
            self.assertEqual(
                MTime(time_stamp=dt[-1]), MTime(time_stamp="1980-01-01T00:00:15.000001")
            )
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sig_figs_ns(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.0000000010", 1, 16)

        with self.subTest("start"):
            self.assertEqual(
                MTime(time_stamp=dt[0]),
                MTime(time_stamp="1980-01-01T00:00:00.000000001"),
            )
        with self.subTest("end"):
            self.assertEqual(
                MTime(time_stamp=dt[-1]),
                MTime(time_stamp="1980-01-01T00:00:15.000000001"),
            )
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sr_sig_figs_ms(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.000", 16, 16)

        with self.subTest("start"):
            self.assertEqual(
                MTime(time_stamp=dt[0]), MTime(time_stamp="1980-01-01T00:00:00.00")
            )
        with self.subTest("end"):
            self.assertEqual(
                MTime(time_stamp=dt[-1]), MTime(time_stamp="1980-01-01T00:00:00.9375")
            )
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sr_sig_figs_us(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.00000", 256, 16)

        with self.subTest("start"):
            self.assertEqual(
                MTime(time_stamp=dt[0]), MTime(time_stamp="1980-01-01T00:00:00.0000")
            )
        with self.subTest("end"):
            self.assertEqual(
                MTime(time_stamp=dt[-1]), MTime(time_stamp="1980-01-01T00:00:00.058594")
            )
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sr_sig_figs_ns(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.00000", 4096, 16)

        with self.subTest("start"):
            self.assertEqual(
                MTime(time_stamp=dt[0]), MTime(time_stamp="1980-01-01T00:00:00.0000")
            )
        with self.subTest("end"):
            self.assertAlmostEqual(
                MTime(time_stamp=dt[-1]).epoch_seconds,
                MTime(time_stamp="1980-01-01T00:00:00.003662109").epoch_seconds,
                6,
            )
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_fix_issue_263(self):
        """
            Note that passing endtime explicitly vs not can causing different values in time coordinates.


        Returns
        -------

        """
        end_str = "2023-10-14T19:47:31.176479359+00:00"
        start_str = "2023-10-14T19:47:23.978079359+00:00"

        dt = 0.13088
        sr = 1 / dt
        n_samples = 56

        tmp1 = make_dt_coordinates(
            start_str, sample_rate=sr, n_samples=n_samples, end_time=end_str
        )
        tmp2 = make_dt_coordinates(
            start_str, sample_rate=sr, n_samples=n_samples, end_time=None
        )

        if sys.version_info >= (3, 9):
            delta_t1 = tmp1.diff()[1:]  # fails in python 3.8
            delta_t2 = tmp2.diff()[1:]  # fails in python 3.8

            # This assertion indicates that delta_t1 is uniform, whereas delta_t2 is not.
            assert len(delta_t1.unique()) == 1
            assert len(delta_t2.unique()) == 2

            # This assertion indicates that the difference is in the first delta.
            assert delta_t1[0] != delta_t2[0]
            assert (delta_t1[1:] == delta_t2[1:]).all()


class TestMakeDtCoordinatesResolution(unittest.TestCase):
    def test_ns_unit(self):
        """pandas 3 infers microseconds from the end time for 3600 samples"""
        dt = make_dt_coordinates("2009-06-16T02:01:04", 10.00064, 3600)
        with self.subTest("unit"):
            self.assertEqual(dt.unit, "ns")
        with self.subTest("steps"):
            self.assertLessEqual(set(np.diff(dt.asi8)), {99_993_600, 99_993_601})

    def test_whole_microsecond_rate(self):
        dt = make_dt_coordinates("2020-01-01T00:00:00", 1000, 3_600_000)
        self.assertEqual(set(np.diff(dt.asi8)), {1_000_000})


class TestMakeDtCoordinatesWholeNsSteps(unittest.TestCase):
    """Whole-ns steps are built in int64: the same index as date_range gives"""

    @staticmethod
    def date_range_index(start, sample_rate, n_samples):
        start = MTime(time_stamp=start)
        end = start + (n_samples - 1) / sample_rate
        dt = pd.date_range(
            start=start.iso_no_tz, end=end.iso_no_tz, periods=n_samples, unit="ns"
        )
        test_sf = max(
            _count_decimal_sig_figs(str(start)),
            _count_decimal_sig_figs(1 / sample_rate),
        )
        for freq, limit in [("ms", 3), ("us", 6), ("ns", 9)]:
            if test_sf < limit:
                return dt.round(freq=freq)
        return dt

    def test_same_as_date_range(self):
        for sample_rate in [1, 8, 10, 128, 256, 1000, 1024, 4096, 24000, 0.1, 1.5]:
            for n_samples in [2, 3, 101, 36001]:
                for start in [
                    "2020-01-01T00:00:00",
                    "2023-09-22T13:51:26.001",
                    "2020-01-01T00:00:00.123456",
                ]:
                    with self.subTest(f"{sample_rate} Hz {n_samples} {start}"):
                        pd.testing.assert_index_equal(
                            make_dt_coordinates(start, sample_rate, n_samples),
                            self.date_range_index(start, sample_rate, n_samples),
                            exact=True,
                        )

    def test_whole_ns_path(self):
        with self.subTest("1000 Hz"):
            self.assertIsNotNone(
                _whole_ns_step_index(
                    "2023-09-22T13:51:26.001", "2023-09-22T13:51:27.001", 1001
                )
            )
        with self.subTest("1024 Hz"):
            self.assertIsNone(
                _whole_ns_step_index("2020-01-01T00:00:00", "2020-01-01T00:00:01", 1025)
            )


class TestMostCommonStep(unittest.TestCase):
    """Counted steps give what scipy.stats.mode of the steps gives"""

    def test_same_as_scipy_mode(self):
        rng = np.random.default_rng(0)
        regular = make_dt_coordinates("2023-09-22T13:51:26.001", 1000, 36001)
        indexes = {
            "1000 Hz": regular,
            "1.5 Hz": make_dt_coordinates("2020-01-01T00:00:00", 1.5, 36000),
            "10.00064 Hz": make_dt_coordinates("2020-01-01T00:00:00", 10.00064, 3600),
            "1024 Hz": make_dt_coordinates("2020-01-01T00:00:00", 1024, 5000),
            "gap": regular.delete(slice(500, 510)),
            "us unit": pd.date_range("2020-01-01", periods=5000, freq="7us", unit="us"),
            "irregular": pd.DatetimeIndex(
                np.cumsum(rng.integers(1, 10**9, 1000)).view("datetime64[ns]")
            ),
            "two samples": regular[:2],
            "blocks tie": pd.DatetimeIndex(
                np.cumsum(np.r_[0, np.full(70000, 1001), np.full(70000, 1000)]).view(
                    "datetime64[ns]"
                )
            ),
            "tie": pd.DatetimeIndex(
                np.cumsum(np.r_[0, np.tile([1001, 1000], 500)]).view("datetime64[ns]")
            ),
        }
        for label, time_index in indexes.items():
            with self.subTest(label):
                expected, counts = scipy.stats.mode(
                    np.diff(time_index) / np.timedelta64(1, "s")
                )
                self.assertEqual(most_common_step(time_index), expected)


class TestSampleRateMatchesStep(unittest.TestCase):
    def test_match(self):
        self.assertTrue(sample_rate_matches_step(1.5, 0.666666666, 36000))
        self.assertTrue(sample_rate_matches_step(10.00064, 0.0999936, 36000))

    def test_no_match(self):
        self.assertFalse(sample_rate_matches_step(2.0, 0.666666666, 36000))
        self.assertFalse(sample_rate_matches_step(10.0, 0.0999936, 36000))

    def test_short_index(self):
        """1 us over the index: MTime rounds the end time to the microsecond"""
        self.assertTrue(sample_rate_matches_step(2048.0, 0.0004885, 3))
        self.assertFalse(sample_rate_matches_step(2048.0, 0.0004885, 36000))

    def test_no_rate(self):
        self.assertFalse(sample_rate_matches_step(None, 0.1, 100))
        self.assertFalse(sample_rate_matches_step(0.0, 0.1, 100))
        self.assertFalse(sample_rate_matches_step(10.0, 0.1, 1))


class TestDecimalSigFigs(unittest.TestCase):
    def test_sig_figs(self):
        for ii in range(1, 12, 1):
            value = f".{ii:0{ii}}1"
            sig_figs = _count_decimal_sig_figs(value)
            with self.subTest(value):
                self.assertEqual(sig_figs, ii + 1)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
