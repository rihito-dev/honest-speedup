"""各例で、新旧の結果が境界ケースを含めて一致することを確かめる。"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import adjacent_returns  # noqa: E402
import bootstrap_counts  # noqa: E402
import rolling_window  # noqa: E402
import topk_persistence  # noqa: E402


class TopKPersistenceTest(unittest.TestCase):
    def check(self, score, symbols, k, horizons, minimum):
        expected = topk_persistence.naive(score.tolist(), symbols, k, horizons, minimum)
        self.assertEqual(expected, topk_persistence.fast(score, symbols, k, horizons, minimum))
        return expected

    def test_random_inputs_with_ties_and_missing(self):
        for seed in range(20):
            score, symbols = topk_persistence.make_input(n_time=80, n_sym=9, seed=seed, missing=0.03)
            self.check(score, symbols, k=3, horizons=[1, 2, 7, 79, 80, 100], minimum=0)
            self.check(score, symbols, k=3, horizons=[1, 5], minimum=60)

    def test_ties_are_broken_by_symbol_name_not_column_order(self):
        # 列の順は B, A, C。t0 は B と A、t1 は B と C が同値で1位を争う
        score = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        member, _ = topk_persistence.member_mask(score, ["B", "A", "C"], k=1)
        self.assertEqual(member.tolist(), [[False, True, False], [True, False, False]])
        # t0 の1位 A は t1 で1位ではない
        self.assertEqual(self.check(score, ["B", "A", "C"], k=1, horizons=[1], minimum=0), {1: (0, 1)})

    def test_one_missing_symbol_drops_the_whole_time(self):
        score = np.array([[3.0, 2.0, 1.0], [3.0, math.nan, 1.0], [3.0, 2.0, 1.0]])
        result = self.check(score, ["A", "B", "C"], k=1, horizons=[1, 2], minimum=0)
        self.assertEqual(result, {1: (0, 0), 2: (1, 1)})

    def test_all_missing(self):
        score = np.full((5, 3), math.nan)
        self.assertEqual(self.check(score, ["A", "B", "C"], k=2, horizons=[1], minimum=1), {1: None})


class BootstrapCountsTest(unittest.TestCase):
    def test_random_inputs(self):
        for seed in range(10):
            eps, sigs = bootstrap_counts.make_input(n_days=30, n_sig=5, seed=seed, mean_per_day=4)
            args = dict(signatures=sigs, trials=50, seed=seed, restart=0.2)
            self.assertEqual(bootstrap_counts.naive(eps, **args), bootstrap_counts.fast(eps, **args))

    def test_trial_with_no_episodes_is_none(self):
        eps = [[], [], []]
        args = dict(signatures=["a"], trials=3, seed=0, restart=0.5)
        self.assertEqual(bootstrap_counts.naive(eps, **args), [None, None, None])
        self.assertEqual(bootstrap_counts.fast(eps, **args), [None, None, None])

    def test_denominator_is_total_episodes_not_mean_of_daily_ratios(self):
        # 1日目: a が1件。2日目: b が3件。日ごとの比率の平均なら a は 0.5 だが、ここでは 1/4
        eps = [[{"signature": "a"}], [{"signature": "b"}] * 3]
        rng = random.Random(0)
        days = bootstrap_counts.draw_days(rng, 2, restart=0.0)
        self.assertEqual(sorted(days), [0, 1])
        result = bootstrap_counts.fast(eps, ["a", "b"], trials=1, seed=0, restart=0.0)
        self.assertEqual(result, [((1, 3), 4)])


class RollingWindowTest(unittest.TestCase):
    def check(self, x, window, min_valid, center=None):
        ref = rolling_window.naive(x.tolist(), window, min_valid)
        cand = rolling_window.fast(x, window, min_valid, center=center)
        self.assertTrue(rolling_window.close_enough(ref, cand))

    def test_random_inputs(self):
        for seed in range(10):
            x = rolling_window.make_input(n=3_000, seed=seed, missing=0.1)
            self.check(x, window=20, min_valid=15)
            self.check(x, window=20, min_valid=15, center=100.0)

    def test_short_series_and_all_missing(self):
        self.check(np.array([1.0, 2.0]), window=3, min_valid=1)
        self.check(np.array([]), window=3, min_valid=1)
        self.check(np.full(10, math.nan), window=3, min_valid=1)

    def test_missing_is_not_filled_with_zero(self):
        means, ok = rolling_window.fast(np.array([2.0, math.nan, 4.0]), window=3, min_valid=2)
        self.assertEqual(ok.tolist(), [False, False, True])
        self.assertEqual(means[2], 3.0)

    def test_centering_reduces_error(self):
        x = rolling_window.make_input(n=200_000, seed=5)
        ref = rolling_window.naive(x.tolist(), 60, 45)
        plain = rolling_window.max_abs_error(ref, rolling_window.fast(x, 60, 45))
        centered = rolling_window.max_abs_error(ref, rolling_window.fast(x, 60, 45, center=100.0))
        self.assertLess(centered, plain)


class AdjacentReturnsTest(unittest.TestCase):
    def test_random_inputs_match_exactly(self):
        for seed in range(5):
            close, adjacent = adjacent_returns.make_input(n_time=200, n_sym=20, seed=seed)
            ref = adjacent_returns.naive(close.tolist(), adjacent.tolist())
            self.assertTrue(adjacent_returns.same(ref, adjacent_returns.fast(close, adjacent)))

    def test_invalid_cases(self):
        close = np.array([[100.0, 100.0, 0.0, math.nan, 100.0], [110.0, 90.0, 5.0, 5.0, math.inf]])
        adjacent = np.array([[True, False, True, True, True]])
        out, ok = adjacent_returns.fast(close, adjacent)
        self.assertEqual(ok.tolist(), [[True, False, False, False, False]])
        self.assertAlmostEqual(out[0, 0], 0.1)
        self.assertTrue(np.isnan(out[0, 1:]).all())

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            adjacent_returns.fast(np.ones((1, 3)), np.ones((0, 3), dtype=bool))
        with self.assertRaises(ValueError):
            adjacent_returns.fast(np.ones((3, 3)), np.ones((3, 3), dtype=bool))


if __name__ == "__main__":
    unittest.main()
