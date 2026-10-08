"""欠損を含む系列の移動平均: 毎時点で窓を取り直す版と累積和の版。

意味として固定するもの:
- 窓は時点 t を含む直前 window 点。系列の先頭で窓が足りなければ無効
- 窓内の欠損(NaN)は0で埋めず、有効な値だけで平均する。有効件数が min_valid 未満なら無効
- 無効な時点は NaN、有効性は別の bool 配列で返す

累積和の差は丸めの順が変わるので、値は厳密には一致しない。
許容誤差は比較の前に下の TOLERANCE で決める。有効性の配列は厳密に一致させる。
"""

from __future__ import annotations

import math
import sys

import numpy as np
from numpy.typing import NDArray

# 比較前に決めた許容誤差。値は100前後、後段の判定は1e-4刻みなので、それより十分小さく取る。
# 結果を見てから広げない。
ATOL = 1e-8
RTOL = 1e-10


def naive(values: list[float], window: int, min_valid: int) -> tuple[list[float], list[bool]]:
    """各時点で窓を切り出して数え直す。O(T × window)。"""
    means, ok = [], []
    for t in range(len(values)):
        if t + 1 < window:
            means.append(math.nan)
            ok.append(False)
            continue
        finite = [x for x in values[t + 1 - window : t + 1] if math.isfinite(x)]
        if len(finite) < min_valid:
            means.append(math.nan)
            ok.append(False)
        else:
            means.append(math.fsum(finite) / len(finite))
            ok.append(True)
    return means, ok


def fast(
    values: NDArray[np.float64], window: int, min_valid: int, center: float | None = None
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """有効値の累積和と有効件数の累積和から、窓の和を差で求める。O(T)。

    center を与えると、その値を引いてから累積する。値の水準が大きいほど
    累積和の桁が大きくなり、差を取ったときに失う桁も増えるため。
    """
    finite = np.isfinite(values)
    shift = 0.0 if center is None else center
    filled = np.where(finite, values - shift, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccnt = np.concatenate(([0], np.cumsum(finite, dtype=np.int64)))
    n = len(values)
    means = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    if n < window:
        return means, ok
    end = np.arange(window, n + 1)  # 窓 [end - window, end)
    s = csum[end] - csum[end - window]
    c = ccnt[end] - ccnt[end - window]
    good = c >= min_valid
    t = end - 1
    ok[t] = good
    means[t[good]] = s[good] / c[good] + shift
    return means, ok


def close_enough(
    reference: tuple[list[float], list[bool]],
    candidate: tuple[NDArray[np.float64], NDArray[np.bool_]],
) -> bool:
    ref_mean, ref_ok = np.array(reference[0], dtype=np.float64), np.array(reference[1], dtype=bool)
    cand_mean, cand_ok = candidate
    if not np.array_equal(ref_ok, cand_ok):
        return False  # 有効性の不一致は誤差では済ませない
    if np.isfinite(cand_mean[~cand_ok]).any():
        return False
    diff = np.abs(cand_mean[ref_ok] - ref_mean[ref_ok])
    return bool(np.all(diff <= ATOL + RTOL * np.abs(ref_mean[ref_ok])))


def max_abs_error(
    reference: tuple[list[float], list[bool]],
    candidate: tuple[NDArray[np.float64], NDArray[np.bool_]],
) -> float:
    ref_mean, ref_ok = np.array(reference[0], dtype=np.float64), np.array(reference[1], dtype=bool)
    if not ref_ok.any():
        return 0.0
    return float(np.max(np.abs(candidate[0][ref_ok] - ref_mean[ref_ok])))


def make_input(n: int, seed: int, level: float = 100.0, missing: float = 0.05) -> NDArray[np.float64]:
    """水準 level の周りを動く系列(AR(1))。許容誤差の前提「値は100前後」に合わせる。"""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1.0, n)
    dev = np.empty(n)
    dev[0] = 0.0
    for t in range(1, n):
        dev[t] = 0.98 * dev[t - 1] + noise[t]
    x = level + dev
    x[rng.random(n) < missing] = np.nan
    return x


def main() -> int:
    from _bench import compare

    x = make_input(n=200_000, seed=5)
    xl = x.tolist()
    args = dict(window=60, min_valid=45)
    ref = naive(xl, **args)
    print(f"rolling_window 最大絶対誤差: 中心化なし {max_abs_error(ref, fast(x, **args)):.3e}"
          f" / 中心化あり {max_abs_error(ref, fast(x, **args, center=100.0)):.3e}"
          f" (許容 atol={ATOL:g}, rtol={RTOL:g})")
    return compare(
        "rolling_window",
        lambda: naive(xl, **args),
        lambda: fast(x, **args, center=100.0),
        equal=close_enough,
        note="T=200,000, 窓60, 有効45件以上, 欠損5%",
    )


if __name__ == "__main__":
    sys.exit(main())
