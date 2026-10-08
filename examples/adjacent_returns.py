"""隣接2点のリターン: Pythonの二重ループ版とNumPyのmask版。

入力は整列済みの close[time, symbol] と、前後2点を繋いでよいかを示す adjacent[time-1, symbol]。
adjacent は読み出し側の品質検査の結果として受け取り、配列上で隣にあることから推定しない。

意味として固定するもの:
- adjacent が偽、どちらかが非有限、どちらかが0以下なら無効。無効は NaN と mask で返す
- 割ってから1を引く。演算の種類と順が同じなので、IEEE 754 の範囲で厳密に一致する
"""

from __future__ import annotations

import math
import sys

import numpy as np
from numpy.typing import NDArray


def naive(close: list[list[float]], adjacent: list[list[bool]]) -> tuple[list[list[float]], list[list[bool]]]:
    out, ok = [], []
    for t in range(1, len(close)):
        row, row_ok = [], []
        for s, (p, c) in enumerate(zip(close[t - 1], close[t])):
            good = adjacent[t - 1][s] and math.isfinite(p) and math.isfinite(c) and p > 0 and c > 0
            row.append(c / p - 1.0 if good else math.nan)
            row_ok.append(good)
        out.append(row)
        ok.append(row_ok)
    return out, ok


def fast(
    close: NDArray[np.float64], adjacent: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    if close.ndim != 2 or close.shape[0] < 2:
        raise ValueError("expected at least two aligned time points")
    prev, curr = close[:-1], close[1:]
    if adjacent.shape != prev.shape:
        raise ValueError("invalid adjacency mask")
    with np.errstate(invalid="ignore"):  # NaN との比較は無効として扱うので警告は不要
        valid = adjacent & np.isfinite(prev) & np.isfinite(curr) & (prev > 0) & (curr > 0)
    out = np.full(prev.shape, np.nan, dtype=np.float64)
    np.divide(curr, prev, out=out, where=valid)
    np.subtract(out, 1.0, out=out, where=valid)
    return out, valid


def same(
    reference: tuple[list[list[float]], list[list[bool]]],
    candidate: tuple[NDArray[np.float64], NDArray[np.bool_]],
) -> bool:
    ref_out, ref_ok = np.array(reference[0]), np.array(reference[1])
    return bool(np.array_equal(ref_ok, candidate[1]) and np.array_equal(ref_out, candidate[0], equal_nan=True))


def make_input(n_time: int, n_sym: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_time, n_sym)), axis=0))
    close[rng.random(close.shape) < 0.01] = np.nan
    close[rng.random(close.shape) < 0.001] = 0.0
    adjacent = rng.random((n_time - 1, n_sym)) > 0.02
    return close, adjacent


def main() -> int:
    from _bench import compare

    close, adjacent = make_input(n_time=5_000, n_sym=100, seed=2)
    cl, al = close.tolist(), adjacent.tolist()
    return compare(
        "adjacent_returns",
        lambda: naive(cl, al),
        lambda: fast(close, adjacent),
        equal=same,
        note="T=5,000 × S=100",
    )


if __name__ == "__main__":
    sys.exit(main())
