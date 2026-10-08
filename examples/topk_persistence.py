"""上位k位の持続率: 辞書ループ版と配列版。

時点 t で上位 k 位に入っていた銘柄が、t + horizon でも上位 k 位にいる割合を求める。

意味として固定するもの:
- 順位はスコア降順、同値は銘柄名の昇順で決める
- どれか1銘柄でもスコアが欠損している時点は、時点ごと除外する
- 分母は「両端の時点が有効な組」での現在の上位件数。minimum 未満なら None
"""

from __future__ import annotations

import math
import random
import sys

import numpy as np
from numpy.typing import NDArray


def naive(
    score: list[list[float]],
    symbols: list[str],
    k: int,
    horizons: list[int],
    minimum: int,
) -> dict[int, tuple[int, int] | None]:
    """時点×horizonごとに順位表を作り直す、よくある書き方。"""
    n_time = len(score)

    def valid(t: int) -> bool:
        return all(math.isfinite(x) for x in score[t])

    def top_k(t: int) -> set[str]:
        ranked = sorted(zip(symbols, score[t]), key=lambda p: (-p[1], p[0]))
        return {sym for sym, _ in ranked[:k]}

    out: dict[int, tuple[int, int] | None] = {}
    for h in horizons:
        held = observed = 0
        for t in range(n_time - h):
            if not (valid(t) and valid(t + h)):
                continue
            now, later = top_k(t), top_k(t + h)
            observed += len(now)
            held += len(now & later)
        out[h] = (held, observed) if observed >= minimum else None
    return out


def member_mask(
    score: NDArray[np.float64], symbols: list[str], k: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """上位k位の所属 member[time, symbol] と、時点の有効性 valid[time] を一度だけ作る。"""
    n_time, n_sym = score.shape
    valid = np.isfinite(score).all(axis=1)
    # 同値を銘柄名の昇順で解くため、名前順の整数を第2キーにする。
    # np.lexsort は最後のキーが第1キー。argpartition では同値の順が保証されない。
    name_rank = np.empty(n_sym, dtype=np.int64)
    name_rank[np.argsort(np.array(symbols), kind="stable")] = np.arange(n_sym)
    order = np.lexsort((np.broadcast_to(name_rank, score.shape), -score), axis=-1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n_sym)[None, :].repeat(n_time, 0), axis=-1)
    return rank < k, valid


def fast(
    score: NDArray[np.float64],
    symbols: list[str],
    k: int,
    horizons: list[int],
    minimum: int,
) -> dict[int, tuple[int, int] | None]:
    member, valid = member_mask(score, symbols, k)
    n_time = score.shape[0]
    out: dict[int, tuple[int, int] | None] = {}
    for h in horizons:
        if h >= n_time:
            out[h] = None if minimum > 0 else (0, 0)
            continue
        pair_ok = valid[: n_time - h] & valid[h:]
        now = member[: n_time - h] & pair_ok[:, None]
        later = member[h:]
        observed = int(np.count_nonzero(now))
        held = int(np.count_nonzero(now & later))
        out[h] = (held, observed) if observed >= minimum else None
    return out


def make_input(
    n_time: int, n_sym: int, seed: int, missing: float = 0.01
) -> tuple[NDArray[np.float64], list[str]]:
    """同値が出るよう小数1桁に丸めた合成スコア。欠損も混ぜる。"""
    rng = np.random.default_rng(seed)
    score = np.round(rng.normal(size=(n_time, n_sym)), 1)
    score[rng.random(score.shape) < missing] = np.nan
    symbols = [f"S{i:03d}" for i in range(n_sym)]
    random.Random(seed).shuffle(symbols)  # 列の並びと名前順を一致させない
    return score, symbols


def main() -> int:
    from _bench import compare

    score, symbols = make_input(n_time=2_000, n_sym=40, seed=7)
    score_list = score.tolist()
    args = dict(symbols=symbols, k=5, horizons=[1, 5, 20, 60], minimum=50)
    return compare(
        "topk_persistence",
        lambda: naive(score_list, **args),
        lambda: fast(score, **args),
        equal=lambda a, b: a == b,
        note="T=2,000 × S=40, k=5, horizon 4種",
    )


if __name__ == "__main__":
    sys.exit(main())
