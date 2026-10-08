"""ブロック・ブートストラップでの種類別比率: 行を複製する版と件数表を足す版。

日ごとに観測された「事象」(episode)があり、各事象は1つの種類(signature)を持つ。
日をブロック単位で復元抽出し、抽出された事象全体に占める各種類の割合を試行ごとに求める。

意味として固定するもの:
- 日の抽出は両版で同じ関数・同じ乱数生成器・同じ消費順を使う
- 比率の分母は「抽出された事象の総数」。日ごとの比率の平均ではない
- 抽出された事象が0件の試行は None

比率そのものではなく (件数, 総数) の整数を返すので、新旧は厳密に一致する。
"""

from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np
from numpy.typing import NDArray

Trial = tuple[tuple[int, ...], int] | None


def draw_days(rng: random.Random, n_days: int, restart: float) -> list[int]:
    """定常ブロック・ブートストラップ。確率 restart で新しい開始日へ飛び、それ以外は翌日へ(末尾は先頭へ戻る)。"""
    days = [rng.randrange(n_days)]
    while len(days) < n_days:
        days.append(rng.randrange(n_days) if rng.random() < restart else (days[-1] + 1) % n_days)
    return days


def naive(
    episodes_by_day: list[list[dict[str, object]]],
    signatures: list[str],
    trials: int,
    seed: int,
    restart: float,
) -> list[Trial]:
    """試行ごとに抽出日の事象を list へ複製して数え直す。"""
    rng = random.Random(seed)
    out: list[Trial] = []
    for _ in range(trials):
        sample: list[dict[str, object]] = []
        for d in draw_days(rng, len(episodes_by_day), restart):
            sample.extend(dict(e) for e in episodes_by_day[d])
        if not sample:
            out.append(None)
            continue
        counter = Counter(e["signature"] for e in sample)
        out.append((tuple(counter[s] for s in signatures), len(sample)))
    return out


def count_table(
    episodes_by_day: list[list[dict[str, object]]], signatures: list[str]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """C[day, signature] と N[day] を一度だけ作る。"""
    index = {s: j for j, s in enumerate(signatures)}
    c = np.zeros((len(episodes_by_day), len(signatures)), dtype=np.int64)
    for d, eps in enumerate(episodes_by_day):
        for e in eps:
            c[d, index[e["signature"]]] += 1
    return c, c.sum(axis=1)


def fast(
    episodes_by_day: list[list[dict[str, object]]],
    signatures: list[str],
    trials: int,
    seed: int,
    restart: float,
) -> list[Trial]:
    c, n = count_table(episodes_by_day, signatures)
    rng = random.Random(seed)  # 乱数の生成器・seed・消費順は naive と同じ
    out: list[Trial] = []
    for _ in range(trials):
        idx = np.fromiter(draw_days(rng, len(episodes_by_day), restart), dtype=np.int64)
        total = int(n[idx].sum())
        out.append(None if total == 0 else (tuple(int(x) for x in c[idx].sum(axis=0)), total))
    return out


def make_input(
    n_days: int, n_sig: int, seed: int, mean_per_day: float = 30.0
) -> tuple[list[list[dict[str, object]]], list[str]]:
    rng = random.Random(seed)
    signatures = [f"sig{j}" for j in range(n_sig)]
    weights = [1.0 / (j + 1) for j in range(n_sig)]
    episodes_by_day = []
    for d in range(n_days):
        n = 0 if rng.random() < 0.05 else int(rng.expovariate(1 / mean_per_day))
        episodes_by_day.append(
            [
                {"day": d, "signature": s, "start": i, "length": rng.randint(1, 30)}
                for i, s in enumerate(rng.choices(signatures, weights, k=n))
            ]
        )
    return episodes_by_day, signatures


def main() -> int:
    from _bench import compare

    eps, sigs = make_input(n_days=250, n_sig=12, seed=11)
    args = dict(signatures=sigs, trials=200, seed=3, restart=0.1)
    return compare(
        "bootstrap_counts",
        lambda: naive(eps, **args),
        lambda: fast(eps, **args),
        equal=lambda a, b: a == b,
        note="250日・1日平均約30事象・12種類・200試行",
    )


if __name__ == "__main__":
    sys.exit(main())
