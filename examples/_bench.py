"""例の共通計測。結果の一致を先に確かめ、一致しなければ時間を出さない。"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

REPEAT = 5  # 反復回数は比較前に固定する


def compare(
    name: str,
    before: Callable[[], Any],
    after: Callable[[], Any],
    equal: Callable[[Any, Any], bool],
    note: str,
) -> int:
    reference, candidate = before(), after()
    if not equal(reference, candidate):
        print(f"{name}: 結果が一致しない。時間は比較しない")
        return 1

    def measure(fn: Callable[[], Any]) -> list[float]:
        runs = []
        for _ in range(REPEAT):
            start = time.perf_counter()
            fn()
            runs.append(time.perf_counter() - start)
        return runs

    b, a = measure(before), measure(after)
    mb, ma = statistics.median(b), statistics.median(a)
    print(f"{name} ({note})")
    print(f"  結果: 一致")
    print(f"  before 中央値 {mb * 1e3:9.2f} ms  範囲 {min(b) * 1e3:.2f}–{max(b) * 1e3:.2f}")
    print(f"  after  中央値 {ma * 1e3:9.2f} ms  範囲 {min(a) * 1e3:.2f}–{max(a) * 1e3:.2f}")
    print(f"  比     {mb / ma:.1f}x (反復 {REPEAT} 回、初回の呼び出しは一致確認に使い計測から除外)")
    return 0
