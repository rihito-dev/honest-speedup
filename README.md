# same-result-faster

[![test](https://github.com/rihito-dev/same-result-faster/actions/workflows/test.yml/badge.svg)](https://github.com/rihito-dev/same-result-faster/actions/workflows/test.yml)

**Make slow Python research code faster without changing its results — a guide, a planning prompt for agents, and runnable before/after examples.**

数時間かかるPythonの研究計算(バックテスト、帰無仮説の試行、ブートストラップ、窓集計など)を、
**結果を変えずに**速くするためのガイドです。

速くすること自体より、「何が同じでなければならないか」を先に固定し、それを確かめながら速くすることを重視しています。
同順位の決め方、欠損の扱い、乱数の消費順、時点の境界、許容誤差の決め方などです。

## 中身

- **[GUIDE.md](GUIDE.md)** — 技術ガイド。遅さの型、直す順番、ライブラリの使い分け、保存形式、乱数、計測と受け入れ
- **[PLANNING_PROMPT.md](PLANNING_PROMPT.md)** — 計算計画を立てるエージェントへの指示。`AGENTS.md`や`CLAUDE.md`から参照させるか、そのまま貼って使う
- **[examples/](examples/)** — よくある遅さの型の「before / after」。どれも新旧の結果が一致することを確かめてから時間を測る

| 例 | 遅さの型 | 直し方 | 一致の基準 |
|---|---|---|---|
| [topk_persistence.py](examples/topk_persistence.py) | 時点ごとに順位表を作り直す | 所属maskを1回作ってboolの積を数える | 厳密一致(整数) |
| [bootstrap_counts.py](examples/bootstrap_counts.py) | 試行ごとに行を複製して数え直す | 日×種類の件数表を足す | 厳密一致(整数) |
| [rolling_window.py](examples/rolling_window.py) | 各時点で窓を切り出して再計算 | 累積和の差 | 事前に決めた許容誤差、有効性は厳密一致 |
| [adjacent_returns.py](examples/adjacent_returns.py) | Pythonの二重ループ | maskとufunc | 厳密一致(同じ演算順) |

## 試す

必要なのはPython 3.10以上とNumPyだけです。

```bash
python -m pip install numpy
```

```bash
python -m unittest discover -s tests -v
```

```bash
cd examples && python topk_persistence.py
```

## 計測結果

以下の環境で各例の`main()`を実行した結果です。反復5回の中央値で、初回の呼び出しは一致確認に使い計測から外しています。
倍率は入力の大きさと形で大きく変わるので、自分の処理で測り直してください。

環境: Apple M1 / RAM 16GB / macOS / Python 3.10.8 / NumPy 2.2.6

| 例 | 入力 | before | after | 比 | 結果 |
|---|---|---:|---:|---:|---|
| topk_persistence | T=2,000 × S=40, k=5, horizon 4種 | 67.95 ms | 2.08 ms | 32.7x | 一致 |
| bootstrap_counts | 250日・1日平均約30事象・12種類・200試行 | 185.90 ms | 10.88 ms | 17.1x | 一致 |
| rolling_window | T=200,000, 窓60, 欠損5% | 784.88 ms | 5.37 ms | 146.1x | 許容誤差内 |
| adjacent_returns | T=5,000 × S=100 | 119.58 ms | 8.67 ms | 13.8x | 一致 |

rolling_windowの最大絶対誤差は、そのまま累積すると約5.3e-10、基準値100を引いてから累積すると約6.4e-13でした
(許容は事前に決めた`atol=1e-8, rtol=1e-10`)。累積和を使うなら、値の水準を引いてから足すと桁落ちが減ります。

## 由来

個人の研究リポジトリで、数時間の計算待ちを短くするために書いたガイドを、
特定のデータ・プロジェクトに依存しない形へ書き直したものです。

## License

MIT
