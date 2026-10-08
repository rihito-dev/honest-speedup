# 結果を変えずにPython計算を速くするガイド

対象: 時系列・パネルデータの研究計算(バックテスト、帰無仮説の試行、ブートストラップ、窓集計など)で、
数時間かかる計算待ちを短くしたい人と、その計画を立てるエージェント。

計算計画を作るエージェントには、先に[計算計画エージェントへの指示](PLANNING_PROMPT.md)を読ませる。
本書はその判断を支える技術解説として使う。

主張は1つ。**速くする前に「何が同じでなければならないか」を固定し、それを確かめながら速くする。**
速くなっても、入力・欠損・時点・乱数・計算条件が変わった結果を同じ実験として扱わない。

## まず読む: 数時間の計算待ちを短くする

最初に疑うのは、ライブラリや保存形式ではなく**同じ仕事の繰り返し**である。
1回0.2秒の小さなPython処理でも、250日×5条件×20試行なら約83分になる。
内側の1回を速くするより、繰り返し自体を減らすほうが効くことが多い。

順番は次のとおり。

1. 繰り返し探索・再集計・オブジェクト生成を減らす
2. 数値カーネルを配列化する、またはコンパイルする
3. 保存形式や読み出しを見直す
4. 並列化を検討する

CPU計算が支配的な処理は、保存形式をParquetへ変えてもその部分は短くならない。

### よくある遅さの型と直し方

| 遅くなりやすい型 | よくある書き方 | 直し方 | 動く例 |
|---|---|---|---|
| 上位k位の持続 | 時点×horizonごとに順位表を作り直し、辞書で照合 | 時点×銘柄の所属maskを1回作り、boolの積を数える | [topk_persistence.py](examples/topk_persistence.py) |
| ブートストラップ | 試行ごとに抽出日の行を複製して数え直す | 日×種類の件数表を1回作り、抽出した日の行を足す | [bootstrap_counts.py](examples/bootstrap_counts.py) |
| 窓集計 | 各時点で窓を切り出して再計算 | 有効値の累積和と有効件数の累積和の差 | [rolling_window.py](examples/rolling_window.py) |
| 要素ごとの計算 | Pythonの二重ループ | maskとufuncで一括計算 | [adjacent_returns.py](examples/adjacent_returns.py) |
| 帰無試行の系列再構成 | 試行ごとにreturnを再計算し、オブジェクトを生成 | 元のreturnと時刻軸を1回用意し、試行内部は数値配列だけ扱う | — |
| 過去日の同時刻探索 | 過去日ごとに先頭から時刻を線形探索 | 日別の時刻→index辞書を1回作って再利用 | — |

各例は新旧の結果が一致することを確かめてから時間を測る。
手元の計測値は[README](README.md#計測結果)にある。

### A. 内側の辞書ループを配列演算へ移す

順位表を`rank[time, symbol]`、有効時点を`valid[time]`として持つ。
top-kの所属を`member = rank < k`として先に作れば、horizonごとの持続件数は
「現在所属かつ将来所属」のbool配列の合計になる。分母は、両端が有効な時点の組での現在所属数。

```text
pair_ok  = valid[:-h] & valid[h:]
now      = member[:-h] & pair_ok[:, None]
later    = member[h:]
observed = count_nonzero(now)
held     = count_nonzero(now & later)
result   = held / observed if observed >= minimum else None
```

ここで削るのは、position×時点×銘柄のPython辞書走査である。
試行ごとに価格系列が違うなら、その順位までは共有してはいけない。
高速化のついでに比較先の時点を変えるなど、仕様の解釈変更を混ぜない。

同順位の扱いは必ず保つ。「スコア降順、同値は銘柄名の昇順」なら、名前順を第2キーにした安定な並べ替えを使う。
`argpartition`への単純な置き換えでは、同順位の決まり方が保証されない。
1銘柄でも欠損した時点を時点ごと除外する、といった既存の条件も保つ。

### B. ブートストラップでは行そのものを複製しない

種類別の比率に必要なのは、抽出された日に属する種類別件数と総件数だけである。
日`d`・種類`s`の件数を`C[d, s]`、日別総数を`N[d]`として一度だけ集計する。
現行と同じ乱数手順で得た日index列`I`に対して`sum(C[I, s]) / sum(N[I])`を計算すれば、
行の複製を省ける。日ごとの比率の平均ではなく、現行どおり抽出した総件数を分母にする。

事象数を`E`、日数を`D`、種類数を`K`、試行数を`B`とすると、
複製・数え直しの部分はおおむね`O(BE)`から`O(E + BDK)`になる。
`K`が大きければ、疎なCounterや出現した種類だけの加算も比べる。
ブロック抽出の再開始確率・連続日・末尾から先頭への折り返し・seed・乱数の呼び出し順はそのまま保つ。
これは標準ライブラリだけでも試せる改善で、NumPyは前提ではない。

比率ではなく(件数, 総数)の整数を比べれば、新旧は厳密に一致させられる。

### C. 数値ループと状態遷移を分ける

「待機→発火→確認→再待機」のような状態遷移や、indexの飛び越しがある処理を、
単純なboolの一括評価へ変えると、検出される事象が変わる。
まずreturn・有効窓・標準偏差など数値の部分を切り出し、状態遷移は元の順序で走査する。

Numbaを試すなら、内側を同じ型の数値配列・整数・boolにし、日時・dict・dataclassの組み立ては外側へ出す。
入口で入力を検査したあと、`@njit(cache=True, fastmath=False)`の逐次カーネルと比べる。
`cache=True`はコンパイル結果の再利用であり、実験結果のキャッシュではない。
JITは初回compile、2回目以降、配列への変換まで含めた全体時間を分けて測る。
`fastmath=False`でも、Pythonの`math.fsum`との厳密一致は自動では保証されない。
分散を`E[x²] - E[x]²`で速く計算し、負になった値を0に丸めて済ませるようなことはしない。
[Numba公式の性能ガイド](https://numba.readthedocs.io/en/stable/user/performance-tips.html)。

### D. 全量を流す前に、短い計測用の入口を作る

正式な実験とは別のbenchmarkとして、固定した日・固定した試行IDの代表部分を使う。
通常の日だけでなく、欠損や長いセッションなどの重い日も含める。
本番の試行数を減らして結果を採用することとは区別する。

```text
python -m cProfile -o profile.out <benchmark入口.py>
python -m pstats profile.out
# pstats内: sort cumulative → stats 30
# 続けて:   sort time → stats 30
```

`cumtime`で重い呼び出し経路、`tottime`で関数自身、`ncalls`で反復回数を見る。
上位にdecodeが来るならI/Oと保存形式、Pythonの数値処理が来るなら配列化やJIT、
検索や同じ集計が来るならindexと前計算を優先する。
profileを付けた時間は改善率に使わず、通常実行で改めて比べる。
[Python profiler公式](https://docs.python.org/3/library/profile.html)。

途中再開は、日・条件・試行IDを固定した完了chunk単位で考える。
入力hash・設定・実装版が一致した完了分だけを再利用し、最後に固定順で集約する。
進捗には完了単位数と直近の所要時間を出し、残り時間は推定と明記する。
これは総演算量を減らさないが、中断後に数時間やり直すことを避けられる。

最初の成果物は「上位の計算コスト」「1つの改善の前後比較」「結果一致の証拠」の3つにする。
ライブラリの全面導入や全実験の書き換えから始めない。

## 1. 最適化しても変えない契約

研究ごとに中身は違うが、次の種類の約束はたいてい必要になる。速くする前に書き出しておく。

- **原本は不変。** 圧縮・列削減・並べ替えは、原本から再生成できる派生側で行う。
- **時点の境界を守る。** 「その時点で知り得たこと」だけで計算する。後から訂正・調整される値を、時点を持たないキーで保存・再利用しない。
- **欠損は欠損のまま。** 0や前の値で埋めない。未観測の対象、品質で除外した件数、分母、除外理由を残す。
- **母集団を黙って縮めない。** inner joinだけで整列すると対象が減る。構成一覧と時刻軸を別に持つ。
- **重複・訂正の選び方を保つ。** 単純な`drop_duplicates`や「最新行を取る」に置き換えない。曖昧なら止める。
- **計算を変えたら版を上げる。** 指標の計算コードを変えたら版を上げて再生成し、実装・入力・設定の同一性を残す。

## 2. ライブラリの役割を分ける

| 道具 | 向いている仕事 | 注意点 |
|---|---|---|
| 標準ライブラリ | 入力検証、manifest、型付きの境界、逐次読み込み、`cProfile` | 小さな処理はそのままでよい。Pythonループだからという理由だけで置き換えない |
| NumPy | 同じ演算を大量の数値へ。`float64`配列・ufunc・mask・broadcast | 配列化のコスト、一時配列、欠損の表し方、演算順も含めて測る |
| PyArrow | 明示したschemaでのParquet保存、必要な列だけのbatch読み込み | 型推論に任せない。`to_table()`などでの全件実体化を避ける |
| DuckDB | Parquetの列・行の絞り込み、結合、集計をSQLで | 結果を全件Pythonオブジェクトへ戻すと節約が消える |
| pandas | 外部SDKとの境界、小さな表の検査 | 組み込みの集計・列演算を優先。行処理なら`itertuples`も比べる |
| Polars | lazyな列変換・集約 | `scan_parquet`から式を組んで最後に実行。DuckDBと役割が重なるなら理由と測定が要る |
| Numba | 状態依存などで配列演算にしにくい数値ループ | JITの初回と定常を分ける。まず`fastmath=False`の逐次版で参照実装と比べる |

DuckDBはParquetの必要な列・条件をreaderへ押し込める。Polarsもlazyな計画で列・条件を絞り込む。
ただし効き方はファイルの配置、統計、クエリによるので、採用前に実行計画を確かめる。
[DuckDB Parquet](https://duckdb.org/docs/stable/data/parquet/overview)、
[Polars lazy最適化](https://docs.pola.rs/user-guide/lazy/optimizations/)。

pandasの`iterrows`は行ごとのdtypeを保たない場合がある。
[pandas公式](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html)。

新しい依存は隔離した環境で比べ、Python・ライブラリ・ネイティブ依存の版を記録する。
複数のエンジンを一度に入れない。

## 3. 保存形式と読み出し

### 3.1 用途別の形式

| 用途 | 形式の方針 | 保つ情報 |
|---|---|---|
| APIの原本・受け入れ済みの記録 | 今の不変形式を維持 | 原文・要求・観測時刻・hash |
| 小さなmanifest・実験条件 | JSON | 入力の集合、版、設定、同一性、品質状態 |
| 大きな派生時系列 | Parquetを候補として実測 | 明示した型、時刻、欠損、原本への対応 |
| 数値カーネルの内部 | NumPy配列＋mask | 軸の順、対象・時刻への対応、観測できたか |
| 人が確認する小さな出力 | CSV | 型・timezone・Nullの約束を別に書く。計算の主な中間形式にはしない |

Parquetは列指向なので、終値だけを使う研究でOHLCV全体を読まずに済む。
原本を置き換えるのではなく、原本のhashを辿れる派生buildとして作る。

### 3.2 分割・並べ方・圧縮

- 全対象の同時点比較なら時刻中心、単一対象の長期間走査なら対象中心の並べ方を比べる。
- 対象×日ごとの極小ファイルを大量に作ると、ファイル列挙とopenの負担が増える。
- row groupは読み出し単位・並列化・統計による読み飛ばしに関わる。万能なサイズはない。
- 圧縮はSnappyとZstandardなどで、書き込み時間・読み込み時間・bytesを同時に比べる。
- writerの版、codec、圧縮level、row group、partition、列順をbuild情報に残す。
- 日付は日付、時刻はtimezone付き、識別コードは文字列として明示する。
- 金額・数量はまず`float64`を保つ。`float32`への縮小は、順位・閾値・hashの変化を検証する別の提案にする。

[DuckDB Parquet tips](https://duckdb.org/docs/stable/data/parquet/tips)。

### 3.3 Arrowで必要な列だけをbatchで読む

```python
import pyarrow as pa
import pyarrow.dataset as ds
from collections.abc import Iterator

def scan_batches(
    path: str,
    columns: list[str],
    predicate: ds.Expression,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=columns, filter=predicate, batch_size=batch_size)
    yield from scanner.to_batches()
```

`batch_size`は行数であって、プロセス全体のメモリ上限ではない。先読み・並列処理・後続のsort/joinも測る。
全batchを`list(...)`に溜めると、逐次処理の利点が消える。
[Arrow Scanner](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html)。

### 3.4 欠損と同一性

疎な表で行を省いても、構成一覧、時刻軸、未観測の対象、品質で除外した件数と理由は別表かmanifestに残す。

Parquetのbytesのhashは輸送・破損の検査用、論理内容のhashは再生成の同一性用として分ける。
主キー・列順・浮動小数の表し方・Nullを正規化してから論理hashを取る。
形式を移すときは新旧の対応をmanifestに書く。JSONのhashとParquetのhashが同じになるとは期待しない。

## 4. ベクトル化とアルゴリズム

### 4.1 先に重複計算を減らす

| 遅くなりやすい処理 | 比べる改善 | 正しさの境界 |
|---|---|---|
| 各条件・試行で同じJSONをdecode | 検証済みbuildから必要な日・列を1回読む | 原本hashとbuildを固定 |
| 時刻ラベルを毎回線形探索 | 時刻→indexの辞書を1回作る | 重複時刻を上書きして隠さない |
| 各時点で窓を再集計 | 累積和＋有効件数、またはrunning sum | 日跨ぎ、欠損、窓端を保つ。差による丸め誤差を確かめる |
| 条件ごとに同じreturnを計算 | 同じ日・lookbackの中間結果を共有 | 時点境界と欠損方針が同じ場合だけ |
| 全順位が要らないのに全sort | 部分選択を比べる | 同順位の扱い・安定順・必要な順位が一致する場合だけ |

窓を単純に走査すると`O(T×W)`、累積和なら`O(T)`にできる場合がある。
ただし中央値や任意の状態遷移は同じ方法で置き換えられるとは限らない。

累積和の差は丸めの順が変わる。値の水準が大きいほど累積和の桁が大きくなり、差で失う桁も増える。
[rolling_window.py](examples/rolling_window.py)では、水準100前後の20万点で最大絶対誤差が約5e-10、
先に基準値100を引いてから累積すると約6e-13まで下がった。
許容誤差は比較の前に決め、結果を見てから広げない。

### 4.2 NumPyは配列演算を使う

`np.vectorize`は名前に反して基本的にPythonループであり、速くする目的では選ばない。
ufuncやbroadcastでまとめて計算する。
[NumPy vectorize](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html)、
[broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting)。

[adjacent_returns.py](examples/adjacent_returns.py)は、整列済みの`close[time, symbol]`と、
前後2点を繋いでよいかを示す`adjacent`から隣接リターンを計算する。
`adjacent`は読み出し側の品質検査の結果として受け取り、配列上で隣にあることから推定しない。
無効な値は数値buffer内だけNaNとし、maskも返す。保存するときはNaNをそのまま書かず、正式なNullと品質情報へ戻す。

### 4.3 broadcastingのメモリを見積もる

`T`時点×`S`対象のfloat64は値だけで`8TS` bytes。
`B`試行を同時に持つと`8BTS` bytesになり、mask・return・再構成系列・sortの作業領域がさらに加わる。
全試行を一度に展開せず、日単位・試行batch単位で処理して必要な集計だけ残す方法と比べる。
日で区切るなら、日を跨ぐ窓に必要な過去分を持ち越し、出力を二重に数えないようにする。

`out=`やsliceで一時配列を減らせる場合がある。高度なindexing、dtype変換、
DataFrame→NumPy→Python listの往復がcopyを作っていないか確かめる。
メモリを減らしたいなら、最終配列の`nbytes`ではなく、変換中を含むピークRSSを測る。

## 5. キャッシュ・並列化・乱数

### 5.1 キャッシュのキー

キーには、入力の原本・buildのhash、schema・計算の版、コードの同一性、全パラメータ、
対象・期間・timezone、時点境界、欠損・品質の方針を含める。乱数を使うなら生成器・seed・streamの規則も含める。
ファイル名・mtime・「最新のbuild」だけをキーにしない。
過去の時点に対して、現在の訂正済みデータや最新の対象一覧のキャッシュを返さない。

保存するキャッシュは、原本から再生成できる派生側に限る。
不一致はcache missとして再計算し、書きかけを完成したbuildとして公開しない。

### 5.2 並列化の順序

独立した日・試行を候補にする。同時点の断面を使う計算を対象ごとに分けて、意味を変えない。
Pythonの数値ループにはprocess、I/O待ちにはthreadを候補にし、エンジン内部の並列化も含めて測る。
worker数×DuckDB/BLASなどのthread数で過剰に並列化しないよう、全体のCPU・メモリ予算を決める。
巨大なPythonオブジェクトのprocess間転送が利得を消すなら、日単位の小さな入力を渡す。

worker数の上限はCPU数だけで決めず、親の保持量とworkerのピークRSSから求める。
batch size、worker数、thread数、メモリ予算、圧縮設定は設定ファイルへ分け、未確定のデフォルトには
`# TENTATIVE`を付ける。速くするために、実験のlookbackや試行数を勝手に削らない。

### 5.3 乱数と演算順

既存の実験が`random.Random`を日index・条件・試行から作っているなら、まずそのstream・消費順・共有範囲を保ったまま最適化する。
[bootstrap_counts.py](examples/bootstrap_counts.py)は、日の抽出を両版で同じ関数・同じ生成器にしている。
NumPyの乱数生成器に替える場合、同じ整数seedでも現行の結果と一致するとは扱わず、別の実装版として記録する。

新しく設計するなら、`SeedSequence`で独立したstreamを作る方法がある。workerの完了順ではなく、
固定した日・条件・試行IDにstreamを対応させる。実行ごとに変わり得るPythonの`hash()`をseedに使わない。
[NumPy並列乱数](https://numpy.org/doc/stable/reference/random/parallel.html)。

並列sum・累積和・累積積・JITは丸めの順を変える場合がある。浮動小数の許容誤差だけでなく、
順位・同順位・閾値判定・理由コードが一致するかを見る。速くするために許容誤差を後から広げない。

## 6. 計測と受け入れ

### 6.1 比較の条件を先に固定する

1. 合成入力で、小さな正常系・境界・失敗系を用意する。実データをtestsやGitに入れない。
2. 性能評価には固定したbuildの入力を使う。対象数、時点数、欠損率、条件数、試行数を記録する。
3. decode、検証、整列、数値計算、集約、hash、保存を分けて測り、全体の時間も測る。
4. 初回と再実行を分ける。OSのキャッシュを排除していない初回を、厳密なcold readとは呼ばない。JITのcompileも分ける。
5. 反復回数と採用基準を比較の前に決め、各回の値と中央値・範囲を残す。期待する倍率を測定結果として書かない。

CPUの調査には`cProfile`、区間の時間には`time.perf_counter()`を使う。
macOSでは`/usr/bin/time -l`でプロセス単位のピークRSSを測れる(単位を明記する)。
`tracemalloc`はnativeな確保を含まないので、全RSSの代わりにならない。
multiprocessingでは、親だけのRSSを全体として扱わない。

### 6.2 記録テンプレート

| 項目 | 記録すること |
|---|---|
| 入力と意味の約束 | build/原本のhash、schema、時点境界、品質方針、対象範囲 |
| コードと環境 | commitと未commit差分の同一性、Python、依存、OS、CPU、RAM |
| 実行設定 | thread、worker、batch、codec、seed/stream、全実験条件 |
| 速度 | 初回/再実行、工程別の秒数、全体の秒数、反復値 |
| メモリ・保存 | 測定方法(親/子)、ピークRSS、入出力bytes、一時領域 |
| 正しさ | 論理hash、差分件数、最大誤差、判定/理由コードの一致、失敗ケース |
| 結論 | 採用/棄却/保留、改善した対象、回帰した点、未確認のこと |

### 6.3 必ず確かめること

- 同じ入力から2回生成して、正規化した論理内容のhashが一致する。
- 時点境界の直前・一致・直後、訂正の前後、対象一覧が変わる境界を保つ。
- 未知の種別、同時刻の重複、Nullの係数などを、最適化で黙って通さない。
- 空入力、全部欠損の対象、途切れた時刻、日跨ぎ、窓不足、同順位、極端な数値を比べる。
- chunk境界とworker数を変えても、母集団・分母・欠損・乱数stream・結果の順が変わらない。
- golden結果と失敗経路を確かめる。既存の研究結果を変更版で上書きしない。

新旧の浮動小数が違うなら、同じhashだと主張しない。厳密に一致する実装を探すか、
差を説明した版変更として扱う。同じ版の再実行は、引き続き再現できなければならない。

## 7. 実装する順番

| 段階 | 成果物 | 次へ進む条件 |
|---|---|---|
| 1. 測る | 固定入力・工程別の時間・ピークRSS・参照結果 | 支配的な工程がわかった |
| 2. 読む量と再計算を減らす | 逐次処理、必要な列、index、中間集計の再利用 | 正しさと全体の改善を確かめた |
| 3. 派生保存を比べる | Parquet＋DuckDB/Arrowの小さな試作 | 再生成・型・欠損・時点境界・容量を確かめた |
| 4. 数値部分を配列化する | NumPyカーネルと参照実装との差の記録 | 変換コスト込みで改善し、結果の約束を保った |
| 5. 残った計算を速くする | 必要ならNumba・試行batch・並列化 | RSS予算・乱数・演算順・再現性を確かめた |

GPUは、CPU側の重複計算削減とNumPy/Numbaを基準にし、大量の独立した配列演算で比べる。
速度だけでなく、CPU参照に対する絶対・相対誤差、順位・閾値・事象・最終結果の差を必須のbenchmarkにする。
比較の方法・採否・報告の形は[計算計画エージェントへの指示 §6.1・§8](PLANNING_PROMPT.md)に従う。
低精度化は自動で採用しない。非同期・遅延評価は処理の完了まで測り、初期化・変換・同期を含めた全体時間で判断する。
分散処理や複数エンジンの併用は、単一マシンの代表的な処理で不足が測定されてから比べる。
依存を足しただけで速くなったと扱わず、採用・棄却・未決と測定条件を同じ記録に残す。
