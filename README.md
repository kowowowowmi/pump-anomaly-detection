# ポンプセンサー異常検知

工業用ポンプの多変量センサーデータから、教師なし学習で異常状態（故障・回復中）を検知するパイプラインの実装・比較研究。

---

## 問題設定

| 項目 | 内容 |
|---|---|
| データ | Kaggle「Pump Sensor Data」|
| 規模 | 220,320行 × 54列（1分間隔、約5ヶ月分）|
| センサー数 | 52チャンネル（sensor_00〜sensor_51）|
| ラベル | `NORMAL` 93.4% / `RECOVERING` 6.6% / `BROKEN` 0.003%（7行）|

`BROKEN` がわずか7行という極端なクラス不均衡により、教師あり分類は非現実的。**NORMAL期間のみで学習し、全期間に適用して復元誤差や異常スコアを計算する異常検知アプローチ**を採用した。

---

## 手法と結果

| 手法 | Precision | Recall | F1 |
|---|---|---|---|
| ARIMA 4センサー OR統合 | 0.85 | 0.94 | 0.89 |
| Isolation Forest（contamination=0.015） | 0.82 | 0.98 | **0.89** |
| ARIMA + IF アンサンブル（OR） | 0.78 | **1.00** | 0.87 |
| LSTM-AE（20エポック、95パーセンタイル閾値） | 0.53 | 0.81 | 0.65 |

ARIMA と Isolation Forest がF1 0.89で同等の最高性能。アンサンブル（OR）はRecall 1.00を達成するが、OR統合の構造上Precisionが低下するトレードオフがある。

---

## 実装内容

### 01_eda.ipynb — 探索的データ分析

- 52センサーの欠損・分布・時系列を可視化
- `sensor_15`：全行欠損のためドロップ
- ACF解析で日周期性なし → STL分解は不採用
- センサー感度ランキング（変動比分析）：**sensor_42, 41, 39, 04** を一軍として選定

### 02_modeling.ipynb — ARIMAによる単一センサー異常検知

- ADF検定で定常性を確認（p値≈0.000）
- PACF でラグ1後カットオフ → **ARIMA(1, 0, 0)** を採用（φ≈0.986）
- NORMAL期間のみで学習し、全期間に残差を計算
- 2種類の検知シグナルを設計：
  - **シグナルA（スパイク検知）**：`|残差| > k × σ_normal`
  - **シグナルB（固着検知）**：`rolling_std(残差, window=50) < k × σ_normal`
    - φ≈1の系列は異常水準に速やかに適応するため、残差の絶対値ではなく「残差分散の異常な小ささ」で RECOVERING を捉える

### 03_modeling_advanced.ipynb — 4センサー統合 + Isolation Forest + アンサンブル

- 選定センサー：`sensor_04, 39, 41, 42`
- ARIMA: 4センサーの検知結果をOR統合 → Recall 0.86→0.94に向上
- Isolation Forest: `X_normal`でfit → 全期間でpredict（`contamination=0.015`）
- ARIMA + IF アンサンブル（OR）: Recall 1.00達成

### 04_lstm_ae.ipynb — LSTM-AE（PyTorch）

- Encoder-Decoder構造のLSTMオートエンコーダーをゼロから実装
- スライディングウィンドウ（window_size=60、stride=1）でシーケンスを整形
- NORMALデータのみで学習 → 全期間の復元誤差を計算
- 閾値：`mean + k×std` では NORMAL/RECOVERING 境界付近の遷移窓が std を膨らませ Recall が低下 → **95パーセンタイル閾値**に切り替えて改善

---

## 環境

```bash
conda activate p2-anomaly  # Python 3.11
```

主要ライブラリ：`pandas` / `numpy` / `matplotlib` / `scikit-learn` / `statsmodels` / `torch`

---

## ディレクトリ構成

```
project2_pump_anomaly_detection/
├── data/
│   └── raw/
│       └── sensor.csv          # Kaggle: Pump Sensor Data
├── notebooks/
│   ├── 01_eda.ipynb            # EDA・センサー選定
│   ├── 02_modeling.ipynb       # ARIMA 単一センサー
│   ├── 03_modeling_advanced.ipynb  # 4センサー + IF + アンサンブル
│   └── 04_lstm_ae.ipynb        # LSTM-AE（PyTorch）
└── src/
```

---

## 設計上の気づき

- **クラス不均衡と手法選択**：BROKEN=7行では教師あり学習が成立しない。ラベルなしで NORMAL の特性を学習し、逸脱を検知する発想が前提になる。
- **ARIMAの限界と固着検知**：φ≈1の系列は異常水準に素早く順応するため、残差の絶対値では RECOVERING を取りこぼす。残差分散（シグナルB）で補完することで Recall が改善した。
- **閾値設計の外れ値耐性**：`mean + k×std` はデータの外れ値に引きずられる。パーセンタイル閾値は分布の形状に依存せず安定する。
- **OR/AND統合のトレードオフ**：OR統合は構造上 Recall↑・Precision↓。AND統合はその逆。アンサンブルの結合方法は目的（見逃しを防ぐ vs 誤検知を減らす）に合わせて選択する。
