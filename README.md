以下に、今までのやり取りを基にしたGitHubのREADME向けのMarkdown文書をまとめました。

---

# 無線信号スペクトラム表示アプリケーション

このアプリケーションは、StreamlitとPlotlyを使用して、無線信号のスペクトラムをリアルタイムで表示するデモプログラムです。指定された周波数帯域に対して、ランダムに生成されたFFTデータを用いてスペクトラムを表示します。

## 機能

- **リアルタイム表示**: 100msごとに信号のスペクトラムが更新されます。
- **カスタムキャリア生成**:
  - 130MHz付近に-18dBmから-20dBmで、帯域幅6kHzのキャリアを常時生成します。
  - 120MHz付近に-40dBmから-60dBmの帯域幅25kHzのキャリアを生成します。
  - その他の周波数帯域では、-80dBmから-100dBmのノイズが生成されます。
- **周波数範囲**: 118MHzから138MHzの間で表示します。
- **Y軸スケール**: グラフのY軸スケールは-110dBmから-10dBmに設定されています。

## 使用技術

- [Streamlit](https://streamlit.io/): インタラクティブなウェブアプリケーションフレームワーク。
- [Plotly](https://plotly.com/python/): 高度なグラフ描画ライブラリ。

## 実行方法

このアプリケーションをローカルで実行するには、以下の手順を実行してください。

1. **必要なライブラリのインストール**:
    ```bash
    pip install streamlit plotly numpy
    ```

2. **アプリケーションの実行**:
    ```bash
    streamlit run app.py
    ```

3. **ブラウザでの表示**:
    コマンドを実行すると、自動的にブラウザが開き、無線信号のスペクトラムが表示されます。

## コード

以下は、このアプリケーションの主要なコードです。

```python
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import time

# カスタムFFTスペクトラムデータを生成
def generate_custom_fft_data(num_points):
    fft_freq = np.linspace(118e6, 138e6, num_points)
    fft_power_dBm = np.random.uniform(-100, -80, num_points)

    carrier_130MHz_start = 130e6 - 3e3
    carrier_130MHz_end = 130e6 + 3e3
    carrier_130MHz_indices = np.where((fft_freq >= carrier_130MHz_start) & (fft_freq <= carrier_130MHz_end))[0]
    if len(carrier_130MHz_indices) > 0:
        fft_power_dBm[carrier_130MHz_indices] = np.random.uniform(-20, -18, len(carrier_130MHz_indices))

    carrier_120MHz_start = 120e6 - 12.5e3
    carrier_120MHz_end = 120e6 + 12.5e3
    carrier_120MHz_indices = np.where((fft_freq >= carrier_120MHz_start) & (fft_freq <= carrier_120MHz_end))[0]
    if len(carrier_120MHz_indices) > 0:
        fft_power_dBm[carrier_120MHz_indices] = np.random.uniform(-60, -40, len(carrier_120MHz_indices))

    return fft_freq, fft_power_dBm

# Streamlitアプリのタイトル
st.title("無線信号スペクトラム表示アプリ（dBm、カスタムデータ）")

placeholder_freq = st.empty()

sampling_rate = 30e6  # 30 MHz
num_points = 4096

while True:
    fft_freq, fft_power_dBm = generate_custom_fft_data(num_points)

    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=fft_freq, y=fft_power_dBm, mode='lines', name='FFT Power (dBm)'))
    fig_freq.update_layout(title="無線信号のスペクトラム (周波数ドメイン, dBm)",
                           xaxis_title="周波数 (Hz)",
                           yaxis_title="振幅 (dBm)",
                           xaxis=dict(range=[min(fft_freq), max(fft_freq)]),
                           yaxis=dict(range=[-110, -10]))

    placeholder_freq.plotly_chart(fig_freq)

    time.sleep(0.1)  # 100ms待機して更新
```

## カスタマイズ

- **周波数範囲**や**キャリアの強度**、**帯域幅**などはコード内で簡単に調整可能です。
- データ生成部分を実際のFFT計算結果に差し替えることで、実際の信号データを表示することも可能です。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

このREADMEをプロジェクトの`README.md`ファイルとして使用することで、他の開発者やユーザーに対してアプリケーションの概要と使用方法を明確に伝えることができます。
