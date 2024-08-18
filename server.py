import streamlit as st
import numpy as np
import plotly.graph_objs as go
import time

# カスタムFFTスペクトラムデータを生成
def generate_custom_fft_data(num_points):
    # 周波数範囲を118MHzから138MHzに設定
    fft_freq = np.linspace(118e6, 138e6, num_points)
    fft_power_dBm = np.random.uniform(-100, -94, num_points)  # その他の周波数帯域のノイズ

    # 130MHzに-18dBmから-20dBmで、帯域幅6kHzのキャリアを生成
    carrier_130MHz_start = 130e6 - 3e3
    carrier_130MHz_end = 130e6 + 3e3
    carrier_130MHz_indices = np.where((fft_freq >= carrier_130MHz_start) & (fft_freq <= carrier_130MHz_end))[0]
    if len(carrier_130MHz_indices) > 0:
        fft_power_dBm[carrier_130MHz_indices] = np.random.uniform(-20, -18, len(carrier_130MHz_indices))

    # 120MHzに-40dBmから-60dBmの25kHzのキャリアを生成
    carrier_120MHz_start = 120e6 - 12.5e3
    carrier_120MHz_end = 120e6 + 12.5e3
    carrier_120MHz_indices = np.where((fft_freq >= carrier_120MHz_start) & (fft_freq <= carrier_120MHz_end))[0]
    if len(carrier_120MHz_indices) > 0:
        fft_power_dBm[carrier_120MHz_indices] = np.random.uniform(-60, -40, len(carrier_120MHz_indices))

    return fft_freq, fft_power_dBm

# Streamlitアプリのタイトル
st.title("無線信号スペクトラム表示アプリ（dBm、カスタムデータ）")

# グラフを表示するための空コンテナを作成
placeholder_freq = st.empty()

# サンプリングレート設定
sampling_rate = 30e6  # 30 MHz

# スペクトラムのポイント数設定
num_points = 4096  # FFTポイント数の代わりに使用。ポイント数を増加。

# 信号の継続的な更新
while True:  # 無限ループで信号を更新
    fft_freq, fft_power_dBm = generate_custom_fft_data(num_points)

    # 周波数ドメインのグラフを更新
    fig_freq = go.Figure()
    #fig_freq.add_trace(go.Scatter(x=fft_freq, y=fft_power_dBm, mode='lines', name='FFT Power (dBm)'))
    fig_freq.add_trace(go.Scatter(x=fft_freq, y=fft_power_dBm, mode='lines', name='FFT Power (dBm)',line=dict(color="yellow")))
    fig_freq.update_layout(title="無線信号のスペクトラム (周波数ドメイン, dBm)",
                           xaxis_title="周波数 (Hz)",
                           yaxis_title="振幅 (dBm)",
                           xaxis=dict(range=[min(fft_freq), max(fft_freq)]),
                           yaxis=dict(range=[-110, -10]), # Y軸スケールの設定
                           plot_bgcolor="black") # 背景色黒
    # グラフを再描画
    placeholder_freq.plotly_chart(fig_freq)

    time.sleep(0.1)  # 100ms待機して更新
