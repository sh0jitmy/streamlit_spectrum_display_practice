import streamlit as st
import numpy as np
import plotly.graph_objs as go
import time

# ランダムなFFTスペクトラムデータを生成
def generate_custom_fft_data(num_points):
    # 周波数範囲を115MHzから145MHzに設定
    fft_freq = np.linspace(115e6, 145e6, num_points)
    fft_power_dBm = np.random.uniform(-100, -80, num_points)  # その他の周波数帯域のノイズ

    # 130MHzに-18dBmから-20dBmで、帯域幅6kHzのキャリアを生成
    carrier_130MHz = np.logical_and(fft_freq >= 130e6 - 3e3, fft_freq <= 130e6 + 3e3)
    fft_power_dBm[carrier_130MHz] = np.random.uniform(-20, -18, carrier_130MHz.sum())

    # 120MHzに-40dBmから-60dBmの25kHzのキャリアを生成
    carrier_120MHz = np.logical_and(fft_freq >= 120e6 - 12.5e3, fft_freq <= 120e6 + 12.5e3)
    fft_power_dBm[carrier_120MHz] = np.random.uniform(-60, -40, carrier_120MHz.sum())

    return fft_freq, fft_power_dBm

# Streamlitアプリのタイトル
st.title("無線信号スペクトラム表示アプリ（dBm、カスタムデータ）")

# グラフを表示するための空コンテナを作成
placeholder_freq = st.empty()

# サンプリングレート設定
sampling_rate = 30e6  # 30 MHz

# スペクトラムのポイント数設定
num_points = 1024  # FFTポイント数の代わりに使用

# 信号の継続的な更新
while True:  # 無限ループで信号を更新
    fft_freq, fft_power_dBm = generate_custom_fft_data(num_points)

    # 周波数ドメインのグラフを更新
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=fft_freq, y=fft_power_dBm, mode='lines', name='FFT Power (dBm)'))
    fig_freq.update_layout(title="無線信号のスペクトラム (周波数ドメイン, dBm)",
                           xaxis_title="周波数 (Hz)",
                           yaxis_title="振幅 (dBm)",
                           xaxis=dict(range=[min(fft_freq), max(fft_freq)]),
                           yaxis=dict(range=[-110, -10]))  # Y軸スケールの設定

    # グラフを再描画
    placeholder_freq.plotly_chart(fig_freq)

    time.sleep(0.1)  # 100ms待機して更新
