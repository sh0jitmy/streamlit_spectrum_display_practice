import streamlit as st
import numpy as np
import plotly.graph_objs as go
import time

# ランダムなFFTスペクトラムデータを生成
def generate_random_fft_data(num_points):
    fft_freq = np.linspace(0, 15e6, num_points)  # 0Hzから15MHzの範囲
    fft_power_dBm = np.random.uniform(-100, 0, num_points)  # -100dBmから0dBmのランダムな振幅
    return fft_freq, fft_power_dBm

# Streamlitアプリのタイトル
st.title("無線信号スペクトラム表示アプリ（dBm、ランダムデータ）")

# グラフを表示するための空コンテナを作成
placeholder_freq = st.empty()

# サンプリングレート設定
sampling_rate = 30e6  # 30 MHz

# スペクトラムのポイント数設定
num_points = 1024  # FFTポイント数の代わりに使用

# 信号の継続的な更新
while True:  # 無限ループで信号を更新
    fft_freq, fft_power_dBm = generate_random_fft_data(num_points)

    # 周波数ドメインのグラフを更新
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=fft_freq, y=fft_power_dBm, mode='lines', name='FFT Power (dBm)'))
    fig_freq.update_layout(title="無線信号のスペクトラム (周波数ドメイン, dBm)",
                           xaxis_title="周波数 (Hz)",
                           yaxis_title="振幅 (dBm)",
                           xaxis=dict(range=[0, max(fft_freq)]))

    # グラフを再描画
    placeholder_freq.plotly_chart(fig_freq)

    time.sleep(0.1)  # 100ms待機して更新
