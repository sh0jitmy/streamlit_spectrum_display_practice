import streamlit as st
import numpy as np
import plotly.graph_objs as go
import time

# 無線信号のサンプルデータを生成
def generate_random_signal_segment(sampling_rate, duration=0.1):
    t = np.arange(0, duration, 1/sampling_rate)
    frequency = np.random.uniform(1e6, 15e6)  # 1MHzから15MHzのランダムな周波数
    amplitude = np.random.uniform(0.1, 1.0)  # 0.1から1.0のランダムな振幅
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal

# FFTを使用して周波数成分を抽出
def compute_fft(signal, sampling_rate):
    N = len(signal)
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(N, d=1/sampling_rate)
    return fft_freq[:N//2], np.abs(fft_result)[:N//2]  # 正の周波数成分のみ返す

# Streamlitアプリのタイトル
st.title("無線信号スペクトラム表示アプリ（ランダム周波数・振幅、リアルタイム更新）")

# グラフを表示するための空コンテナを作成
placeholder_freq = st.empty()

# サンプリングレート設定
sampling_rate = 30e6  # 30 MHz

# 信号の継続的な更新
full_signal = np.array([])  # 全体の信号データを格納

while True:  # 無限ループで信号を更新
    signal_segment = generate_random_signal_segment(sampling_rate)
    full_signal = np.concatenate([full_signal, signal_segment])

    # 周波数成分の計算
    fft_freq, fft_magnitude = compute_fft(full_signal, sampling_rate)

    # 周波数ドメインのグラフを更新
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=fft_freq, y=fft_magnitude, mode='lines', name='FFT Magnitude'))
    fig_freq.update_layout(title="無線信号のスペクトラム (周波数ドメイン)",
                           xaxis_title="周波数 (Hz)",
                           yaxis_title="振幅",
                           xaxis=dict(range=[0, max(fft_freq)]))

    # グラフを再描画
    placeholder_freq.plotly_chart(fig_freq)

    time.sleep(0.1)  # 100ms待機して更新
