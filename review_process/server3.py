import streamlit as st
import numpy as np
import plotly.graph_objs as go
import time

# 無線信号のサンプルデータを生成
def generate_random_signal_segment(sampling_rate, duration=0.1):
    t = np.arange(0, duration, 1/sampling_rate)
    frequency = np.random.uniform(1, 50)  # 1Hzから50Hzのランダムな周波数
    amplitude = np.random.uniform(0.1, 1.0)  # 0.1から1.0のランダムな振幅
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, signal

# FFTを使用して周波数成分を抽出
def compute_fft(signal, sampling_rate):
    N = len(signal)
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(N, d=1/sampling_rate)
    return fft_freq[:N//2], np.abs(fft_result)[:N//2]  # 正の周波数成分のみ返す

# Streamlitアプリのタイトル
st.title("無線信号波形表示アプリ（ランダム周波数・振幅、リアルタイム更新）")

# ユーザー入力
duration = st.slider("表示時間 (秒)", 1, 10, 5)
sampling_rate = st.slider("サンプリングレート (Hz)", 100, 10000, 1000)

# グラフを表示するための空コンテナを作成
placeholder_time = st.empty()
placeholder_freq = st.empty()

# 信号の継続的な更新
full_signal = np.array([])  # 全体の信号データを格納

for _ in range(int(duration * 10)):  # 表示時間の間ループ（100ms毎に更新）
    t, signal_segment = generate_random_signal_segment(sampling_rate)
    full_signal = np.concatenate([full_signal, signal_segment])

    # 周波数成分の計算
    fft_freq, fft_magnitude = compute_fft(full_signal, sampling_rate)
    
    # 時間ドメインのグラフを更新
    #fig_time = go.Figure()
    #fig_time.add_trace(go.Scatter(x=np.linspace(0, len(full_signal)/sampling_rate, len(full_signal)), y=full_signal, mode='lines', name='Signal'))
    #fig_time.update_layout(title="無線信号の波形 (時間ドメイン)",
    #                       xaxis_title="時間 (秒)",
    #                       yaxis_title="振幅")

    # 周波数ドメインのグラフを更新
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=fft_freq, y=fft_magnitude, mode='lines', name='FFT Magnitude'))
    fig_freq.update_layout(title="無線信号のスペクトラム (周波数ドメイン)",
                           xaxis_title="周波数 (Hz)",
                           yaxis_title="振幅",
                           xaxis=dict(range=[0, max(fft_freq)]))

    # グラフを再描画
    #placeholder_time.plotly_chart(fig_time)
    placeholder_freq.plotly_chart(fig_freq)

    time.sleep(0.1)  # 100ms待機して更新
