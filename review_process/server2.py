import streamlit as st
import numpy as np
import plotly.graph_objs as go

# 無線信号のサンプルデータを生成
def generate_random_signal(duration, sampling_rate):
    t = np.arange(0, duration, 1/sampling_rate)
    signal = np.zeros_like(t)
    
    for i in range(len(t)):
        if i % (sampling_rate // 10) == 0:  # 100msごとに周波数と振幅を変える
            frequency = np.random.uniform(1, 50)  # 1Hzから50Hzのランダムな周波数
            amplitude = np.random.uniform(0.1, 1.0)  # 0.1から1.0のランダムな振幅
        
        signal[i] = amplitude * np.sin(2 * np.pi * frequency * t[i])
    
    return t, signal

# FFTを使用して周波数成分を抽出
def compute_fft(signal, sampling_rate):
    N = len(signal)
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(N, d=1/sampling_rate)
    return fft_freq[:N//2], np.abs(fft_result)[:N//2]  # 正の周波数成分のみ返す

# Streamlitアプリのタイトル
st.title("無線信号波形表示アプリ（ランダム周波数・振幅）")

# ユーザー入力
duration = st.slider("持続時間 (秒)", 1, 10, 5)
sampling_rate = st.slider("サンプリングレート (Hz)", 100, 10000, 1000)

# 信号の生成
t, signal = generate_random_signal(duration, sampling_rate)

# 周波数成分の計算
fft_freq, fft_magnitude = compute_fft(signal, sampling_rate)

# Plotlyを使用して波形を描画
fig_time = go.Figure()
fig_time.add_trace(go.Scatter(x=t, y=signal, mode='lines', name='Signal'))
fig_time.update_layout(title="無線信号の波形 (時間ドメイン)",
                       xaxis_title="時間 (秒)",
                       yaxis_title="振幅")

# Plotlyを使用して周波数スペクトラムを描画
fig_freq = go.Figure()
fig_freq.add_trace(go.Scatter(x=fft_freq, y=fft_magnitude, mode='lines', name='FFT Magnitude'))
fig_freq.update_layout(title="無線信号のスペクトラム (周波数ドメイン)",
                       xaxis_title="周波数 (Hz)",
                       yaxis_title="振幅",
                       xaxis=dict(range=[0, max(fft_freq)]))

# グラフを表示
st.plotly_chart(fig_time)
st.plotly_chart(fig_freq)
