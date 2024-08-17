import streamlit as st
import numpy as np
import plotly.graph_objs as go

# 無線信号のサンプルデータを生成
def generate_signal(frequency, amplitude, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, signal

# Streamlitアプリのタイトル
st.title("無線信号波形表示アプリ")

# ユーザー入力
frequency = st.slider("周波数 (Hz)", 1, 100, 10)
amplitude = st.slider("振幅", 0.1, 1.0, 0.5)
duration = st.slider("持続時間 (秒)", 1, 10, 5)
sampling_rate = st.slider("サンプリングレート (Hz)", 100, 10000, 1000)

# 信号の生成
t, signal = generate_signal(frequency, amplitude, duration, sampling_rate)

# Plotlyを使用して波形を描画
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=signal, mode='lines', name='Signal'))
fig.update_layout(title="無線信号の波形",
                  xaxis_title="時間 (秒)",
                  yaxis_title="振幅",
                  xaxis=dict(range=[0, duration]),
                  yaxis=dict(range=[-amplitude, amplitude]))

# グラフを表示
st.plotly_chart(fig)
