# src/visualization/progress_trends.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_emotion_trend(df, date_col='date', emotion_col='emotion', save_path=None):
    """
    Plots the frequency of top emotions over time.
    """
    if date_col not in df.columns:
        print(f"[Warning] No '{date_col}' column found. Trend visualization skipped.")
        return

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])
    top_emotions = df[emotion_col].value_counts().head(5).index.tolist()
    df_plot = df[df[emotion_col].isin(top_emotions)]

    # Group by date and emotion
    emotion_over_time = df_plot.groupby([date_col, emotion_col]).size().unstack(fill_value=0)
    emotion_over_time.rolling(window=7, min_periods=1).mean().plot(figsize=(12,6))
    plt.title('Top Emotions Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Frequency (7-Day Rolling Average)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_distortion_trend(df, date_col='date', distortion_col='distortion_count', save_path=None):
    """
    Plots count of detected distortions over time.
    """
    if date_col not in df.columns:
        print(f"[Warning] No '{date_col}' column found. Trend visualization skipped.")
        return
    df[date_col] = pd.to_datetime(df[date_col])
    distortion_by_date = df.groupby(date_col)[distortion_col].sum()
    distortion_by_date.rolling(window=7, min_periods=1).mean().plot(figsize=(12,6))
    plt.title('Cognitive Distortion Intensity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Distortion Count (7-Day Rolling Avg)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
