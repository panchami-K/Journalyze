# src/insights/period_summary.py

import pandas as pd

def period_insight_summary(df, date_col='date', freq='M'):
    # Assumes date_col is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Group by period
    period_groups = df.groupby(pd.Grouper(key=date_col, freq=freq))

    summaries = []
    for period, group in period_groups:
        if group.empty:
            continue

        # Dominant emotions & distortions
        top_emotion = group['emotion'].value_counts().idxmax() if 'emotion' in group else None
        avg_distortion = group['distortion_count'].mean() if 'distortion_count' in group else None
        common_distortions = pd.Series(sum(group['detected_distortions'], [])).value_counts().head(2).to_dict()
        # Watch for empty lists
        entry_count = len(group)

        # Build summary
        summary = {
            'period': period.strftime('%Y-%m') if freq=='M' else str(period),
            'entry_count': entry_count,
            'top_emotion': top_emotion,
            'avg_distortion_count': round(avg_distortion, 2) if avg_distortion is not None else None,
            'common_distortions': common_distortions,
        }
        summaries.append(summary)

    return pd.DataFrame(summaries)

def print_period_summaries(df_summary):
    print("\n=== PERIODIC INSIGHT SUMMARY ===")
    for idx, row in df_summary.iterrows():
        print(
            f"Period: {row['period']} | Entries: {row['entry_count']} | "
            f"Top Emotion: {row['top_emotion']} | Avg Distortion: {row['avg_distortion_count']} | "
            f"Distortions: {row['common_distortions']}"
        )
