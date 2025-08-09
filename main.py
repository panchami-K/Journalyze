import pandas as pd
import os

# === Config and Data Pipeline ===
from src.data_preprocess.load_data import load_journal_data, load_yaml_config
from src.data_preprocess.preprocess import preprocess_dataframe

# === Exploratory Data Analysis ===
from src.eda.perform_eda import basic_eda_report, print_eda_report

# === Feature Engineering ===
from src.features.feature_engineering import feature_engineering_pipeline
from src.features.pattern_detection import detect_recurring_patterns, print_pattern_report

# === Advanced Features: Traits, Quirks, Peer Groups ===
from src.features.personality_traits import add_big5_traits
from src.features.quirk_detection import detect_journal_quirks, summarize_quirks
from src.features.norming import peer_group_clusters

# === Insights and Summaries ===
from src.insights.psychological_inference import psychological_inference, user_insight_report, print_user_insight_report
from src.insights.period_summary import period_insight_summary, print_period_summaries
from src.insights.period_feedback import attach_period_feedback, print_period_feedback

# === Visualization ===
from src.visualization.progress_trends import plot_emotion_trend, plot_distortion_trend

# === Saving Utility ===
def save_processed_data(df: pd.DataFrame, config: dict) -> None:
    output_path = config['data']['processed_data_dir']
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "cleaned_journal_features.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✅ Processed data with features saved to: {output_file}")
    print(f"Columns in output: {df.columns.tolist()}")

# === Main Pipeline ===
def main() -> None:
    print("🚀 Starting Journal Analysis Pipeline...\n")

    # Step 1: Load config and data
    config = load_yaml_config()
    df = load_journal_data(config)
    print(f"📄 Raw data loaded. Shape: {df.shape}")

    # Step 2: Preprocess text
    df_clean = preprocess_dataframe(df, config)
    print("\n🧼 Preprocessing complete.")

    # Step 3: EDA
    report = basic_eda_report(df_clean)
    print_eda_report(report)

    # Step 4: Core Feature Engineering
    df_features = feature_engineering_pipeline(df_clean, config)
    print("🛠️ Feature engineering complete.")

    # Step 5: Add date if needed
    df_features['date'] = pd.date_range(start='2024-01-01', periods=len(df_features), freq='D')

    # Step 6: Personality Traits, Quirks, Peer Groups
    df_features = add_big5_traits(df_features, text_col='text_clean')
    df_features, quirk_model = detect_journal_quirks(df_features, text_col='text_clean', n_clusters=5)
    print("\n=== Quirk/Recurring Pattern Example Summaries ===")
    for cluster, samples in summarize_quirks(df_features).items():
        print(f"Quirk Cluster {cluster}: {samples}")
    trait_cols = [c for c in df_features.columns if c.startswith("big5_")]
    df_features, peer_model = peer_group_clusters(df_features, feature_cols=trait_cols, n_clusters=3)
    print("Peer group assignments (first 10):", df_features['peer_group'].head(10).tolist())

    # Step 7: Psychological Insight
    df_features = psychological_inference(df_features, text_col='text_clean')
    insight_summary = user_insight_report(df_features)
    print_user_insight_report(insight_summary)

    # Step 8: Pattern Detection (themes, loops, triggers, etc.)
    pattern_report = detect_recurring_patterns(df_features, config)
    print_pattern_report(pattern_report)

    # Step 9: Evolution and Feedback
    period_summary_df = period_insight_summary(df_features, date_col='date', freq='M')
    print_period_summaries(period_summary_df)
    period_feedback_df = attach_period_feedback(period_summary_df)
    print_period_feedback(period_feedback_df)

    # Step 10: Visualization (optional/interactive)
    plot_emotion_trend(df_features, date_col='date', emotion_col='emotion')
    plot_distortion_trend(df_features, date_col='date', distortion_col='distortion_count')

    # Step 11: Persist full data
    save_processed_data(df_features, config)

# === Entry Point ===
if __name__ == "__main__":
    main()
