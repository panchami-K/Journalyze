# src/eda/perform_eda.py
import pandas as pd

def basic_eda_report(df):
    report = {}

    # General shape
    report['shape'] = df.shape

    # Data types
    report['dtypes'] = df.dtypes.to_dict()

    # Missing values
    report['missing_values'] = df.isnull().sum().to_dict()

    # Descriptive stats (numerical)
    report['describe'] = df.describe(include='all').to_dict()

    # Top value counts for each column
    report['value_counts'] = {}
    for col in df.columns:
        if df[col].nunique() < 20:  # Only for columns with reasonable cardinality
            report['value_counts'][col] = df[col].value_counts().to_dict()

    # Unique values for key columns
    report['unique_emotions'] = df['emotion'].unique().tolist() if 'emotion' in df.columns else []
    report['unique_bias'] = df['bias/distortion'].unique().tolist() if 'bias/distortion' in df.columns else []

    # Average text length
    if 'text' in df.columns:
        report['avg_text_length'] = df['text'].astype(str).apply(len).mean()

    return report

def print_eda_report(report):
    print(f"Dataset shape: {report['shape']}")
    print(f"Column types: {report['dtypes']}")
    print(f"Missing values: {report['missing_values']}")
    print("-------\nExamples of statistical summary:")
    for k,v in (report['describe'].items()):
        print(f" • {k} : {str(v)[:120]}")
    print("\nUnique emotions: ", report['unique_emotions'])
    print("Unique biases/distortions: ", report['unique_bias'])
    print(f"Avg text length: {report.get('avg_text_length', 'n/a'):.2f}")
    print("\nSample value counts (selected columns):")
    for k, v in report['value_counts'].items():
        print(f" - {k}: {v}")
