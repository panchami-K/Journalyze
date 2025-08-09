# src/data_preprocess/load_data.py
import pandas as pd
from pathlib import Path
import yaml

def load_yaml_config(path='src/config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_journal_data(config):
    data_dir = Path(config['data']['raw_data_dir'])
    filenames = config['data']['filenames']
    frames = []
    for fname in filenames:
        file_path = data_dir / fname
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found.")
        df = pd.read_csv(file_path)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    return df_all
