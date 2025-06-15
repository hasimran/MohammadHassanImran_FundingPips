import pandas as pd
import logging
from metrics import compute_trader_metrics

logging.basicConfig(filename="Log/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(filepath):
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        raise

def clean_data(df):
    logging.info("Cleaning data...")
    df['opened_at'] = pd.to_datetime(df['opened_at'], errors='coerce')
    df['closed_at'] = pd.to_datetime(df['closed_at'], errors='coerce')
    df['trade_duration_sec'] = (df['closed_at'] - df['opened_at']).dt.total_seconds()
    df['is_profitable'] = df['profit'] > 0
    logging.info("Cleaning complete.")
    return df
    
if __name__ == "__main__":
    file_path = "Data/test_task_trades.csv"
    df = load_data(file_path)
    df = clean_data(df)
    metrics = compute_trader_metrics(df)
    print(metrics)