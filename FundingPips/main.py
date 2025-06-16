import pandas as pd
import logging
from metrics import compute_trader_metrics
from metrics import analyze_top_traders
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

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
    
    # Load/clean data
    df = load_data(file_path)
    df = clean_data(df)
    
    # Compute metrics  
    metrics = compute_trader_metrics(df)
    print(metrics)
    logging.info("Computing metrics complete.")
    
    # Analyze top traders
    analyze_top_traders(df, metrics, top_n=10)
    logging.info("Analysis of top traders complete.")
    
    # ML classifier for high performers
    # Section below just selects features, splits data, trains a Random Forest model, and evaluates it
    threshold = metrics['trader_quality_score'].quantile(0.7)
    metrics['high_performer'] = (metrics['trader_quality_score'] >= threshold).astype(int)

    feature_cols = [
        'win_rate',
        'profit_factor',
        'sortino_ratio',
        'expected_payoff',
        'total_trades',
        'max_drawdown'
    ]
    X = metrics[feature_cols]
    y = metrics['high_performer']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report for High vs Low Performers:\n", report)
    logging.info("ML model training and evaluation complete.")

    # Saving CSVs
    # Predictions
    probs = clf.predict_proba(X_test)[:, 1]  # this is the probability of high‐performer
    preds = clf.predict(X_test)

    results = X_test.copy()
    results['actual'] = y_test
    results['predicted'] = preds
    results['probability'] = probs
    results.to_csv('Data/trader_predictions.csv', index_label='trader_id')

    # Feature importances
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    fi.to_csv('Data/feature_importances.csv', index=False)

    # Saving the trained model model
    joblib.dump(clf, 'Data/trader_classifier.joblib')
    logging.info("Saved Random Forest model to Data/trader_classifier.joblib.")