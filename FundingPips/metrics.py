import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Calculate maximum drawdown
def max_drawdown(profits):
    cumulative = profits.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    max_dd = drawdown.max() / peak.max() if peak.max() != 0 else np.nan
    return max_dd

# Calculate average profit excluding outliers (5th to 95th percentile)
def outlier_adj_profit(profits):
    lower, upper = profits.quantile([0.05, 0.95])
    adj_profits = profits[(profits >= lower) & (profits <= upper)]
    return adj_profits.mean() if len(adj_profits) > 0 else np.nan

def compute_trader_metrics(df):
    trader_groups = df.groupby('trading_account_login')
    filtered_df = trader_groups.filter(lambda x: len(x) > 1) # Filter out accounts with only one trade

    metrics = filtered_df.groupby('trading_account_login').agg(
        total_trades=('identifier', 'count'),
        win_rate=('is_profitable', 'mean'),
        avg_profit=('profit', 'mean'),
        std_profit=('profit', 'std'),
        avg_duration_sec=('trade_duration_sec', 'mean'),
        gross_profit=('profit', lambda x: x[x > 0].sum()),
        gross_loss=('profit', lambda x: abs(x[x < 0].sum())),
        max_drawdown=('profit', max_drawdown),
        outlier_adj_avg_profit=('profit', outlier_adj_profit),
        downside_std=('profit', lambda x: x[x < 0].std() if len(x[x < 0]) > 0 else np.nan)
    )
    
    metrics['profit_factor'] = metrics['gross_profit'] / metrics['gross_loss'].replace(0, np.nan)
    metrics['sharpe_proxy'] = metrics['avg_profit'] / metrics['std_profit'].replace(0, np.nan)
    metrics['sortino_ratio'] = metrics['avg_profit'] / metrics['downside_std'].replace(0, np.nan)
    metrics['expected_payoff'] = metrics['avg_profit'] * metrics['win_rate']
    metrics = metrics.drop(columns=['gross_profit', 'gross_loss', 'downside_std', 'std_profit', 'avg_profit'])
    
    score_components = metrics[['win_rate', 'profit_factor', 'sortino_ratio', 'expected_payoff', 'total_trades', 'outlier_adj_avg_profit']].copy()
    score_components = score_components.fillna(0)

    # Normalize all score components to [0, 1]
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(score_components)

    # Invert drawdown as lower is better
    drawdown_scaled = 1 - MinMaxScaler().fit_transform(metrics[['max_drawdown']].fillna(1)) 
    
    # Equally weighted composite score
    composite = np.mean(np.hstack([normalized, drawdown_scaled]), axis=1) 

    metrics['trader_quality_score'] = composite

    output_file='Data/trader_metrics.csv'
    metrics.to_csv(output_file)
    
    return metrics