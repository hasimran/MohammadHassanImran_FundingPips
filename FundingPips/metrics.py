import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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

# Analyzes trading patterns of the top 10 traders
def analyze_top_traders(df, metrics, top_n=10):
    top_ids = metrics.sort_values('trader_quality_score', ascending=False).head(top_n).index.tolist()
    top_df = df[df['trading_account_login'].isin(top_ids)].copy()

    # Instrument preference mix
    inst_mix = (
        top_df.groupby('trading_account_login')['symbol']
              .value_counts(normalize=True)
              .unstack(fill_value=0)
    )
    inst_mix.to_csv('Data/top_traders_instrument_mix.csv')

    # Trade duration distribution
    plt.figure(figsize=(10, 6))
    sb.boxplot(
        data=top_df,
        x='trading_account_login',
        y='trade_duration_sec'
    )
    plt.xticks(rotation=45)
    plt.title('Trade Duration by Top Trader')
    plt.xlabel('Trader ID')
    plt.ylabel('Duration (sec)')
    plt.tight_layout()
    plt.savefig('Data/top_traders_duration.png')
    plt.close()

    # Stop-loss and take-profit hit rates
    def sl_tp_stats(grades):
        sl_rate = (grades['close_price'] == grades['price_sl']).mean()
        tp_rate = (grades['close_price'] == grades['price_tp']).mean()
        return pd.Series({'sl_hit_rate': sl_rate, 'tp_hit_rate': tp_rate})

    sltp = top_df.groupby('trading_account_login').apply(sl_tp_stats)
    sltp.to_csv('Data/top_traders_sltp_rates.csv')

    # Trade hour distribution heatmap
    top_df['hour'] = top_df['opened_at'].dt.hour
    heat = (
        top_df.groupby(['trading_account_login', 'hour'])
              .size()
              .unstack(fill_value=0)
    )
    heat = heat.div(heat.sum(axis=1), axis=0)
    plt.figure(figsize=(10, 6))
    sb.heatmap(
        heat,
        cmap='YlGnBu',
        cbar_kws={'label': 'Trade Frequency'}
    )
    plt.title('Trading Hour Distribution for Top Traders')
    plt.xlabel('Hour of Day')
    plt.ylabel('Trader ID')
    plt.tight_layout()
    plt.savefig('Data/top_traders_hourly_heatmap.png')
    plt.close()

    # Pip distribution per trade
    plt.figure(figsize=(10, 6))
    sb.histplot(
        data=top_df,
        x='pips',
        hue='trading_account_login',
        element='step',
        stat='density',
        common_norm=False
    )
    plt.title('Pip Distribution per Trade for Top Traders')
    plt.xlabel('Pips')
    plt.tight_layout()
    plt.savefig('Data/top_traders_pips_dist.png')
    plt.close()

    print("\nInstrument mix (%, by trader):")
    print(inst_mix)
    print("\nSL/TP hit rates:")
    print(sltp)

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

    # Section below visualizes some relevant metrics
    plt.figure(figsize=(10, 6))
    sb.scatterplot(
        data=metrics,
        x='win_rate',
        y='profit_factor',
        hue='trader_quality_score',
        palette='viridis',
        size='total_trades',
        sizes=(20, 200),
        alpha=0.8
    )
    plt.title("Win Rate vs Profit Factor (Colored by Trader Quality Score)")
    plt.xlabel("Win Rate")
    plt.ylabel("Profit Factor")
    plt.yscale("log") # To better visualize data as there are many outliers
    plt.legend(title="Quality Score", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("Data/win_rate_vs_profit_factor.png")
    plt.close()
    
    top10 = metrics.sort_values("trader_quality_score", ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    sb.barplot(
        x=top10.index.astype(str),
        y=top10["trader_quality_score"],
        palette='Blues_d'
    )
    plt.title("Top 10 Traders by Quality Score")
    plt.xlabel("Trader ID")
    plt.ylabel("Quality Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Data/top_10_traders_by_quality_score.png")
    plt.close()
    
    # Saving output metrics as CSV
    output_file='Data/trader_metrics.csv'
    metrics.to_csv(output_file)
    
    return metrics