import pandas as pd
import numpy as np
from typing import Dict, Any
import math


def calculate_metrics(df: pd.DataFrame, strategy: str) -> Dict[str, any]:
    metrics = {}        
    
    # Time period
    metrics['Start'] = df.index.min()
    metrics['End'] = df.index.max()
    metrics['Duration'] = metrics['End'] - metrics['Start']

    # Calculate metrics
    metrics['Exposure Time [%]'] = round((df['Total_Return'].count() / len(df)) * 100, 2)
    metrics['Equity Initial [$]'] = 10000
    metrics['Equity Final [$]'] = round(df['Cumulative_Return'].iloc[-1] * 10000, 2)  # Assuming $10,000 initial investment
    metrics['Equity Peak [$]'] = round(df['Cumulative_Return'].max() * 10000, 2)
    metrics['Return [%]'] = round((df['Cumulative_Return'].iloc[-1] - 1) * 100, 2)
    metrics['Return (Ann.) [%]'] = round(((1 + metrics['Return [%]'] / 100) ** (252 / len(df)) - 1) * 100, 2)
    metrics['Volatility (Ann.) [%]'] = round(df['Total_Return'].std() * np.sqrt(252) * 100, 2)
    metrics['Sharpe Ratio'] = round(metrics['Return (Ann.) [%]'] / metrics['Volatility (Ann.) [%]'], 2)
    
    # Sortino and Calmar Ratio calculations
    negative_returns = df['Total_Return'][df['Total_Return'] < 0]
    downside_deviation = np.sqrt(np.mean(negative_returns**2)) * np.sqrt(252)
    metrics['Sortino Ratio'] = round(metrics['Return (Ann.) [%]'] / (downside_deviation * 100), 2)
    
    # Max Drawdown calculation
    max_drawdown = df['Drawdown'].min()
    metrics['Max. Drawdown [%]'] = round(max_drawdown * 100, 2)
    metrics['Calmar Ratio'] = round(metrics['Return (Ann.) [%]'] / abs(metrics['Max. Drawdown [%]']), 2)
    
    # Average Drawdown
    drawdowns = df['Drawdown'][df['Drawdown'] < 0]
    metrics['Avg. Drawdown [%]'] = round(drawdowns.mean() * 100, 2)
    
    # Drawdown Duration calculations
    drawdown_periods = (df['Drawdown'] < 0).astype(int).diff().fillna(0)
    drawdown_starts = df.index[drawdown_periods == 1]
    drawdown_ends = df.index[drawdown_periods == -1]
    if len(drawdown_ends) < len(drawdown_starts):
        drawdown_ends = drawdown_ends.append(df.index[-1:])
    drawdown_durations = drawdown_ends - drawdown_starts
    metrics['Max. Drawdown Duration'] = drawdown_durations.max()
    metrics['Avg. Drawdown Duration'] = drawdown_durations.mean()
    
    #Get number of trade-metrics                    
    action_columns = [col for col in df.columns if col.endswith('_signal')]
    buy_count = (df[action_columns] == 'Buy').sum().sum()
    sell_count = (df[action_columns] == 'Sell').sum().sum()
  
    # Trade metrics
    metrics['# Trades'] = buy_count + sell_count
    metrics['Best Day [%]'] = round(df['Total_Return'].max() * 100, 2)
    metrics['Worst Day [%]'] = round(df['Total_Return'].min() * 100, 2)
    metrics['Avg. Trade [%]'] = round(df['Total_Return'].mean() * 100, 2)
    metrics['Max. Trade Duration'] = (df.index[-1] - df.index[0]).days
    metrics['strategy'] = strategy

    for metric in metrics:
        print(f"{metric}: {metrics[metric]}")
    
    return metrics
