import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .metrics import calculate_metrics


def plot_results(df: pd.DataFrame, stats: dict) -> None:
    """
    Create subplots to visualize trading strategy results with cumulative return
    and individual stock returns with buy/sell signals.

    Args:
    df (pd.DataFrame): DataFrame containing strategy data
    stats (dict): Dictionary containing strategy statistics
    """

    # Ensure 'Date' column exists
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df.index)

    # Count the number of individual stocks
    stock_columns = [col for col in df.columns if col.endswith('_return') and col != 'Total_Return']
    num_stocks = len(stock_columns)

    # Set up the subplots
    fig, axs = plt.subplots(num_stocks + 1, 1, figsize=(16, 6 * (num_stocks + 1)), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Cumulative Return
    axs[0].fill_between(df['Date'], df['Cumulative_Return'] - 1, 0, alpha=0.3, color='#1e90ff', label='Cumulative Return')
    axs[0].plot(df['Date'], df['Cumulative_Return'] - 1, color='#1e90ff', linewidth=2)
    axs[0].set_title('Your Strategy Cumulative Return', fontsize=14)
    axs[0].set_ylabel('Return', fontsize=12)
    axs[0].legend(loc='upper left')

    # Plot individual stocks
    for i, column in enumerate(stock_columns, 1):
        stock_name = column.split('_')[0]
        cumulative_return = (1 + df[column]).cumprod() - 1
        
        axs[i].plot(df['Date'], cumulative_return, label=f'{stock_name} Cumulative', linewidth=1.5)
        
        # Plot buy and sell signals
        signal_column = f'{stock_name}_signal'
        if signal_column in df.columns:
            buy_mask = df[signal_column] == "Buy"
            sell_mask = df[signal_column] == "Sell"
            
            axs[i].scatter(df.loc[buy_mask, 'Date'], cumulative_return.loc[buy_mask], 
                        marker='^', color='g', s=100, label='Buy')
            axs[i].scatter(df.loc[sell_mask, 'Date'], cumulative_return.loc[sell_mask], 
                        marker='v', color='r', s=100, label='Sell')

        axs[i].set_title(f'{stock_name} Cumulative Return', fontsize=14)
        axs[i].set_ylabel('Return', fontsize=12)
        axs[i].legend(loc='upper left')

    # Set common x-label
    fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12)

    # Add text box with key statistics
    stats_text = f"""
    Strategy: {stats['strategy']}
    Return: {stats['Return [%]']:.2f}%
    Sharpe Ratio: {stats['Sharpe Ratio']:.2f}
    Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%
    """
    fig.text(0.02, 0.02, stats_text, fontsize=10, va="bottom", ha="left", bbox={"facecolor":"white", "alpha":0.8, "pad":5})

    plt.tight_layout()
    plt.show()