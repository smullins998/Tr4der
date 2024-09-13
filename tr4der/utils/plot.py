import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .metrics import calculate_metrics


def plot_results(df: pd.DataFrame, stats: dict) -> None:
    """
    Create a plot to visualize trading strategy results with cumulative return,
    individual stock returns, and buy/sell signals.

    Args:
    df (pd.DataFrame): DataFrame containing strategy data
    stats (dict): Dictionary containing strategy statistics
    """
    # Ensure 'Date' column exists
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df.index)

    # Set up the plot
    plt.figure(figsize=(16, 10))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Cumulative Return as area
    plt.fill_between(df['Date'], df['Cumulative_Return'] - 1, 0, alpha=0.3, color='#1e90ff', label='Cumulative Return')
    plt.plot(df['Date'], df['Cumulative_Return'] - 1, color='#1e90ff', linewidth=2)

    # Calculate and plot cumulative returns for individual stocks
    for column in df.columns:
        if column.endswith('_return') and column != 'Total_Return':
            stock_name = column.split('_')[0]
            cumulative_return = (1 + df[column]).cumprod() - 1
            plt.plot(df['Date'], cumulative_return, label=f'{stock_name} Cumulative', linewidth=1.5)


    plt.title('Trading Strategy Results', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.legend(loc='upper left')

    # Add text box with key statistics
    stats_text = f"""
    Return: {stats['Return [%]']:.2f}%
    Sharpe Ratio: {stats['Sharpe Ratio']:.2f}
    Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%
    """
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, va="bottom", ha="left", bbox={"facecolor":"white", "alpha":0.8, "pad":5})

    plt.tight_layout()
    plt.show()