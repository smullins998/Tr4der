from tr4der.utils.data_loader import DataLoader
from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import Optional, List
from tr4der.utils.metrics import calculate_metrics
from tr4der.utils.plot import plot_results



class SimpleStrategies:
    
    def __init__(self):
        pass

    def __str__(self):
        return f"Object to experiment with different simple trades"
        
    @staticmethod
    def long(df: DataFrame = None) -> None:

        original_len_cols = len(df.columns) # Save the original length of the columns
        
        # Calculate returns for long only strategy
        for stock in df.columns:
            df[f'{stock}_return'] = df[stock].pct_change()
        
        # Use mean instead of sum for Total_Return
        df['Total_Return'] = df[[col for col in df.columns if 'return' in col]].mean(axis=1)
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1

        # Add signal column
        for stock in df.columns[:original_len_cols]:
            df[f'{stock}_signal'] = 0
            df.loc[df.index[0], f'{stock}_signal'] = 'Buy'
            df.loc[df.index[-1], f'{stock}_signal'] = 'Sell'
    
        # Get output
        metrics = calculate_metrics(df, 'long')
        plot_results(df, metrics)



    @staticmethod
    def short(df: DataFrame) -> None:
        
        original_len_cols = len(df.columns) # Save the original length of the columns
        
        # Calculate returns for short only strategy
        for stock in df.columns:
            df[f'{stock}_return'] = -1 * df[stock].pct_change()
        
        # Use mean instead of sum for Total_Return
        df['Total_Return'] = df[[col for col in df.columns if 'return' in col]].mean(axis=1)
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1

        # Add signal column
        for stock in df.columns[:original_len_cols]:
            df[f'{stock}_signal'] = 0
            df.loc[df.index[0], f'{stock}_signal'] = 'Sell'
            df.loc[df.index[-1], f'{stock}_signal'] = 'Buy'
    
        # Get output
        metrics = calculate_metrics(df, 'short')
        plot_results(df, metrics)
    
    
    
    
    @staticmethod
    def long_short(df: DataFrame = None, 
                   long_tickers: Optional[List[str]] = None, 
                   short_tickers: Optional[List[str]] = None,
                   ) -> None:
                
        original_len_cols = len(df.columns) # Save the original length of the columns

        # Calculate returns for short only strategy
        for stock in df.columns:
            if stock in long_tickers:
                df[f'{stock}_return'] = df[stock].pct_change()
            elif stock in short_tickers:
                df[f'{stock}_return'] = -1 * (df[stock].pct_change())
            else:
                df[f'{stock}_return'] = 0
                 
        # Use mean instead of sum for Total_Return
        df['Total_Return'] = df[[col for col in df.columns if 'return' in col]].mean(axis=1)
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1

        # Add signal column
        for stock in df.columns[:original_len_cols]:
            df[f'{stock}_signal'] = 0
            if stock in long_tickers:
                df.loc[df.index[0], f'{stock}_signal'] = 'Buy'
                df.loc[df.index[-1], f'{stock}_signal'] = 'Sell'
            elif stock in short_tickers:
                df.loc[df.index[0], f'{stock}_signal'] = 'Sell'
                df.loc[df.index[-1], f'{stock}_signal'] = 'Buy'

        # Get output
        metrics = calculate_metrics(df, 'pair_trade')
        plot_results(df, metrics)