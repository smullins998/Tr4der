from tr4der.utils.data_loader import DataLoader
from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import Optional, List
from tr4der.utils.metrics import calculate_metrics
from tr4der.utils.plot import plot_results


class SimpleTrades:
    
    def __init__(self):
        pass

    def __str__(self):
        return f"Object to experiment with different trades"
        
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

        # Add action column
        for stock in df.columns[:original_len_cols]:
            df[f'{stock}_action'] = 0
            df.loc[df.index[0], f'{stock}_action'] = 'buy'
            df.loc[df.index[-1], f'{stock}_action'] = 'sell'
    
        # Get output
        metrics = calculate_metrics(df, 'long')
        plot_results(df, metrics)



    def short(self, df: DataFrame) -> None:
        pass
    
    
    
    
    @staticmethod
    def pair_trade(df: DataFrame = None, 
                   long_ticker: Optional[List[str]] = None, 
                   short_ticker: Optional[List[str]] = None,
                   ) -> None:
        print('Your strategy is long {long_ticker} and short {short_ticker}!')
    
