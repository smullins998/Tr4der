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

        # Add action column
        for stock in df.columns[:original_len_cols]:
            df[f'{stock}_action'] = 0
            df.loc[df.index[0], f'{stock}_action'] = 'sell'
            df.loc[df.index[-1], f'{stock}_action'] = 'buy'
    
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

        # Add action column
        for stock in df.columns[:original_len_cols]:
            df[f'{stock}_action'] = 0
            if stock in long_tickers:
                df.loc[df.index[0], f'{stock}_action'] = 'buy'
                df.loc[df.index[-1], f'{stock}_action'] = 'sell'
            elif stock in short_tickers:
                df.loc[df.index[0], f'{stock}_action'] = 'sell'
                df.loc[df.index[-1], f'{stock}_action'] = 'buy'

        # Get output
        metrics = calculate_metrics(df, 'pair_trade')
        plot_results(df, metrics)
    
    
    
    @staticmethod
    def mean_reversion(df: DataFrame = None) -> None:
        
        '''We use a moving average approach here for stationary mean reversion.
        '''
        
        original_len_cols = len(df.columns)

        # Calculate returns and z-scores
        for stock in df.columns[:original_len_cols]:
            df[f'{stock}_return'] = df[stock].pct_change()
            df[f'{stock}_ma'] = df[stock].rolling(window=20).mean()
            df[f'{stock}_ratio'] = df[stock] / df[f'{stock}_ma']
            
            percentiles = [.5, .95]
            p = df[f'{stock}_ratio'].dropna().quantile(percentiles)
            long = p.iloc[0]
            short = p.iloc[1]
            
            df[f'{stock}_position'] = np.nan
            df.loc[df[f'{stock}_ratio'] > short, f'{stock}_position'] = -1
            df.loc[df[f'{stock}_ratio'] < long, f'{stock}_position'] = 1
            df[f'{stock}_position'] = df[f'{stock}_position'].fillna(method='ffill') # We will always be in or out of a position.
            df[f'{stock}_strategy_return'] = df[f'{stock}_position'].shift() * df[f'{stock}_return']

        # Calculate total return across all stocks
        df['Total_Return'] = df[[col for col in df.columns if col.endswith('_strategy_return')]].mean(axis=1)
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1

        # Get output
        metrics = calculate_metrics(df, 'mean_reversion')
        plot_results(df, metrics)
        
     
    @staticmethod
    def pair_trade(df: DataFrame = None) -> None:
        '''We use the price spread between the two trades; it may make more sense to use percent change.
        '''
        original_len_cols = len(df.columns)
        col_list = list(df.columns)
        
        if original_len_cols  > 2:
            raise ValueError("Number of stocks must be 2 for pair trading.")

        # Calculate spread and z-score for each pair
        first_stock = col_list[0]
        second_stock = col_list[1]
        
        df[f'{first_stock}_return'] = df[first_stock].pct_change()
        df[f'{second_stock}_return'] = df[second_stock].pct_change()

        df[f'{first_stock}_{second_stock}_spread'] = df[first_stock] - df[second_stock]
        df[f'{first_stock}_{second_stock}_spread_ma'] = df[f'{first_stock}_{second_stock}_spread'].rolling(window=20).mean()
        df[f'{first_stock}_{second_stock}_spread_zscore'] = (df[f'{first_stock}_{second_stock}_spread'] - df[f'{first_stock}_{second_stock}_spread_ma']) / df[f'{first_stock}_{second_stock}_spread'].rolling(window=20).std()

    
        # Generate signals
        df[f'{first_stock}_signal'] = np.where(df[f'{first_stock}_{second_stock}_spread_zscore'] > 1, -1, 
                                      np.where(df[f'{first_stock}_{second_stock}_spread_zscore'] < -1, 1, np.nan))
        
        df[f'{second_stock}_signal'] = np.where(df[f'{first_stock}_{second_stock}_spread_zscore'] > 1, 1, 
                                       np.where(df[f'{first_stock}_{second_stock}_spread_zscore'] < -1, -1, np.nan))

        df[f'{first_stock}_signal'] = df[f'{first_stock}_signal'].ffill()
        df[f'{second_stock}_signal'] = df[f'{second_stock}_signal'].ffill()

        df['Total_Return'] = ((df[f'{first_stock}_return'] * df[f'{first_stock}_signal']) + (df[f'{second_stock}_return'] * df[f'{second_stock}_signal'])) / 2
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1

        # Get output
        metrics = calculate_metrics(df, 'pair_trade')
        plot_results(df, metrics)
     
        
    
    def momentum(df: DataFrame = None) -> None:
        pass

        