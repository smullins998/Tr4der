from tr4der.utils.data_loader import DataLoader
from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import Optional, List
from tr4der.utils.metrics import calculate_metrics
from tr4der.utils.plot import plot_results


class TechnicalAnalysisStrategies:
    
    def __init__(self):
        pass

    def __str__(self):
        return f"Object to experiment with different technical trades"
        

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
     
        
    
    @staticmethod
    def momentum(df: DataFrame = None, window: int = 5) -> None:
        original_len_cols = len(df.columns)
        
        for stock in df.columns[:original_len_cols]:
            df[f'{stock}_return'] = df[stock].pct_change()
            df[f'{stock}_momentum'] = df[f'{stock}_return'].rolling(window=window).mean()
            df[f'{stock}_position'] = np.where(df[f'{stock}_momentum'].isna(), 0, 
                                               np.where(df[f'{stock}_momentum'] > 0, 1, -1))
            df[f'{stock}_strategy'] = df[f'{stock}_position'].shift(1) * df[f'{stock}_return']
        
        # Calculate total return across all stocks
        df['Total_Return'] = df[[col for col in df.columns if col.endswith('_strategy')]].mean(axis=1)
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1
        
        # Get output
        metrics = calculate_metrics(df, 'momentum')
        plot_results(df, metrics)
        

    
    @staticmethod
    def bollinger_bands(df: DataFrame = None, window: int = 20, num_std: int = 2) -> None:
        
        for stock in df.columns:
            df[f'{stock}_return'] = df[stock].pct_change()
            df[f'{stock}_ma'] = df[stock].rolling(window=window).mean()
            df[f'{stock}_std'] = df[stock].rolling(window=window).std()
            df[f'{stock}_upper_band'] = df[f'{stock}_ma'] + (num_std * df[f'{stock}_std'])
            df[f'{stock}_lower_band'] = df[f'{stock}_ma'] - (num_std * df[f'{stock}_std'])
            df[f'{stock}_position'] = np.where(df[stock].isna(), np.nan, 
                                               np.where(df[stock] > df[f'{stock}_upper_band'], -1, 
                                                         np.where(df[stock] < df[f'{stock}_lower_band'], 1, np.nan)))   
            df[f'{stock}_position'] = df[f'{stock}_position'].ffill()
            df[f'{stock}_strategy'] = df[f'{stock}_position'].shift(1) * df[f'{stock}_return']
            
        df['Total_Return'] = df[[col for col in df.columns if col.endswith('_strategy')]].mean(axis=1)
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1
        
        # Get output
        metrics = calculate_metrics(df, 'BollingerBands')
        plot_results(df, metrics)
    