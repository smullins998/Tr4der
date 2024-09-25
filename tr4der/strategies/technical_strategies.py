from tr4der.utils.data_loader import DataLoader
from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import Optional, List
from tr4der.utils.metrics import calculate_metrics
from tr4der.utils.plot import plot_results
import ta

class TechnicalAnalysisStrategies:
    
    def __init__(self):
        pass

    def __str__(self):
        return "Object to experiment with different technical trades"
    
    
    @staticmethod
    def momentum_strategy(df: DataFrame = None, window: int = 5) -> None:
        original_len_cols = len(df.columns)
        
        for stock in df.columns[:original_len_cols]:
            df[f'{stock}_return'] = df[stock].pct_change()
            df[f'{stock}_momentum'] = df[f'{stock}_return'].rolling(window=window).mean()
            df[f'{stock}_position'] = np.where(df[f'{stock}_momentum'].isna(), 0, 
                                               np.where(df[f'{stock}_momentum'] > 0, 1, -1))
            df[f'{stock}_strategy'] = df[f'{stock}_position'].shift(1) * df[f'{stock}_return']

            # Add buy and sell signals only when position changes
            df[f'{stock}_signal'] = np.where(df[f'{stock}_position'] != df[f'{stock}_position'].shift(), 
                                             np.where(df[f'{stock}_position'] == 1, 'Buy', 
                                                      np.where(df[f'{stock}_position'] == -1, 'Sell', None)),
                                             None)

        # Calculate total return across all stocks
        df['Total_Return'] = df[[col for col in df.columns if col.endswith('_strategy')]].mean(axis=1)
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1
        
        # Get output
        metrics = calculate_metrics(df, 'momentum')
        plot_results(df, metrics)
        

    @staticmethod
    def macd_trend_following(df: DataFrame = None, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> None:
        for stock in df.columns:
            df[f'{stock}_return'] = df[stock].pct_change()
            df[f'{stock}_EMA12'] = df[stock].ewm(span=fast_window, adjust=False).mean()
            df[f'{stock}_EMA26'] = df[stock].ewm(span=slow_window, adjust=False).mean()
            df[f'{stock}_MACD'] = df[f'{stock}_EMA12'] - df[f'{stock}_EMA26']
            df[f'{stock}_MACD_signal'] = df[f'{stock}_MACD'].rolling(window=signal_window).mean()
            df[f'{stock}_position'] = np.where(df[f'{stock}_MACD'].isna(), np.nan, 
                                               np.where(df[f'{stock}_MACD'] > df[f'{stock}_MACD_signal'], 1, 
                                                         np.where(df[f'{stock}_MACD'] < df[f'{stock}_MACD_signal'], -1, np.nan)))
            df[f'{stock}_position'] = df[f'{stock}_position'].ffill()
            df[f'{stock}_strategy'] = df[f'{stock}_position'].shift(1) * df[f'{stock}_return']

            # Add buy and sell signals only when position changes
            df[f'{stock}_signal'] = np.where(df[f'{stock}_position'] != df[f'{stock}_position'].shift(), 
                                                np.where(df[f'{stock}_position'] == 1, 'Buy', 
                                                        np.where(df[f'{stock}_position'] == -1, 'Sell', None)),
                                                None)


        df['Total_Return'] = df[[col for col in df.columns if col.endswith('_strategy')]].mean(axis=1)
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1
        
        # Get output
        metrics = calculate_metrics(df, 'macd')
        plot_results(df, metrics)


    
    @staticmethod
    def mean_reversion_moving_average(df: DataFrame = None) -> None:
        
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
            df[f'{stock}_strategy'] = df[f'{stock}_position'].shift() * df[f'{stock}_return']

            # Add buy and sell signals only when position changes
            df[f'{stock}_signal'] = np.where(df[f'{stock}_position'] != df[f'{stock}_position'].shift(), 
                                             np.where(df[f'{stock}_position'] == 1, 'Buy', 
                                                      np.where(df[f'{stock}_position'] == -1, 'Sell', None)),
                                             None)

        # Calculate total return across all stocks
        df['Total_Return'] = df[[col for col in df.columns if col.endswith('_strategy')]].mean(axis=1)
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1

        # Get output
        metrics = calculate_metrics(df, 'mean_reversion')
        plot_results(df, metrics)
        
     
    @staticmethod
    def mean_reversion_bollinger_bands(df: DataFrame = None, window: int = 20, num_std: int = 2) -> None:
        
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

            # Add buy and sell signals only when position changes
            df[f'{stock}_signal'] = np.where(df[f'{stock}_position'] != df[f'{stock}_position'].shift(), 
                                             np.where(df[f'{stock}_position'] == 1, 'Buy', 
                                                      np.where(df[f'{stock}_position'] == -1, 'Sell', None)),
                                             None)

        df['Total_Return'] = df[[col for col in df.columns if col.endswith('_strategy')]].mean(axis=1)
        df['Cumulative_Return'] = (1 + df['Total_Return']).cumprod()
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1
        
        # Get output
        metrics = calculate_metrics(df, 'bollinger_bands')
        plot_results(df, metrics)
    
    
    @staticmethod
    def pairs_trading(df: DataFrame = None, entry_threshold: float = 2.0, exit_threshold: float = 0.5, 
                      mean_window: int = 50, std_window: int = 20) -> None:
        '''Improved pairs trading strategy using log returns and dynamic position sizing.'''
        if len(df.columns) != 2:
            raise ValueError("Number of stocks must be 2 for pair trading.")

        first_stock, second_stock = df.columns

        # Calculate log returns
        for stock in [first_stock, second_stock]:
            df[f'{stock}_log_return'] = np.log(df[stock] / df[stock].shift(1))

        # Calculate spread using log prices
        df['spread'] = np.log(df[first_stock]) - np.log(df[second_stock])
        df['spread_ma'] = df['spread'].rolling(window=mean_window).mean()
        df['spread_std'] = df['spread'].rolling(window=std_window).std()
        df['spread_zscore'] = (df['spread'] - df['spread_ma']) / df['spread_std']

        # Generate signals with dynamic position sizing
        def position_size(zscore, entry, exit):
            if abs(zscore) < exit:
                return 0
            elif abs(zscore) > entry:
                return np.sign(zscore) * min(abs(zscore) / entry, 1)
            else:
                return np.nan  # Will be forward filled

        df['position'] = df['spread_zscore'].apply(lambda z: position_size(z, entry_threshold, exit_threshold))
        df['position'] = df['position'].fillna(method='ffill')

        df[f'{first_stock}_position'] = -df['position']
        df[f'{second_stock}_position'] = df['position']

        # Add buy and sell signals only when position changes
        for stock in [first_stock, second_stock]:
            df[f'{stock}_signal'] = np.where(df[f'{stock}_position'] != df[f'{stock}_position'].shift(), 
                                            np.where(df[f'{stock}_position'] > 0, 'Buy', 
                                                    np.where(df[f'{stock}_position'] < 0, 'Sell', None)),
                                            None)
        df = df.iloc[mean_window:,:]

        # Calculate returns
        df['Total_Return'] = (df[f'{first_stock}_log_return'] * df[f'{first_stock}_position'] + 
                              df[f'{second_stock}_log_return'] * df[f'{second_stock}_position'])
        df['Cumulative_Return'] = np.exp(df['Total_Return'].cumsum()) 
        df['Drawdown'] = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax()) - 1

        # Get output
        metrics = calculate_metrics(df, 'pair_trade')
        plot_results(df, metrics)