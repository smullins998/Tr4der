from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import ta
import re

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from tr4der.utils.metrics import calculate_metrics
from tr4der.utils.plot import plot_results


class MachineLearningStrategies:
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __str__(self):
        return f"Object to experiment with different machine learning trades"
    
    @classmethod
    def _calculate_technical_indicators(cls, ticker: str, df: DataFrame, technical_indicators: List = ['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14']) -> DataFrame:
        ''' Helper function to calculate the technical indicators and the previous day return
        '''
        # Calculate the return for the ticker
        df[f'{ticker}_return'] = df[ticker].pct_change()
        
        # Calculate the technical indicators
        for indicator in technical_indicators:
            window_match = int(re.search(r'\d+', indicator).group())
            indicator_match = " ".join(re.findall("[a-zA-Z]+", indicator)).lower()
            if window_match:
                indicator_type = re.search(r'(sma|ema|rsi)', indicator.lower()).group(1)
                if indicator_match == 'sma':
                    df[indicator] = ta.trend.SMAIndicator(df[ticker], window=window_match).sma_indicator().shift(1)
                elif indicator_match == 'ema':
                    df[indicator] = ta.trend.EMAIndicator(df[ticker], window=window_match).ema_indicator().shift(1)
                elif indicator_match == 'rsi':
                    df[indicator] = ta.momentum.RSIIndicator(df[ticker], window=window_match).rsi().shift(1)
                elif indicator_match == 'macd':
                    df[indicator] = ta.trend.MACD(df[ticker], window_fast=12, window_slow=26, window_sign=9).macd().shift(1)
            df[indicator] = df[indicator].pct_change()
        
        # Calculate the previous day return
        df['Previous_Day_Return'] = df[ticker].pct_change().shift(1)
        df = df.dropna()

        return df
    
    
    @staticmethod
    def linear_regression(df: DataFrame, lags: int = 5) -> None:
        'We will only support one ticker for now'
        
        assert len(df.columns) == 1, "We only support one ticker for now"
        
        ticker = df.columns[0]
        LAG_NAMES = []
        
        #Create n-lags for each ticker
        df[f'{ticker}_return'] = df[ticker].pct_change()
        for i in range(1,lags+1):
            df[f'{ticker}_Lag_{i}'] = df[ticker].shift(i)
            LAG_NAMES.append(f'{ticker}_Lag_{i}')
        
        df = df.dropna()
        train, test = train_test_split(df, test_size=0.3, shuffle=False, random_state=0)
        
        # Use combined_df for further processing if needed
        model = LinearRegression()
        model.fit(train[LAG_NAMES], train[f'{ticker}_return'])
        
        # Calculate predictions and metrics for test data
        test['Predicted_Return'] = model.predict(test[LAG_NAMES])
        test['position'] = np.where(test['Predicted_Return'] > 0, 1, -1)
        test['Total_Return'] = test['position'] * test[f'{ticker}_return']
        test['Cumulative_Return'] = (1 + test['Total_Return']).cumprod()
        test['Drawdown'] = (test['Cumulative_Return'] / test['Cumulative_Return'].cummax()) - 1

        test[f'{ticker}_signal'] = np.where(test['Predicted_Return'] > 0, 'Buy', 'Sell')

        # Get output
        metrics = calculate_metrics(test, 'linear_regression')
        plot_results(test, metrics)
        
    @staticmethod
    def svm_regression(df: DataFrame, technical_indicators: List = ['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14']) -> None:
        'We will only support one ticker for now'
        
        assert len(df.columns) == 1, "We only support one ticker for now"
        
        ticker = df.columns[0]
        df[f'{ticker}_return'] = df[ticker].pct_change()
                
        df = MachineLearningStrategies._calculate_technical_indicators(ticker, df, technical_indicators)
        
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(df, test_size=0.5, random_state=0, shuffle=False)

        # Separate features and target variable
        train_features = train_data[technical_indicators + ['Previous_Day_Return']]
        train_target = train_data[f'{ticker}_return']

        # Create and train the SVR model
        svr = SVR(kernel='rbf')
        svr.fit(train_features, train_target)

        # Make predictions on the test set
        test_features = test_data[technical_indicators + ['Previous_Day_Return']]
        test_target = test_data[f'{ticker}_return']
        predictions = svr.predict(test_features)
        test_data['predictions'] = predictions

        #Let's add signals for buying/selling
        test_data[f'{ticker}_signal'] = np.where(test_data['predictions'] > 0, 'Buy', 'Sell')
        test_data[f'{ticker}_position'] = np.where(test_data[f'{ticker}_signal'] == 'Buy', 1, -1)
        test_data[f'Total_Return'] = test_data[f'{ticker}_position'] * test_data[f'{ticker}_return']
        test_data[f'Cumulative_Return'] = (1 + test_data[f'Total_Return']).cumprod()
        test_data[f'Drawdown'] = (test_data[f'Cumulative_Return'] / test_data[f'Cumulative_Return'].cummax()) - 1

        metrics = calculate_metrics(test_data, 'svm_regression')
        plot_results(test_data, metrics)
        
        
    @staticmethod
    def decision_tree_regression(df: DataFrame, technical_indicators: List = ['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14']) -> None:
        'We will only support one ticker for now'
        
        assert len(df.columns) == 1, "We only support one ticker for now"
        
        ticker = df.columns[0]
                
        df = MachineLearningStrategies._calculate_technical_indicators(ticker, df, technical_indicators)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(df, test_size=0.5, random_state=0, shuffle=False)

        # Separate features and target variable
        train_features = train_data[technical_indicators + ['Previous_Day_Return']]
        train_target = train_data[f'{ticker}_return']

        # Create and train the SVR model
        dtr = DecisionTreeRegressor()
        dtr.fit(train_features, train_target)

        # Make predictions on the test set
        test_features = test_data[technical_indicators + ['Previous_Day_Return']]
        test_target = test_data[f'{ticker}_return']
        predictions = dtr.predict(test_features)
        test_data['predictions'] = predictions


        #We actually find a very good negative correlation between this ML model. 
        #Let's add signals for buying/selling
        test_data[f'{ticker}_signal'] = np.where(test_data['predictions'] > 0, 'Buy', 'Sell')

        test_data[f'{ticker}_position'] = np.where(test_data[f'{ticker}_signal'] == 'Buy', 1, -1)
        test_data[f'Total_Return'] = test_data[f'{ticker}_position'] * test_data[f'{ticker}_return']
        test_data[f'Cumulative_Return'] = (1 + test_data[f'Total_Return']).cumprod()
        test_data[f'Drawdown'] = (test_data[f'Cumulative_Return'] / test_data[f'Cumulative_Return'].cummax()) - 1
        
        metrics = calculate_metrics(test_data, 'decision_tree_regression')
        plot_results(test_data, metrics)
        
    
    @staticmethod
    def nearest_neighbors_regression(df: DataFrame, technical_indicators: List = ['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14']) -> None:
        'We will only support one ticker for now'
        
        assert len(df.columns) == 1, "We only support one ticker for now"
        
        ticker = df.columns[0]
        
        df = MachineLearningStrategies._calculate_technical_indicators(ticker, df, technical_indicators)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(df, test_size=0.5, random_state=0, shuffle=False)

        # Separate features and target variable
        train_features = train_data[technical_indicators + ['Previous_Day_Return']]
        train_target = train_data[f'{ticker}_return']

        knn = KNeighborsRegressor()
        knn.fit(train_features, train_target)
        
        # Make predictions on the test set
        test_features = test_data[technical_indicators + ['Previous_Day_Return']]
        test_target = test_data[f'{ticker}_return']
        predictions = knn.predict(test_features)
        test_data['predictions'] = predictions
        
        #We actually find a very good negative correlation between this ML model. 
        #Let's add signals for buying/selling
        test_data[f'{ticker}_signal'] = np.where(test_data['predictions'] > 0, 'Buy', 'Sell')

        test_data[f'{ticker}_position'] = np.where(test_data[f'{ticker}_signal'] == 'Buy', 1, -1)
        test_data[f'Total_Return'] = test_data[f'{ticker}_position'] * test_data[f'{ticker}_return']
        test_data[f'Cumulative_Return'] = (1 + test_data[f'Total_Return']).cumprod()
        test_data[f'Drawdown'] = (test_data[f'Cumulative_Return'] / test_data[f'Cumulative_Return'].cummax()) - 1

        metrics = calculate_metrics(test_data, 'nearest_neighbors_regression')
        plot_results(test_data, metrics)
        
    
    @staticmethod
    def neural_network_regression(df: DataFrame, 
                                  technical_indicators: List = ['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14'],
                                  max_iterations: int = 1000,
                                  hidden_layers: Tuple[int, int] = (100,100)) -> None:
        'We will only support one ticker for now'
        
        assert len(df.columns) == 1, "We only support one ticker for now"
        
        ticker = df.columns[0]
        
        # Calculate technical indicators
        df = MachineLearningStrategies._calculate_technical_indicators(ticker, df, technical_indicators)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(df, test_size=0.5, random_state=0, shuffle=False)

        # Separate features and target variable
        train_features = train_data[technical_indicators + ['Previous_Day_Return']]
        train_target = train_data[f'{ticker}_return']

        # Create and train the SVR model
        mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=max_iterations, shuffle=False)
        mlp.fit(train_features, train_target)

        # Make predictions on the test set
        test_features = test_data[technical_indicators + ['Previous_Day_Return']]
        test_target = test_data[f'{ticker}_return']
        predictions = mlp.predict(test_features)
        test_data['predictions'] = predictions


        #We actually find a very good negative correlation between this ML model. 
        #Let's add signals for buying/selling
        test_data[f'{ticker}_signal'] = np.where(test_data['predictions'] > 0, 'Buy', 'Sell')
        test_data[f'{ticker}_position'] = np.where(test_data[f'{ticker}_signal'] == 'Buy', 1, -1)
        test_data[f'Total_Return'] = test_data[f'{ticker}_position'] * test_data[f'{ticker}_return']
        test_data[f'Cumulative_Return'] = (1 + test_data[f'Total_Return']).cumprod()
        test_data[f'Drawdown'] = (test_data[f'Cumulative_Return'] / test_data[f'Cumulative_Return'].cummax()) - 1
        
        metrics = calculate_metrics(test_data, 'neural_network_regression')
        plot_results(test_data, metrics)
        
    @staticmethod
    def LSTM(df: DataFrame, technical_indicators: List = ['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14'], 
             sequence_length: int = 60, epochs: int = 25, batch_size: int = 32) -> None:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np

        assert len(df.columns) == 1, "We only support one ticker for now"
        
        ticker = df.columns[0]
        
        # Calculate technical indicators
        df = MachineLearningStrategies._calculate_technical_indicators(ticker, df, technical_indicators)
        
        # Prepare the data
        features = df[technical_indicators + ['Previous_Day_Return']].values
        target = df[ticker].values
        
        # Normalize the data
        scaler_features = MinMaxScaler()
        scaler_target = MinMaxScaler()
        features_scaled = scaler_features.fit_transform(features)
        target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - sequence_length):
            X.append(features_scaled[i:(i + sequence_length)])
            y.append(target_scaled[i + sequence_length])
        X, y = np.array(X), np.array(y)
        
        # Split the data
        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build the LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
        
        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler_target.inverse_transform(predictions)
        y_test = scaler_target.inverse_transform(y_test)
        
        # Prepare results DataFrame
        test_data = df.iloc[-len(y_test):].copy()
        test_data['predictions'] = predictions
        
        # Add trading signals and calculate returns
        test_data[f'{ticker}_signal'] = np.where(test_data['predictions'] > test_data[ticker].shift(1), 'Buy', 'Sell')
        test_data[f'{ticker}_position'] = np.where(test_data[f'{ticker}_signal'] == 'Buy', 1, -1)
        test_data[f'{ticker}_return'] = test_data[ticker].pct_change()
        test_data[f'Total_Return'] = test_data[f'{ticker}_position'] * test_data[f'{ticker}_return']
        test_data[f'Cumulative_Return'] = (1 + test_data[f'Total_Return']).cumprod()
        test_data[f'Drawdown'] = (test_data[f'Cumulative_Return'] / test_data[f'Cumulative_Return'].cummax()) - 1
        
        # Calculate metrics and plot results
        metrics = calculate_metrics(test_data, 'LSTM')
        plot_results(test_data, metrics)
        