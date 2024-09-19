from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from tr4der.utils.metrics import calculate_metrics
from tr4der.utils.plot import plot_results


class MachineLearningStrategies:
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __str__(self):
        return f"Object to experiment with different machine learning trades"
    
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


        print(df.head())
        # Get output
        metrics = calculate_metrics(test, 'linear_regression')
        plot_results(test, metrics)
        
        
        
        
        