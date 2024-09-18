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
    def linear_regression_strategy(df: DataFrame) -> None:
        pass