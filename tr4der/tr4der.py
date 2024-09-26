import yfinance as yf
from pandas import DataFrame
import pandas as pd
from datetime import date
import openai
from typing import Dict, Any
import os
from .utils import DataLoader
from .strategies import SimpleStrategies, TechnicalAnalysisStrategies, MachineLearningStrategies

class Tr4der:

    def __str__(self) -> str:
        return "Object to load data from Yahoo Finance"

    def __init__(self):

        # Allow flexibility to explore different strategies with different parameters.
        # Ultimately the user could pass in a strategy object and then we can call the execute method.
        self.SimpleStrategies = SimpleStrategies()
        self.TechnicalAnalysisStrategies = TechnicalAnalysisStrategies()
        self.MachineLearningStrategies = MachineLearningStrategies()

    def set_api_key(self, api_key: str):
        openai.api_key = api_key

    def query(self, query: str) -> str:
        # Load the data
        self._data_loader = DataLoader(query)
        self._data_loader.execute_code()

    @property
    def strategy_data(self) -> DataFrame:
        df = self._data_loader.strategy_data
        df = df[
            [ticker for ticker in df.columns if ticker != "Date" and "_" not in ticker]
        ]
        return df
