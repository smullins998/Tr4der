import pandas as pd
import yfinance as yf
import openai
import ast

from .gpt_helper import GptHelper
from .type_transformation import transform_data
from ..config import openai_api_key
 
class DataLoader:
    
    def __init__(self, data_prompt: str):
        
        # Transform the data from csv to a dataframe
        self._ticker_data = pd.read_csv("./tr4der/data/all_ticker_data.csv", header=0).transpose()
        self._ticker_data.columns = self._ticker_data.iloc[0]
        self._ticker_data = self._ticker_data.iloc[1:]
        self._ticker_data.reset_index(inplace=True)
        self._ticker_data.rename(columns={"index": "ticker"}, inplace=True)
        self._ticker_data = transform_data(self._ticker_data)
        
        # Initialize the GptHelper class
        self._gpt_helper = GptHelper(openai_api_key, data_prompt, self._ticker_data)

    @property
    def strategy_data(self):
        return self._strategy_data

    
    def load_data(self) -> None:
        self._gpt_helper.load_data()
        self._strategy_data = self._gpt_helper.strategy_data
        
        
