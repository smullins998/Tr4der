import yfinance as yf
from pandas import DataFrame
import pandas as pd
from datetime import date
import openai
from typing import Dict, Any
import os
from tqdm import tqdm
from tr4der.utils.data_loader import DataLoader



#I think eventually I want to add support for options data and historicals. 
#This should be with the yf api. Or maybe we use Optionistics. There has to be a way.

class Tr4der:
    
    def __str__(self) -> str:
        return "Object to load data from Yahoo Finance"
    
    def __init__(self, data_prompt: str):
        #Load the data
        self._data_loader = DataLoader(data_prompt)
        self._data_loader.execute_code()
        
        
    @property
    def strategy_data(self) -> DataFrame:
        return self._data_loader.strategy_data
        


