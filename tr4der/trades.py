
from tr4der.utils.data_loader import DataLoader
from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import Optional, List

class SimpleTrades:
    
    def __init__(self):
        pass

    def __str__(self):
        return f"Object to experiment with different trades"
        
    
    def long_only(self, df: DataFrame) -> None:
        pass
    
    def short_only(self, df: DataFrame) -> None:
        pass
    
    @staticmethod
    def pair_trade(self, 
                   df: DataFrame = None, 
                   long_ticker: Optional[List[str]] = None, 
                   short_ticker: Optional[List[str]] = None,
                   ) -> None:
        print('Your strategy is long {long_ticker} and short {short_ticker}!')
    
    