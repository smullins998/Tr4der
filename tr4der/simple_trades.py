
from utils.data_loader import DataLoader
from tr4der import Tr4der
from pandas import DataFrame
import numpy as np
import pandas as pd

class SimpleTrades:
    
    def __init__(self):
        self.strategy_data = Tr4der.strategy_data


    def __str__(self):
        return f"Object to experiment with different trades"
        
    
    def long_only(self, df: DataFrame) -> None:
        pass