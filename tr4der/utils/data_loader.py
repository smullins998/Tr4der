import ast

import openai
import pandas as pd
import yfinance as yf

from .gpt_helper import GptHelper
from .type_transformation import transform_data

try:
    from .config import openai_api_key
except ImportError:
    raise ImportError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY in the config file."
    )


class DataLoader:

    def __init__(self, data_prompt: str):

        # Transform the data from csv to a dataframe
        self._ticker_data = pd.read_csv(
            "./tr4der/data/all_ticker_data.csv", header=0
        ).transpose()
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

    def execute_code(self) -> None:
        self._gpt_helper.execute_code()
        self._strategy_data = self._gpt_helper.strategy_data
