   
from typing import Dict, Any

import openai
from pandas import DataFrame
from datetime import date
import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path
from ..config import open_ai_model



class GptHelper:
    
    def __init__(self, api_key: str, data_prompt: str, ticker_data: DataFrame) -> None:
        openai.api_key = api_key
        self._data_prompt = data_prompt
        self._ticker_data = ticker_data
        self._load_prompts()
        
    @staticmethod
    def _api_key_validation(func) -> None:
        def wrapper(*args, **kwargs):
            if not openai.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            return func(*args, **kwargs)
        return wrapper
    
        
    def _load_prompts(self) -> None:
        prompts_path = Path(__file__).parent / 'prompts.yaml'
        with open(prompts_path, 'r') as f:
            self._prompts = yaml.safe_load(f)
        
        
    @_api_key_validation
    def _gpt_code_generate(self) -> str:
        
        response = openai.chat.completions.create(
            model=open_ai_model,
            messages=[
                {"role": "system", "content": self._prompts['gpt_code_generate']['system']},
                {"role": "user", "content": self._prompts['gpt_code_generate']['user'].format(
                    tickers=self._filtered_data['ticker'].tolist(),
                    data_prompt=self._data_prompt
                )}
            ]
        )
        self._gpt_code = response.choices[0].message.content
    
    def _gpt_code_execute(self) -> Dict[str, Any]:
        namespace: Dict[str, Any] = {"yf": yf, "DataFrame": DataFrame, "date": date, 'self': self}
        
        try:
            print('Loading data...')
            exec(self._gpt_code, namespace)
            # Add this line to capture the result
            
            self._strategy_data = namespace.get('_strategy_data')
            print(self._strategy_data)
        except Exception as e:
            raise RuntimeError(f"Error executing GPT response: {str(e)}")
        
        return namespace.get('_strategy_data')
    
    @_api_key_validation
    def _pandas_code_generate(self, data_prompt: str) -> str:

        response = openai.chat.completions.create(
            model=open_ai_model,
            messages=[
                {"role": "system", "content": self._prompts['pandas_code_generate']['system']},
                {"role": "user", "content": self._prompts['pandas_code_generate']['user'].format(
                    columns=', '.join(self._ticker_data.columns),
                    data_prompt=self._data_prompt
                )}
            ]
        )
        self._pandas_code = response.choices[0].message.content
        return self._pandas_code

    def _pandas_code_execute(self) -> None:
        namespace: Dict[str, Any] = {"pd": pd, "self": self}
        
        try:
            exec(f"result = {self._pandas_code}", namespace)
            self._filtered_data = namespace['result']
        except Exception as e:
            raise RuntimeError(f"Error executing pandas code: {str(e)}")


    @_api_key_validation
    def _gpt_identify_strategy(self) -> str:
        
        response = openai.chat.completions.create(
            model=open_ai_model,
            messages=[
                {"role": "system", "content": self._prompts['pandas_code_generate']['system']},
                {"role": "user", "content": self._prompts['pandas_code_generate']['user'].format(
                    columns=', '.join(self._ticker_data.columns),
                    data_prompt=self._data_prompt
                )}
            ]
        )
        self._strategy_identifier = response.choices[0].message.content
        return self._strategy_identifier


    def load_data(self) -> None:

        self._pandas_code_generate(self._data_prompt)
        self._pandas_code_execute()
        self._gpt_code_generate()
        self._gpt_code_execute()


    @property
    def strategy_data(self) -> DataFrame:
        return self._strategy_data