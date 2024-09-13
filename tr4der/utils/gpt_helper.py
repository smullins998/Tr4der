from typing import Dict, Any
from datetime import date
from pathlib import Path

import openai
import pandas as pd
import yaml
import yfinance as yf
from pandas import DataFrame
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

    def _generate_openai_response(self, prompt_key: str, **kwargs) -> str:
        response = openai.chat.completions.create(
            model=open_ai_model,
            messages=[
                {"role": "system", "content": self._prompts[prompt_key]['system']},
                {"role": "user", "content": self._prompts[prompt_key]['user'].format(**kwargs)}
            ]
        )
        return response.choices[0].message.content
        
    @_api_key_validation
    def _gpt_code_generate(self) -> None:
        self._gpt_code = self._generate_openai_response('gpt_code_generate', 
            tickers=self._filtered_data['ticker'].tolist(),
            data_prompt=self._data_prompt
        )
        print(self._gpt_code)
    
    def _gpt_code_execute(self) -> Dict[str, Any]:
        namespace: Dict[str, Any] = {"yf": yf, "DataFrame": DataFrame, "date": date, 'self': self}
        
        try:
            print('Loading data...')
            exec(self._gpt_code, namespace)
            self._strategy_data = namespace.get('_strategy_data')
            # Check if 'Date' is in the index, if not set it as the index
            if isinstance(self._strategy_data, DataFrame):
                if 'Date' not in self._strategy_data.index.names:
                    if 'Date' in self._strategy_data.columns:
                        self._strategy_data.set_index('Date', inplace=True)
                    else:
                        print("Warning: 'Date' column not found in the DataFrame.")
            else:
                print("Warning: _strategy_data is not a DataFrame.")
        except Exception as e:
            raise RuntimeError(f"Error executing GPT response: {str(e)}")
        
        return self._strategy_data
    
    @_api_key_validation
    def _pandas_code_generate(self, data_prompt: str) -> str:
        self._pandas_code = self._generate_openai_response('pandas_code_generate', 
            columns=', '.join(self._ticker_data.columns),
            data_prompt=self._data_prompt
        )
        print(self._pandas_code)
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
        from ..simple_trades import SimpleTrades

        self._trading_methods = [method for method in dir(SimpleTrades) if callable(getattr(SimpleTrades, method)) and not (method.startswith('__') and method.endswith('__'))] + ['other']
        
        messages = [
            {"role": "system", "content": self._prompts['strategy_identifier']['system']},
            {"role": "user", "content": self._prompts['strategy_identifier']['user'].format(
                data_prompt=self._data_prompt,
                trading_methods=self._trading_methods
            )}
        ]

        response = openai.chat.completions.create(
            model=open_ai_model,
            messages=messages
        )

        self._strategy_identifier = response.choices[0].message.content
        if self._strategy_identifier == 'other':
            raise ValueError("Strategy not found. Please reference the list of strategies and try again, or use custom strategy input.")

    @_api_key_validation
    def _gpt_call_strategy(self) -> str:
        from ..simple_trades import SimpleTrades
        import inspect  # Import inspect module
    
        
        strategy_method = getattr(SimpleTrades, self._strategy_identifier)
        signature = inspect.signature(strategy_method)
        args = signature.parameters  # Get the parameters of the method

        messages = [
            {"role": "system", "content": self._prompts['strategy_call']['system']},
            {"role": "user", "content": self._prompts['strategy_call']['user'].format(
                data_prompt=self._data_prompt,
                strategy_definition=strategy_method,
                args=args  # Include args in the message
            )}
        ]

        response = openai.chat.completions.create(
            model=open_ai_model,
            messages=messages
        )
        self._strategy_function_call = response.choices[0].message.content
        
        print(messages)
        print(self._strategy_function_call)
    
    
    def _gpt_call_strategy_execute(self) -> None:
        from ..simple_trades import SimpleTrades
        
        #Call the strategy
        namespace: Dict[str, Any] = {
            "pd": pd, 
            "self": self, 
            "SimpleTrades": SimpleTrades,
            "yf": yf,
            "DataFrame": DataFrame,
            "date": date
        }
        try:
            exec(f"result = {self._strategy_function_call}", namespace)
            result = namespace['result']
        except Exception as e:
            raise RuntimeError(f"Error executing strategy code: {str(e)}")


    def execute_code(self) -> None:
        self._pandas_code_generate(self._data_prompt)
        self._pandas_code_execute()
        self._gpt_code_generate()
        self._gpt_code_execute()
        self._gpt_identify_strategy()
        self._gpt_call_strategy()
        self._gpt_call_strategy_execute()
        
        
    @property
    def strategy_data(self) -> DataFrame:
        return self._strategy_data