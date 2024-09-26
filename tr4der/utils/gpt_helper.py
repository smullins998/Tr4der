from datetime import date
from pathlib import Path
from typing import Any, Dict

import openai
import pandas as pd
import yaml
import yfinance as yf
from pandas import DataFrame

from .config import open_ai_model


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
        prompts_path = Path(__file__).parent / "prompts.yaml"
        with open(prompts_path, "r") as f:
            self._prompts = yaml.safe_load(f)

    def _generate_openai_response(self, prompt_key: str, **kwargs) -> str:
        response = openai.chat.completions.create(
            model=open_ai_model,
            messages=[
                {"role": "system", "content": self._prompts[prompt_key]["system"]},
                {
                    "role": "user",
                    "content": self._prompts[prompt_key]["user"].format(**kwargs),
                },
            ],
        )
        return response.choices[0].message.content

    @_api_key_validation
    def _gpt_code_generate(self) -> None:
        self._gpt_code = self._generate_openai_response(
            "gpt_code_generate",
            tickers=self._filtered_data["ticker"].tolist(),
            data_prompt=self._data_prompt,
            today=date.today().strftime("%Y-%m-%d"),
        )
        print(self._gpt_code)

    def _gpt_code_execute(self) -> Dict[str, Any]:
        namespace: Dict[str, Any] = {
            "yf": yf,
            "DataFrame": DataFrame,
            "date": date,
            "self": self,
        }

        try:
            print("Loading data...")
            exec(self._gpt_code, namespace)
            self._strategy_data = namespace.get("_strategy_data")
            print("strategy data", self._strategy_data)
            # Check if 'Date' is in the index, if not set it as the index
            if isinstance(self._strategy_data, DataFrame):
                if "Date" not in self._strategy_data.index.names:
                    if "Date" in self._strategy_data.columns:
                        self._strategy_data.set_index("Date", inplace=True)
                    else:
                        print("Warning: 'Date' column not found in the DataFrame.")
            else:
                print("Warning: _strategy_data is not a DataFrame.")
        except Exception as e:
            raise RuntimeError(f"Error executing GPT response: {str(e)}")

        return self._strategy_data

    @_api_key_validation
    def _pandas_code_generate(self, data_prompt: str) -> str:
        self._pandas_code = self._generate_openai_response(
            "pandas_code_generate",
            columns=", ".join(self._ticker_data.columns),
            data_prompt=self._data_prompt,
        )
        print(self._pandas_code)
        return self._pandas_code

    def _pandas_code_execute(self) -> None:
        namespace: Dict[str, Any] = {"pd": pd, "self": self}

        try:
            exec(f"result = {self._pandas_code}", namespace)
            self._filtered_data = namespace["result"]
        except Exception as e:
            raise RuntimeError(f"Error executing pandas code: {str(e)}")

    @_api_key_validation
    def _gpt_identify_strategy(self) -> str:
        from ..strategies.machine_learning_strategies import MachineLearningStrategies
        from ..strategies.simple_strategies import SimpleStrategies
        from ..strategies.technical_strategies import TechnicalAnalysisStrategies

        strategy_classes = [
            SimpleStrategies,
            TechnicalAnalysisStrategies,
            MachineLearningStrategies,
        ]
        self._trading_methods = []

        for cls in strategy_classes:
            self._trading_methods.extend(
                [
                    method
                    for method in dir(cls)
                    if callable(getattr(cls, method)) and not method.startswith("__")
                ]
            )
        self._trading_methods.append("other")

        messages = [
            {
                "role": "system",
                "content": self._prompts["strategy_identifier"]["system"],
            },
            {
                "role": "user",
                "content": self._prompts["strategy_identifier"]["user"].format(
                    data_prompt=self._data_prompt, trading_methods=self._trading_methods
                ),
            },
        ]

        response = openai.chat.completions.create(
            model=open_ai_model, messages=messages
        )

        self._strategy_identifier = response.choices[0].message.content
        if self._strategy_identifier == "other":
            raise ValueError(
                "Strategy not found. Please reference the list of strategies and try again, or use custom strategy input."
            )

        return self._strategy_identifier

    @_api_key_validation
    def _gpt_call_strategy(self) -> str:
        import inspect

        from ..strategies.machine_learning_strategies import MachineLearningStrategies
        from ..strategies.simple_strategies import SimpleStrategies
        from ..strategies.technical_strategies import TechnicalAnalysisStrategies

        # Get attributes from multiple classes
        classes_to_inspect = [
            SimpleStrategies,
            TechnicalAnalysisStrategies,
            MachineLearningStrategies,
        ]
        all_methods = {}

        for cls in classes_to_inspect:
            methods = {
                method: getattr(cls, method)
                for method in dir(cls)
                if callable(getattr(cls, method)) and not method.startswith("__")
            }
            all_methods.update(methods)
        print(all_methods)
        strategy_method = all_methods.get(self._strategy_identifier)
        if not strategy_method:
            raise ValueError(
                f"Strategy '{self._strategy_identifier}' not found in any of the classes."
            )

        signature = inspect.signature(strategy_method)
        args = signature.parameters

        messages = [
            {"role": "system", "content": self._prompts["strategy_call"]["system"]},
            {
                "role": "user",
                "content": self._prompts["strategy_call"]["user"].format(
                    data_prompt=self._data_prompt,
                    strategy_definition=strategy_method,
                    args=args,
                ),
            },
        ]

        response = openai.chat.completions.create(
            model=open_ai_model, messages=messages
        )
        self._strategy_function_call = response.choices[0].message.content

        print(messages)
        print(self._strategy_function_call)

    def _gpt_call_strategy_execute(self) -> None:
        from ..strategies.machine_learning_strategies import MachineLearningStrategies
        from ..strategies.simple_strategies import SimpleStrategies
        from ..strategies.technical_strategies import TechnicalAnalysisStrategies

        # Call the strategy
        namespace: Dict[str, Any] = {
            "pd": pd,
            "self": self,
            "SimpleStrategies": SimpleStrategies,
            "TechnicalAnalysisStrategies": TechnicalAnalysisStrategies,
            "MachineLearningStrategies": MachineLearningStrategies,
            "yf": yf,
            "DataFrame": DataFrame,
            "date": date,
        }
        try:
            exec(f"result = {self._strategy_function_call}", namespace)
            result = namespace["result"]
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
