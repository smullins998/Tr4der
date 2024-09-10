import yfinance as yf
from pandas import DataFrame
import pandas as pd
from datetime import date
import openai
from typing import Dict, Any
import os

openai.api_key = "sk-KYMTz3dbIpE9T46HklkZI4g1n89NeN1Xsp7zOtyK60T3BlbkFJmfEKBHcO_CRFrZJzhbT25xCIJFD1kehCARRTqyfXUA"

class LoadData:
    
    def __init__(self, data_prompt: str) -> None:
        
        #
        self.ticker_data = pd.read_csv("../all_ticker_data.csv", header=None).transpose()
        self.data_prompt = data_prompt
        
        
        # self.pandas_intermediate_data = self._pandas_intermediate_data(self.data_prompt)
        
        
        # self.gpt_response = self._get_gpt_response()
        # self.data = self._execute_gpt_response()

    def _get_gpt_response(self) -> str:
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that parses the user's prompt and returns the appropriate Python code to get data from Yahoo Finance. Return only Python code without markdown syntax or comments."},
                {"role": "user", "content": self.data_prompt}
            ]
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    
    def _execute_gpt_response(self) -> Dict[str, Any]:
        
        namespace: Dict[str, Any] = {"yf": yf, "DataFrame": DataFrame, "date": date}
        
        try:
            exec(self.pandas_intermediate_data, namespace)
        except Exception as e:
            raise RuntimeError(f"Error executing GPT response: {str(e)}")
        
        # Find the variable that holds the result (assuming it's the last assignment)
        result_vars = [var for var in namespace if not var.startswith("__") and var not in ["yf", "DataFrame", "date"]]
        if not result_vars:
            raise ValueError("No result found in GPT response")
        
        return namespace[result_vars[-1]]
    
    def _pandas_intermediate_data(self, data_prompt: str) -> DataFrame:
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes python pandas code to filter a dataframe to only include the columns that are most relevant to the user's prompt. The dataframe has attributes as rows and tickers as columns. That attributes are the same fields as using the method yf.info on a ticker. The dataframe is called self.ticker_data"},
                {"role": "user", "content": self.data_prompt}
            ]
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    

    def __str__(self) -> str:
        return "Object to load data from Yahoo Finance"

inst = LoadData("I want to analyze all of the tickers on the NASDAQ 100.")
print(inst.ticker_data)

