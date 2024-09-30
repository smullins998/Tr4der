import ast

import pandas as pd


# Method to transform the data into the correct types to handle
def transform_data(df) -> None:

    # Convert string columns
    string_columns = [
        "address1",
        "city",
        "state",
        "country",
        "phone",
        "website",
        "industry",
        "industryKey",
        "sector",
        "longBusinessSummary",
        "irWebsite",
        "exchange",
        "longName",
        "uuid",
        "messageBoardId",
        "recommendationKey",
        "lastSplitFactor",
    ]
    df[string_columns] = df[string_columns].astype(str)

    # Convert ZIP to string (to preserve leading zeros)
    df["zip"] = df["zip"].astype(str)

    # Convert integer columns (nullable to handle NaNs)
    integer_columns = [
        "fullTimeEmployees",
        "auditRisk",
        "boardRisk",
        "compensationRisk",
        "shareHolderRightsRisk",
        "overallRisk",
        "maxAge",
        "priceHint",
        "volume",
        "regularMarketVolume",
        "averageVolume",
        "bidSize",
        "askSize",
        "floatShares",
        "sharesOutstanding",
        "sharesShort",
        "sharesShortPriorMonth",
        "impliedSharesOutstanding",
        "numberOfAnalystOpinions",
    ]
    df[integer_columns] = (
        df[integer_columns].apply(pd.to_numeric, errors="ignore").astype("Int64")
    )  # Coerce invalid values to NaN and use nullable Int64

    # Convert float columns (handling NaNs by coercion)
    float_columns = [
        "previousClose",
        "open",
        "dayLow",
        "dayHigh",
        "regularMarketPreviousClose",
        "regularMarketOpen",
        "regularMarketDayLow",
        "regularMarketDayHigh",
        "dividendRate",
        "dividendYield",
        "payoutRatio",
        "beta",
        "trailingPE",
        "forwardPE",
        "bid",
        "ask",
        "fiftyTwoWeekLow",
        "fiftyTwoWeekHigh",
        "priceToSalesTrailing12Months",
        "fiftyDayAverage",
        "twoHundredDayAverage",
        "trailingAnnualDividendRate",
        "trailingAnnualDividendYield",
        "profitMargins",
        "priceToBook",
        "earningsQuarterlyGrowth",
        "netIncomeToCommon",
        "trailingEps",
        "forwardEps",
        "pegRatio",
        "enterpriseToRevenue",
        "enterpriseToEbitda",
        "52WeekChange",
        "SandP52WeekChange",
        "lastDividendValue",
        "targetHighPrice",
        "targetLowPrice",
        "targetMeanPrice",
        "targetMedianPrice",
        "recommendationMean",
        "totalCashPerShare",
        "quickRatio",
        "currentRatio",
        "debtToEquity",
        "revenuePerShare",
        "returnOnAssets",
        "returnOnEquity",
        "earningsGrowth",
        "revenueGrowth",
        "grossMargins",
        "ebitdaMargins",
        "operatingMargins",
        "trailingPegRatio",
        "marketCap",
        "enterpriseValue",
        "totalCash",
        "totalDebt",
        "totalRevenue",
        "operatingCashflow",
        "freeCashflow",
        "ebitda",
    ]
    df[float_columns] = df[float_columns].astype(
        float, errors="ignore"
    )  # Convert with coercion for errors

    # First filter out any rows with market cap below 100 million
    # Results can be majorly impacted by microcaps

    df = df[df["marketCap"] > 100000000]

    # Convert epoch timestamps to datetime
    epoch_columns = [
        "compensationAsOfEpochDate",
        "exDividendDate",
        "sharesShortPreviousMonthDate",
        "dateShortInterest",
        "lastFiscalYearEnd",
        "nextFiscalYearEnd",
        "mostRecentQuarter",
        "lastSplitDate",
        "firstTradeDateEpochUtc",
    ]
    df[epoch_columns] = df[epoch_columns].apply(
        lambda col: pd.to_datetime(col, unit="s", errors="ignore")
    )

    # Convert complex columns (like lists or dictionaries)
    df["companyOfficers"] = df["companyOfficers"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else x
    )

    return df
