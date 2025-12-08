import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import holidays
import numpy as np

from dataclasses import dataclass
import yaml

@dataclass
class ElectricityConfig:
    # Usage and rates
    ANNUAL_USAGE: int
    BASELINE_USAGE_MWH: int
    FLAT_RATE_C_PER_KWH: float

    # Battery parameters
    BATTERY_INEFFICIENCY_FACTOR: float
    ELECTRICITY_SHIFT_FACTOR: float

    # Additional costs (€/MWh)
    NETWORK_USAGE: float
    TAX_RATE: float
    ELECTRICITY_TAX: float
    ADDITIONAL_COST: float
    KONZESSION: float
    CHP_SURCHARGE: float

    # Battery investment
    BATTERY_COST_PER_KWH: float
    OPEX_PERCENT_OF_CAPEX: float

    # Battery size
    BATTERY_SIZE_KWH: int
    BATTERY_POWER: int
    SOC_FACTOR: float

    @classmethod
    def from_yaml(cls, path: str) -> "ElectricityConfig":
        """Load configuration values from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

def process_hourly_energy_profile(start_date: str = "2025-01-01 00:00", end_date: str = "2025-12-31 23:45") -> pd.DataFrame:
    """Function which takes raw energy profile data and converts it to hourly usage"""
    data = pd.read_excel("data/energy_profile.xlsx", header=2)
    data.drop(columns=["Unnamed: 0"], inplace=True)

    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    data_types = ["SA", "FT", "WT"]
    new_months = []
    for month in months:
        for data_type in data_types:
            new_months.append(month+"_"+data_type)

    data.columns = ["Time"] + new_months
    data.drop(index=0, inplace=True)
    data.dropna(axis=0,subset=["Time"], inplace=True)

    data = data.melt(id_vars="Time")

    data.columns = ["Time Period", "Date", "value"]
    data["month_name"] = data.Date.apply(lambda x: x.split("_")[0])
    data["day_type"] = data.Date.apply(lambda x: x.split("_")[1])
    data["time"] = data["Time Period"].apply(lambda x: x.split("-")[0])
    data["time"] = pd.to_datetime(data["time"], format="%H:%M").dt.time

    data.drop(columns=["Date", "Time Period"], inplace=True)

    day_map = {
    "Monday": "WT",
    "Tuesday": "WT",
    "Wednesday": "WT",
    "Thursday": "WT",
    "Friday": "WT",
    "Saturday": "SA",
    "Sunday": "FT"
    }

    de_holidays = holidays.Germany(years=2025)

    # 1. Create all 15-minute timestamps for 2025
    dt_index = pd.date_range(
        start=start_date,
        end=end_date,   # last 15-minute slot of the year
        freq="15min"
    )

    df = pd.DataFrame({"datetime": dt_index})

    # 2. Add date, time, weekday name
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time
    df["weekday"] = df["datetime"].dt.day_name() 
    df["month_name"] = df["datetime"].dt.month_name()
    # df["weekday"] = df["datetime"].dt.day_name(locale="de_DE")  # if you have locale set up
    # Name of holiday (or None)
    df["holiday_name"] = df["datetime"].dt.date.map(de_holidays.get)

    # Boolean flag for “is public holiday?”
    df["is_holiday"] = df["holiday_name"].notna()
    df["day_type"] = df["weekday"].map(day_map)
    df.loc[df["is_holiday"], "day_type"] = "FT"

    pd.set_option("display.max_rows", 10)

    df = df.merge(
    data,
    on = ["month_name", "day_type", "time"],
    how="left"
    ).drop(columns=["day_type", "time", "date", "holiday_name", "is_holiday"])

    df = df.set_index("datetime")

    df_hourly = pd.DataFrame(df.resample("h")["value"].mean()).reset_index()

    return df_hourly

def process_day_ahead_data(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Function which processes raw day-ahead price data and merges with hourly energy profile"""
    day_ahead = pd.read_csv("data/day_ahead_1yr.csv", sep=";")

    day_ahead = day_ahead[["Start date", "End date", "Germany/Luxembourg [€/MWh] Original resolutions", "∅ DE/LU neighbours [€/MWh] Original resolutions"]]

    day_ahead.columns = ["start_date", "end_date", "de_price", "neighbour_price"]
    day_ahead["start_date"] = pd.to_datetime(day_ahead["start_date"], format="%b %d, %Y %I:%M %p")
    day_ahead["end_date"] = pd.to_datetime(day_ahead["end_date"], format="%b %d, %Y %I:%M %p")

    df_combined = df_hourly.merge(
        day_ahead[["start_date", "de_price", "neighbour_price"]],
        left_on="datetime",
        right_on="start_date",
        how="left"
    ).drop(columns=["start_date"])

    day_ahead_hourly = pd.DataFrame(df_combined.resample("h", on="datetime")["de_price"].mean()).reset_index()

    return day_ahead_hourly

def calculate_usage_and_price(df_hourly: pd.DataFrame, day_ahead_hourly: pd.DataFrame, config: ElectricityConfig | None = None ) -> pd.DataFrame:
    """Function which calculates total costs based on usage and price data
    
    Args:
        df_hourly (pd.DataFrame): DataFrame containing hourly usage data
        day_ahead_hourly (pd.DataFrame): DataFrame containing hourly day-ahead price data
        config (ElectricityConfig | None): Configuration parameters. If None, loads from 'config.yaml'.

    Returns:
        pd.DataFrame: DataFrame with calculated costs
    """

    if config is None:
        config = ElectricityConfig.from_yaml("config.yaml")

    SCALE_FACTOR = config.ANNUAL_USAGE / config.BASELINE_USAGE_MWH

    COMBINED_ADDITIONAL_COSTS = config.NETWORK_USAGE + config.ELECTRICITY_TAX + config.ADDITIONAL_COST + config.KONZESSION + config.CHP_SURCHARGE 

    df_usage_and_price = df_hourly.merge(
        day_ahead_hourly,
        on="datetime",
        how="left"
    ).rename(columns={"de_price": "c_per_kwh_variable", "value": "raw_kwh_usage"})

    df_usage_and_price['month'] = df_usage_and_price['datetime'].dt.month
    df_usage_and_price['month_name'] = df_usage_and_price['datetime'].dt.strftime('%B')
    df_usage_and_price['hour_of_day'] = df_usage_and_price['datetime'].dt.hour

    df_usage_and_price["c_per_kwh_variable"] = df_usage_and_price["c_per_kwh_variable"] / 10 # Convert €/MWh to c€/kWh

    # Fill missing prices with average price
    df_usage_and_price.loc[df_usage_and_price["c_per_kwh_variable"].isna(), "c_per_kwh_variable"] = df_usage_and_price["c_per_kwh_variable"].mean()
    df_usage_and_price["scaled_kwh_usage"] = df_usage_and_price["raw_kwh_usage"] * SCALE_FACTOR
    df_usage_and_price["c_fixed_costs_kwh"] = COMBINED_ADDITIONAL_COSTS

    df_usage_and_price["c_variable_and_fixed_per_kwh"] = df_usage_and_price["c_per_kwh_variable"] + df_usage_and_price["c_fixed_costs_kwh"]

    df_usage_and_price["c_total_variable_cost"] = (
        (df_usage_and_price["scaled_kwh_usage"] * (df_usage_and_price["c_variable_and_fixed_per_kwh"])) * (1 + config.TAX_RATE)
    )

    df_usage_and_price["c_per_kwh_flat_rate_cost"] = config.FLAT_RATE_C_PER_KWH
    df_usage_and_price["c_total_flat_cost"] = (df_usage_and_price["c_per_kwh_flat_rate_cost"] * df_usage_and_price["scaled_kwh_usage"]) * (1+config.TAX_RATE)


    return df_usage_and_price

def shift_electricity_usage(df_usage_and_price: pd.DataFrame, config: ElectricityConfig | None = None) -> pd.DataFrame:
    """Function which shifts the electricity usage to occur at times with lower prices 

    The algorithm first inverts the price weights so that lower prices have higher weights. It then applies a temperature-based 
    weighting to create a more peaked distribution at the lower prices. The lower the temperature, the more pronounced the shift towards lower prices.
    
    Args:
        df_usage_and_price (pd.DataFrame): DataFrame containing usage and price data
        config (ElectricityConfig | None): Configuration parameters. If None, loads from 'config.yaml'.
    
    Returns:
        pd.DataFrame: DataFrame with shifted usage and calculated costs
    """

    if config is None:
        config = ElectricityConfig.from_yaml("config.yaml")
    
    T = config.ELECTRICITY_SHIFT_FACTOR

    hourly_monthly_avg = df_usage_and_price.groupby(['hour_of_day', 'month_name'])['c_variable_and_fixed_per_kwh'].mean().reset_index()

    g = hourly_monthly_avg.groupby("month_name")

    # 1) Per-month weights (sum to 1 within each month)
    hourly_monthly_avg["weighted_mean"] = g["c_variable_and_fixed_per_kwh"].transform(
        lambda x: x / x.sum()
    )

    # 2) Per-month inverse weights
    hourly_monthly_avg["inverse_weighted_mean"] = g["weighted_mean"].transform(
        lambda x: x.max() - x
    )

    # 3) (Optional) Normalize inverse weights per month as well
    hourly_monthly_avg["inverse_weighted_mean"] = g["inverse_weighted_mean"].transform(
        lambda x: x / x.sum()
    )

    hourly_monthly_avg["temperature_weighted_usage"] = g["inverse_weighted_mean"].transform(
            lambda x: np.exp(x / T) / np.exp(x / T).sum()
    )

    return hourly_monthly_avg

def shift_electricity_usage_daily(df_usage_and_price: pd.DataFrame, config: ElectricityConfig | None = None) -> pd.DataFrame:
    """Function which shifts the electricity usage to occur at times with lower prices 

    The algorithm first inverts the price weights so that lower prices have higher weights. It then applies a temperature-based 
    weighting to create a more peaked distribution at the lower prices. The lower the temperature, the more pronounced the shift towards lower prices.
    
    Args:
        df_usage_and_price (pd.DataFrame): DataFrame containing usage and price data
        config (ElectricityConfig | None): Configuration parameters. If None, loads from 'config.yaml'.
    
    Returns:
        pd.DataFrame: DataFrame with shifted usage and calculated costs
    """

    if config is None:
        config = ElectricityConfig.from_yaml("config.yaml")
    
    T = config.ELECTRICITY_SHIFT_FACTOR

    df = df_usage_and_price.copy()
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["year"] = df["datetime"].dt.year

    hourly_daily_average = df.groupby(['hour_of_day', 'day_of_year', "year"])['c_variable_and_fixed_per_kwh'].mean().reset_index()

    g = hourly_daily_average.groupby(["day_of_year", "year"])

    # 1) Per-month weights (sum to 1 within each month)
    hourly_daily_average["weighted_mean"] = g["c_variable_and_fixed_per_kwh"].transform(
        lambda x: x / x.sum()
    )

    # 2) Per-month inverse weights
    hourly_daily_average["inverse_weighted_mean"] = g["weighted_mean"].transform(
        lambda x: x.max() - x
    )

    # 3) (Optional) Normalize inverse weights per month as well
    hourly_daily_average["inverse_weighted_mean"] = g["inverse_weighted_mean"].transform(
        lambda x: x / x.sum()
    )

    hourly_daily_average["temperature_weighted_usage"] = g["inverse_weighted_mean"].transform(
            lambda x: np.exp(x / T) / np.exp(x / T).sum()
    )

    return hourly_daily_average

def merge_temperature_shift_with_usage(
        df_usage_and_price: pd.DataFrame, 
        hourly_daily_average: pd.DataFrame,
        on_cols=["day_of_year", "year"],
        config: ElectricityConfig | None = None
    ) -> pd.DataFrame:
    """This function merges the temperature shifted usage profile with the original usage data.
    Args:
        df_usage_and_price (pd.DataFrame) - the dateframe
        hourly_daily_average (pd.DataFrame) - hourly values averaged over each day dataframe with temperature weighted values
        on_cols (list) - columns to join on, default is ["day_of_year", "month_name", "year"]
        config (ElectricityConfig | None) - configuration
    
    Returns:
        pd.DataFrame
    """

    if config is None:
        config = ElectricityConfig.from_yaml("config.yaml")

    df = df_usage_and_price.copy()
    hourly_daily_average = hourly_daily_average.copy()
    df = df.merge(
        hourly_daily_average[["hour_of_day", "day_of_year", "year", "temperature_weighted_usage"]],
        left_on=[df["datetime"].dt.hour, df["datetime"].dt.dayofyear, df["datetime"].dt.year],
        right_on=["hour_of_day", "day_of_year", "year"],
        how="left"
    )

    df["day_of_year"] = df["datetime"].dt.dayofyear
    
    usage_by_day = df.groupby(["year", "day_of_year"], as_index=False)["scaled_kwh_usage"].sum()
    usage_by_day = usage_by_day.rename(columns={"scaled_kwh_usage": "daily_total_usage_kwh"})
    
    df = df.merge(
        usage_by_day,
        on=on_cols,
        how="left"
    )

    df = df.drop(columns=["hour_of_day_x", "hour_of_day_y"])

    return df

def calculate_shifted_electricity_cost(
        df_usage_and_price: pd.DataFrame, 
        hourly_monthly_avg: pd.DataFrame, on_cols=["day_of_year", "month_name"], 
        config: ElectricityConfig | None = None
    ) -> pd.DataFrame:
    """This function calculates the shifted electricity costs based on the shifted usage profile.

    Args:
        df_usage_and_price (pd.DataFrame) - the dateframe
        hourly_monthly_avg (pd.DataFrame) - hourly values averaged over each month dataframe with temperature weighted values
        on_cols (list) - columns to join on, default is ["day_of_year", "month_name"]
        config (ElectricityConfig | None) - configuration object, if None loads from 'config.yaml'
    
    Returns: 
        pd.DataFrame

    """

    if config is None:
        config = ElectricityConfig.from_yaml("config.yaml")

    df = df_usage_and_price.copy()
    hourly_monthly_avg = hourly_monthly_avg.copy()

    df = df.merge(
        hourly_monthly_avg[["hour_of_day", "month_name", "temperature_weighted_usage"]],
        left_on=[df["datetime"].dt.hour, df['datetime'].dt.month_name()],
        right_on=["hour_of_day", "month_name"],
        how="left"
    )

    df["day_of_year"] = df["datetime"].dt.dayofyear
    
    usage_by_day = df.groupby(["month_name", "day_of_year"], as_index=False)["scaled_kwh_usage"].sum()
    usage_by_day = usage_by_day.rename(columns={"scaled_kwh_usage": "daily_total_usage_kwh"})
    
    df = df.merge(
        usage_by_day,
        on=on_cols,
        how="left"
    )

    df["shifted_usage_kwh"] = (df["temperature_weighted_usage"] * df["daily_total_usage_kwh"]) 

    df["c_total_variable_shifted_cost"] = df["shifted_usage_kwh"] * 1/config.BATTERY_INEFFICIENCY_FACTOR * (df["c_per_kwh_variable"] + df["c_fixed_costs_kwh"]) * (1 + config.TAX_RATE)

    df.drop(columns=["hour_of_day", "day_of_year", "hour_of_day_x", "hour_of_day_y", "daily_total_usage_kwh"], inplace=True)
    return df