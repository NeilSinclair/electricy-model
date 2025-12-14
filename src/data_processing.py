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
    ANNUAL_USAGE: int = 1_000_000
    BASELINE_USAGE_MWH: int = 200_00
    FLAT_RATE_C_PER_KWH: float = 30.0

    # Battery parameters
    BATTERY_INEFFICIENCY_FACTOR: float = 0.95

    # Additional costs (c/kWh)
    NETWORK_USAGE: float = 8.71
    TAX_RATE: float= 0.19
    ELECTRICITY_TAX: float = 2.05
    ADDITIONAL_COST: float = 1.88
    KONZESSION: float = 1.66
    CHP_SURCHARGE: float = 0.45

    # Battery investment
    BATTERY_COST_PER_KWH: float = 300.0
    OPEX_PERCENT_OF_CAPEX: float = 0.05

    # Battery size
    BATTERY_SIZE_KWH: int = 120
    BATTERY_POWER: int = 50
    SOC_FACTOR: float = 0.8

    # Other
    INFLATION_RATE: float = 0.02

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

    de_holidays = holidays.Germany(years=[2025,2024])

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
    )#.drop(columns=["day_type", "time", "date", "holiday_name", "is_holiday"])

    df = df.set_index("datetime")

    df_hourly = pd.DataFrame(df.resample("h")["value"].mean()).reset_index()

    return df_hourly

def process_heat_energy_profile(data_path: str = 'data/heat_data.csv') -> pd.DataFrame:
    """Function which processes raw heat energy profile data and merges with hourly energy profile"""
    heat_data_hourly = pd.read_csv(data_path, sep=';')
    heat_data_hourly["Timestamp"] = pd.to_datetime(heat_data_hourly["Timestamp"], unit="s", utc=True)
    heat_data_hourly["Timestamp"] = heat_data_hourly["Timestamp"].dt.tz_convert(None)

    heat_data_hourly = heat_data_hourly.rename(columns={"Timestamp": "datetime", "Heat Pump Demand kWh (electrical)": "value"})
    heat_data_hourly["value"] = heat_data_hourly["value"].str.replace(',', '.')
    heat_data_hourly = heat_data_hourly.astype({'value': 'float'})
    heat_data_hourly = heat_data_hourly[['datetime', 'value']]

    return heat_data_hourly

def process_day_ahead_data(df_hourly: pd.DataFrame, data_path: str = 'data/day_ahead_1yr.csv') -> pd.DataFrame:
    """Function which processes raw day-ahead price data and merges with hourly energy profile"""
    day_ahead = pd.read_csv(data_path, sep=";")

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
        config = ElectricityConfig.from_yaml("config/config.yaml")

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

def get_usage_data():
    """Function which loads and processes all necessary data and configuration."""
    config = ElectricityConfig.from_yaml("config/config.yaml")
    df_hourly = process_heat_energy_profile()
    day_ahead_hourly = process_day_ahead_data(df_hourly)
    df_usage_and_price = calculate_usage_and_price(df_hourly, day_ahead_hourly, config=config)

    return config, df_usage_and_price
