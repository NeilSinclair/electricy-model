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
    SCALE_FACTOR: float = 1.0
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
    TES_COST_PER_KWH: float = 200.0
    OPEX_PERCENT_OF_CAPEX: float = 0.05

    # Heat Pump parameters
    HEAT_PUMP_COP: float = 3.5
    HEAT_PUMP_COST_PER_KW: float = 1800.0
    HEAT_PUMP_SIZE_KW: float = 100.0
    HEAT_PUMP_OUTPUT_KW: float = 50.0 

    # Battery size
    TES_SIZE_KWH: float = 120.0
    TES_POWER: float = 50.0

    # Other
    INFLATION_RATE: float = 0.02
    INVESTMENT_DURATION_YEARS: int = 10

    # Gas heating
    GAS_HEATING_C_PER_KWH: float = 12.0
    GAS_CONVERSION_RATIO: float = 3.2

    @classmethod
    def from_yaml(cls, path: str) -> "ElectricityConfig":
        """Load configuration values from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

def process_heat_energy_profile(data_path: str = 'data/heat_data.csv') -> pd.DataFrame:
    """Function which processes raw heat energy profile data and merges with hourly energy profile"""
    heat_data_hourly = pd.read_csv(data_path, sep=';')
    heat_data_hourly["Timestamp"] = pd.to_datetime(heat_data_hourly["Timestamp"], unit="s", utc=True)
    heat_data_hourly["Timestamp"] = heat_data_hourly["Timestamp"].dt.tz_convert(None) # type: ignore

    heat_data_hourly = heat_data_hourly.rename(columns={"Timestamp": "datetime", "Heat demand kWh (heat)": "value"})
    heat_data_hourly["value"] = heat_data_hourly["value"].str.replace(',', '.')
    heat_data_hourly = heat_data_hourly.astype({'value': 'float'})
    heat_data_hourly = heat_data_hourly[['datetime', 'value']]

    return heat_data_hourly

def process_day_ahead_data(df_hourly: pd.DataFrame, data_path: str = 'data/day_ahead_1yr.csv') -> pd.DataFrame:
    """Function which processes raw day-ahead price data and merges with hourly energy profile
    
    The source of this data is ... 
    """
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

    SCALE_FACTOR = config.SCALE_FACTOR

    COMBINED_ADDITIONAL_COSTS = config.NETWORK_USAGE + config.ELECTRICITY_TAX + config.ADDITIONAL_COST + config.KONZESSION + config.CHP_SURCHARGE 

    df_usage_and_price = df_hourly.merge(
        day_ahead_hourly,
        on="datetime",
        how="left"
    ).rename(columns={"de_price": "c_per_kwh_variable", "value": "raw_kwh_usage"})

    df_usage_and_price['month'] = df_usage_and_price['datetime'].dt.month # type: ignore
    df_usage_and_price['month_name'] = df_usage_and_price['datetime'].dt.strftime('%B')  # type: ignore
    df_usage_and_price['hour_of_day'] = df_usage_and_price['datetime'].dt.hour  # type: ignore

    df_usage_and_price["c_per_kwh_variable"] = df_usage_and_price["c_per_kwh_variable"] / 10 # Convert €/MWh to c€/kWh

    # Fill missing prices with average price
    df_usage_and_price.loc[df_usage_and_price["c_per_kwh_variable"].isna(), "c_per_kwh_variable"] = df_usage_and_price["c_per_kwh_variable"].mean()
    df_usage_and_price["scaled_kwh_usage"] = df_usage_and_price["raw_kwh_usage"] * SCALE_FACTOR
    df_usage_and_price["c_fixed_costs_kwh"] = COMBINED_ADDITIONAL_COSTS

    df_usage_and_price["c_variable_and_fixed_per_kwh"] = df_usage_and_price["c_per_kwh_variable"] + df_usage_and_price["c_fixed_costs_kwh"]

    df_usage_and_price["c_total_variable_cost"] = (
        (df_usage_and_price["scaled_kwh_usage"] / config.HEAT_PUMP_COP * (df_usage_and_price["c_variable_and_fixed_per_kwh"])) 
    )

    df_usage_and_price["c_per_kwh_flat_rate_cost"] = config.FLAT_RATE_C_PER_KWH
    df_usage_and_price["c_total_flat_cost"] = (df_usage_and_price["c_per_kwh_flat_rate_cost"] * df_usage_and_price["scaled_kwh_usage"] / config.HEAT_PUMP_COP)

    return df_usage_and_price

def get_usage_data(config: ElectricityConfig | None = None) -> tuple[ElectricityConfig, pd.DataFrame]:
    """Function which loads and processes all necessary data and configuration."""
    if config is None:
        config = ElectricityConfig.from_yaml("config/config.yaml")
    df_hourly = process_heat_energy_profile()
    day_ahead_hourly = process_day_ahead_data(df_hourly)
    df_usage_and_price = calculate_usage_and_price(df_hourly, day_ahead_hourly, config=config)

    return config, df_usage_and_price

def get_max_heat_demand(df_usage_and_price: pd.DataFrame) -> float:
    """Function which calculates the maximum heat demand from the usage data."""
    max_heat_demand = df_usage_and_price["scaled_kwh_usage"].max()
    return max_heat_demand
