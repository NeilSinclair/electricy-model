import pandas as pd
from src.data_processing import ElectricityConfig


def calculate_projections(usage_df: pd.DataFrame, battery_size: float, config: ElectricityConfig | None = None) -> pd.DataFrame:
    """Calculate breakeven analysis based on usage data and configuration.

    Args:
        usage_df (pd.DataFrame): DataFrame containing electricity usage data.
        battery_size (float): Size of the battery in kWh.
        config (ElectricityConfig | None): Configuration object, if None loads from 'config.yaml'.

    Returns:
        pd.DataFrame: DataFrame containing breakeven analysis results.
    """
    if config is None:
        config = ElectricityConfig.from_yaml("config/config.yaml")

    monthly_costs = usage_df.resample("MS", on="datetime").agg({
            "c_total_variable_cost": "sum",
            "c_variable_total_cost_with_battery": "sum",
            "c_total_flat_cost": "sum"
    }).reset_index()    

    battery_capex = battery_size * config.BATTERY_COST_PER_KWH 

    dt_index = pd.date_range(
        start=monthly_costs["datetime"].min(),
        end=monthly_costs["datetime"].max() + pd.Timedelta(weeks=(52*config.INVESTMENT_DURATION_YEARS)+2),  
        freq="MS"
    )

    df_dates = pd.DataFrame({"datetime": dt_index})

    monthly_costs_extended = pd.merge(
        df_dates,
        monthly_costs,
        on="datetime",
        how="left"
    )

    for d in monthly_costs_extended.datetime:
        if d <= monthly_costs_extended["datetime"].max() - pd.Timedelta(weeks=52):
            next_year = (d + pd.DateOffset(years=1)).to_period('M').to_timestamp()
            
            # Get scalar values instead of Series
            current_variable_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_total_variable_cost"].values[0] # type: ignore
            current_optimised_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_variable_total_cost_with_battery"].values[0] # type: ignore
            current_fixed_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_total_flat_cost"].values[0] # type: ignore
            
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_total_variable_cost"] = current_variable_cost * (1 + config.INFLATION_RATE)
            
            # Add in the battery opex spread over the year
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_variable_total_cost_with_battery"] = (
                (current_optimised_cost + (battery_capex / 365 * config.OPEX_PERCENT_OF_CAPEX)) * (1 + config.INFLATION_RATE)
            )
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_total_flat_cost"] = current_fixed_cost * (1 + config.INFLATION_RATE)

    monthly_costs_extended["c_total_variable_cost_cumulative"] = monthly_costs_extended["c_total_variable_cost"].cumsum()
    monthly_costs_extended["c_variable_total_cost_with_battery_cumulative"] = monthly_costs_extended["c_variable_total_cost_with_battery"].cumsum()
    monthly_costs_extended["c_total_flat_cost_cumulative"] = monthly_costs_extended["c_total_flat_cost"].cumsum()

    return monthly_costs_extended

def extend_raw_data(df: pd.DataFrame, years: int, config: ElectricityConfig) -> pd.DataFrame:
    """Extend raw usage data by a specified number of years.

    Args:
        df (pd.DataFrame): Original DataFrame containing usage data.
        years (int): Number of years to extend the data.
        config (ElectricityConfig): Configuration object.

    Returns:
        pd.DataFrame: Extended DataFrame.
    """
    cols = ["c_variable_and_fixed_per_kwh", "raw_kwh_usage", "scaled_kwh_usage", "c_total_variable_cost", "c_total_flat_cost"]
    df = df.copy()

    if True:
        for col in ["raw_kwh_usage", "scaled_kwh_usage","c_total_variable_cost", "c_total_flat_cost"]:
            df[col] = df[col] * years
        return df
    else:
        dt_index = pd.date_range(
            start=df["datetime"].min(),
            end=df["datetime"].max() + pd.DateOffset(years=(years-1)),
            freq="h"
        )

        df_dates = pd.DataFrame({"datetime": dt_index})

        df_extended = pd.merge(
            df_dates,
            df[["datetime"] + cols],
            on="datetime",
            how="left"
        )

        # Fill missing values by repeating the original data pattern
        original_length = len(df)
        for i in range(len(df_extended)):
            for col in cols:
                if pd.isna(df_extended.loc[i, col]):
                    # Get the inflation factor over time
                    # multiplier = (1 + config.INFLATION_RATE) ** (i // 365) if col == "c_variable_and_fixed_per_kwh" else 1
                    multiplier = 1
                    df_extended.loc[i, col] = df.loc[i % original_length, col] * (multiplier) # type: ignore

        return df_extended

def inflation_adjusted_cost(cost: float, years: int, inflation_rate: float) -> float:
    """Calculate the inflation-adjusted cost over a number of years.

    Args:
        cost (float): Initial cost.
        years (int): Number of years.
        inflation_rate (float): Annual inflation rate.

    Returns:
        float: Inflation-adjusted cost.
    """
    return cost * ((1 + inflation_rate) ** years - 1) / inflation_rate

def calculate_inflation_adjusted_costs(usage_data: pd.DataFrame, optimised_total_cost: float,investment_duration_years: int, config: ElectricityConfig) -> dict[str, float]:
        optimised_inflation_adjusted_cost_without_battery = inflation_adjusted_cost(
            usage_data['c_variable_total_cost_without_battery'].sum(), 
            investment_duration_years, 
            config.INFLATION_RATE
            ) 
        
        optimised_inflation_adjusted_cost_without_battery_delta = inflation_adjusted_cost(
            usage_data['c_total_flat_cost'].sum()/100 - usage_data['c_variable_total_cost_with_battery'].sum(), 
            investment_duration_years, 
            config.INFLATION_RATE
            ) 
        
        optimised_inflation_adjusted_with_battery_cost = inflation_adjusted_cost(
            optimised_total_cost, 
            investment_duration_years, 
            config.INFLATION_RATE
            )
        
        optimised_inflation_adjusted_vs_flat_cost_delta = inflation_adjusted_cost(
            usage_data['c_total_flat_cost'].sum()/100 - optimised_total_cost, 
            investment_duration_years, 
            config.INFLATION_RATE
            )
        
        optimised_inflation_adjusted_vs_variable_cost_delta = inflation_adjusted_cost(
            usage_data['c_total_variable_cost'].sum()/100 - optimised_total_cost, 
            investment_duration_years, 
            config.INFLATION_RATE
            )
        
        return {
            "optimised_inflation_adjusted_cost_without_battery": optimised_inflation_adjusted_cost_without_battery,
            "optimised_inflation_adjusted_cost_without_battery_delta": optimised_inflation_adjusted_cost_without_battery_delta,
            "optimised_inflation_adjusted_with_battery_cost": optimised_inflation_adjusted_with_battery_cost,
            "optimised_inflation_adjusted_vs_flat_cost_delta": optimised_inflation_adjusted_vs_flat_cost_delta,
            "optimised_inflation_adjusted_vs_variable_cost_delta": optimised_inflation_adjusted_vs_variable_cost_delta
        }
