import pandas as pd
from src.data_processing import ElectricityConfig
from src.pulp_optimiser import BatteryDispatchResult


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
            "c_variable_total_cost_without_battery": "sum",
            "c_total_flat_cost": "sum"
    }).reset_index()    

    battery_capex = battery_size * config.BATTERY_COST_PER_KWH  * (1 + config.TAX_RATE)

    dt_index = pd.date_range(
        start=monthly_costs["datetime"].min(),
        # end=monthly_costs["datetime"].max() + pd.Timedelta(weeks=(52*config.INVESTMENT_DURATION_YEARS)+2),  
        end=monthly_costs["datetime"].max() + pd.DateOffset(years=config.INVESTMENT_DURATION_YEARS-1),
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
            # We calculate this as the LP optimied cost and then add the battery opex and capex later
            current_optimised_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_variable_total_cost_without_battery"].values[0] # type: ignore
            current_fixed_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_total_flat_cost"].values[0] # type: ignore
            
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_total_variable_cost"] = current_variable_cost * (1 + config.INFLATION_RATE)
            
            # Add in the battery opex spread over the year; note the tax was added the c_variable_total_cost_with_battery before
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_variable_total_cost_without_battery"] = (
                (current_optimised_cost * (1 + config.INFLATION_RATE)) + ((battery_capex / 365 * config.OPEX_PERCENT_OF_CAPEX) * (1 + config.INFLATION_RATE))
            )
            # Tax was added when this data was initially processed
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_total_flat_cost"] = current_fixed_cost * (1 + config.INFLATION_RATE)

    monthly_costs_extended["c_total_variable_cost_cumulative"] = monthly_costs_extended["c_total_variable_cost"].cumsum()
    monthly_costs_extended["c_variable_total_cost_with_battery_cumulative"] = monthly_costs_extended["c_variable_total_cost_without_battery"].cumsum() + battery_capex
    monthly_costs_extended["c_total_flat_cost_cumulative"] = monthly_costs_extended["c_total_flat_cost"].cumsum()

    return monthly_costs_extended

def inflation_adjusted_cost(cost: float, years: int, inflation_rate: float) -> float:
    """Calculate the inflation-adjusted cost over a number of years.

    Args:
        cost (float): Initial cost.
        years (int): Number of years.
        inflation_rate (float): Annual inflation rate.

    Returns:
        float: Inflation-adjusted cost.
    """
    # We take years - 1 because we already have the first year
    return cost * ((1 + inflation_rate) ** years - 1) / inflation_rate

def calculate_inflation_adjusted_costs(usage_data: pd.DataFrame, optimisation_results: BatteryDispatchResult, investment_duration_years: int, config: ElectricityConfig) -> dict[str, float]:
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

        # To calculate this, we need to tkae the usage cost excluding battery CAPEX and then add the CAPEX in separately at the end
        optimised_inflation_adjusted_with_battery_cost = (
             optimised_inflation_adjusted_cost_without_battery + optimisation_results.battery_size * config.BATTERY_COST_PER_KWH * (1 + config.TAX_RATE)
        )
        
        # Take the difference between flat cost and variable cost with battery and then add the CAPEX at the end
        optimised_inflation_adjusted_vs_flat_cost_delta = inflation_adjusted_cost(
            usage_data['c_total_flat_cost'].sum()/100 - usage_data['c_variable_total_cost_without_battery'].sum(), 
            investment_duration_years, 
            config.INFLATION_RATE
            ) - optimisation_results.battery_size * config.BATTERY_COST_PER_KWH * (1 + config.TAX_RATE)
        
        optimised_inflation_adjusted_vs_variable_cost_delta = inflation_adjusted_cost(
            usage_data['c_total_variable_cost'].sum()/100 - optimisation_results.total_cost, 
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
