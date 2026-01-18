import pandas as pd
from src.data_processing import ElectricityConfig
from src.pulp_optimiser import BatteryDispatchResult


def calculate_projections(usage_df: pd.DataFrame, tes_size: float, heat_pump_size: float, years: int, config: ElectricityConfig | None = None) -> pd.DataFrame:
    """Calculate breakeven analysis based on usage data and configuration.

    Args:
        usage_df (pd.DataFrame): DataFrame containing electricity usage data.
        tes_size (float): Size of the TES in kWh.
        heat_pump_size (float): Size of the heat pump in kW.
        years (int): Investment duration in years.
        config (ElectricityConfig | None): Configuration parameters. If None, loads from 'config.yaml'.

    Returns:
        pd.DataFrame: DataFrame containing breakeven analysis results.
    """
    if config is None:
        config = ElectricityConfig.from_yaml("config/config.yaml")

    usage_df = usage_df.copy()
    usage_df["c_total_gas_cost"] = (
        usage_df["scaled_kwh_heat_usage"] * config.GAS_HEATING_C_PER_KWH
    )

    monthly_costs = usage_df.resample("MS", on="datetime").agg({
            "c_total_variable_cost": "sum",
            "c_variable_total_cost_without_battery": "sum",
            "c_total_flat_cost": "sum",
            "c_total_gas_cost": "sum"
    }).reset_index()    


    tes_capex_opti = tes_size * config.TES_COST_PER_KWH  
    heat_pump_capex_opti = heat_pump_size * config.HEAT_PUMP_COST_PER_KW
    tes_capex_default = config.TES_SIZE_KWH * config.TES_COST_PER_KWH  
    heat_pump_capex_default = config.HEAT_PUMP_SIZE_KW * config.HEAT_PUMP_COST_PER_KW

    dt_index = pd.date_range(
        start=monthly_costs["datetime"].min(),
        end=monthly_costs["datetime"].max() + pd.DateOffset(years=years-1),
        freq="MS"
    )

    df_dates = pd.DataFrame({"datetime": dt_index})

    monthly_costs_extended = pd.merge(
        df_dates,
        monthly_costs,
        on="datetime",
        how="left"
    )

    print(f"Last date in extended monthly costs: {monthly_costs_extended['datetime'].max()}")

    for d in monthly_costs_extended.datetime:
        if d <= monthly_costs_extended["datetime"].max() - pd.DateOffset(years=1):
            next_year = (d + pd.DateOffset(years=1)).to_period('M').to_timestamp()
            
            # Get scalar values instead of Series
            current_variable_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_total_variable_cost"].values[0] # type: ignore
            # We calculate this as the LP optimied cost and then add the battery opex and capex later
            current_optimised_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_variable_total_cost_without_battery"].values[0] # type: ignore
            current_fixed_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_total_flat_cost"].values[0] # type: ignore
            current_gas_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_total_gas_cost"].values[0] # type: ignore
            
            ####################################
            ### - Variable Cost Projection - ###
            ####################################

            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_total_variable_cost"] = current_variable_cost * (1 + config.ELECTRICITY_INFLATION_RATE)
            
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_variable_total_cost_without_battery"] = (
                current_optimised_cost * (1 + config.ELECTRICITY_INFLATION_RATE)
            )
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_total_flat_cost"] = current_fixed_cost * (1 + config.ELECTRICITY_INFLATION_RATE)
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_total_gas_cost"] = current_gas_cost * (1 + config.GAS_INFLATION_RATE)

    monthly_costs_extended["c_total_variable_cost_cumulative"] = (
        monthly_costs_extended["c_total_variable_cost"].cumsum() + ((tes_capex_default + heat_pump_capex_default) * 100)
    )
    monthly_costs_extended["c_variable_total_cost_with_battery_cumulative"] = (
         (monthly_costs_extended["c_variable_total_cost_without_battery"].cumsum() + tes_capex_opti + heat_pump_capex_opti) * 100
    )
    monthly_costs_extended["c_total_flat_cost_cumulative"] = (
        monthly_costs_extended["c_total_flat_cost"].cumsum() + ((tes_capex_default + heat_pump_capex_default) * 100)
    )

    monthly_costs_extended["c_total_gas_cost_cumulative"] = (
        monthly_costs_extended["c_total_gas_cost"].cumsum()
    )

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
    if inflation_rate > 0:
        return cost * ((1 + inflation_rate) ** years - 1) / inflation_rate
    else:
        return cost * years

def calculate_inflation_adjusted_costs(usage_data: pd.DataFrame, optimisation_results: BatteryDispatchResult, investment_duration_years: int, config: ElectricityConfig) -> dict[str, float]:
        optimised_inflation_adjusted_cost_without_battery = inflation_adjusted_cost(
            usage_data['c_variable_total_cost_without_battery'].sum(), 
            investment_duration_years, 
            config.ELECTRICITY_INFLATION_RATE
            ) 
        
        optimised_inflation_adjusted_cost_without_battery_delta = inflation_adjusted_cost(
            usage_data['c_total_flat_cost'].sum()/100 - usage_data['c_variable_total_cost_with_battery'].sum(), 
            investment_duration_years, 
            config.ELECTRICITY_INFLATION_RATE
            ) 

        # To calculate this, we need to tkae the usage cost excluding battery CAPEX and then add the CAPEX in separately at the end
        optimised_inflation_adjusted_with_battery_cost = (
            optimised_inflation_adjusted_cost_without_battery + 
            ((optimisation_results.tes_size * config.TES_COST_PER_KWH +   # type: ignore
            optimisation_results.heat_pump_size * config.HEAT_PUMP_COST_PER_KW) # type: ignore
             )
        )
        
        # Take the difference between flat cost and variable cost with battery and then add the CAPEX at the end
        optimised_inflation_adjusted_vs_flat_cost_delta = inflation_adjusted_cost(
            usage_data['c_total_flat_cost'].sum()/100 - usage_data['c_variable_total_cost_without_battery'].sum(), 
            investment_duration_years, 
            config.ELECTRICITY_INFLATION_RATE
            ) + ((config.TES_SIZE_KWH - optimisation_results.tes_size) * config.TES_COST_PER_KWH + (config.HEAT_PUMP_SIZE_KW - optimisation_results.heat_pump_size) * config.HEAT_PUMP_COST_PER_KW)  # type: ignore
        
        
        # The difference here is also the difference between the optimised battery and heat pump cost
        optimised_inflation_adjusted_vs_variable_cost_delta = inflation_adjusted_cost(
            usage_data['c_total_variable_cost'].sum()/100 - usage_data['c_variable_total_cost_without_battery'].sum(), 
            investment_duration_years, 
            config.ELECTRICITY_INFLATION_RATE
            ) + ((config.TES_SIZE_KWH - optimisation_results.tes_size) * config.TES_COST_PER_KWH + (config.HEAT_PUMP_SIZE_KW - optimisation_results.heat_pump_size) * config.HEAT_PUMP_COST_PER_KW)  # type: ignore
        
        return {
            "optimised_inflation_adjusted_cost_without_battery": optimised_inflation_adjusted_cost_without_battery,
            "optimised_inflation_adjusted_cost_without_battery_delta": optimised_inflation_adjusted_cost_without_battery_delta,
            "optimised_inflation_adjusted_with_battery_cost": optimised_inflation_adjusted_with_battery_cost,
            "optimised_inflation_adjusted_vs_flat_cost_delta": optimised_inflation_adjusted_vs_flat_cost_delta,
            "optimised_inflation_adjusted_vs_variable_cost_delta": optimised_inflation_adjusted_vs_variable_cost_delta
        }
