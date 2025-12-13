import pandas as pd
from src.data_processing import ElectricityConfig


def calculate_projections(usage_df: pd.DataFrame, config: ElectricityConfig | None = None) -> pd.DataFrame:
    """Calculate breakeven analysis based on usage data and configuration.

    Args:
        usage_df (pd.DataFrame): DataFrame containing electricity usage data.
        config (ElectricityConfig | None): Configuration object, if None loads from 'config.yaml'.

    Returns:
        pd.DataFrame: DataFrame containing breakeven analysis results.
    """
    if config is None:
        config = ElectricityConfig.from_yaml("config.yaml")

    monthly_costs = usage_df.resample("MS", on="datetime").agg({
            "c_total_variable_cost": "sum",
            "c_variable_total_cost_with_battery": "sum"
    }).reset_index()    


    dt_index = pd.date_range(
        start=monthly_costs["datetime"].min(),
        end=monthly_costs["datetime"].max() + pd.Timedelta(weeks=(52*10)+2),   # last 15-minute slot of the year
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
            current_variable_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_total_variable_cost"].values[0]
            current_shifted_cost = monthly_costs_extended.loc[monthly_costs_extended["datetime"] == d, "c_variable_total_cost_with_battery"].values[0]
            
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_total_variable_cost"] = current_variable_cost * (1 + config.INFLATION_RATE)
            monthly_costs_extended.loc[monthly_costs_extended["datetime"] == next_year, "c_variable_total_cost_with_battery"] = current_shifted_cost * (1 + config.INFLATION_RATE)
            
    monthly_costs_extended["c_total_variable_cost_cumulative"] = monthly_costs_extended["c_total_variable_cost"].cumsum()
    monthly_costs_extended["c_variable_total_cost_with_battery_cumulative"] = monthly_costs_extended["c_variable_total_cost_with_battery"].cumsum()

    return monthly_costs_extended