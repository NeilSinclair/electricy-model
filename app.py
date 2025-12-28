import yaml
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter

import logging

from src.data_processing import get_usage_data, ElectricityConfig
from src.pulp_optimiser import solve_battery_dispatch_pulp, BatteryDispatchResult
from src.cost_calculations import calculate_projections, inflation_adjusted_cost, calculate_inflation_adjusted_costs

logging.basicConfig(level=logging.INFO)

# Load configuration from YAML
@st.cache_data
def load_config():
    files = sorted(
        (p for p in Path("config").iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    logging.info(f"Loading configuration from {files[0]}")
    return ElectricityConfig.from_yaml(files[0]) # type: ignore

def reset_config_to_default():
    return ElectricityConfig()

def main():
    st.set_page_config(page_title="Electricity Cost Model", layout="wide")
    if 'usage_data' not in st.session_state:
        _, st.session_state.usage_data = get_usage_data()
        st.session_state.usage_data = st.session_state.usage_data[st.session_state.usage_data.datetime < '2025-12-01']
        st.session_state.usage_data['c_variable_total_cost_with_battery'] = 0.0
    
    if 'config' not in st.session_state:
        st.session_state.config = load_config()

    
    if 'original_cost' not in st.session_state:
        st.session_state.original_cost = (
            (st.session_state.usage_data['c_variable_and_fixed_per_kwh'] * 
            st.session_state.usage_data['raw_kwh_usage']).sum() * (1 + st.session_state.config.TAX_RATE) / 100
        )
    
    if 'optimisation_results' not in st.session_state:
        st.session_state.optimisation_results = BatteryDispatchResult(
            grid=[],
            charge=[],
            discharge=[],
            soc=[],
            mode=[],
            price=[],
            total_cost=None
        )

    if 'optimisation_results_with_battery' not in st.session_state:
        st.session_state.optimisation_results_with_battery = None

    if 'breakeven' not in st.session_state:
        st.session_state.breakeven = None

    if 'profile_with_battery' not in st.session_state:
        st.session_state.profile_with_battery = None

    if 'investment_duration_years' not in st.session_state:
        st.session_state.investment_duration_years = st.session_state.config.INVESTMENT_DURATION_YEARS

    if 'inflation_adjusted_costs' not in st.session_state:
        st.session_state.inflation_adjusted_costs = {}
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select Option:", ["Cost Modelling", "Raw Data"])
    
    if option == "Cost Modelling":
        show_cost_modelling(st.session_state.config)
    elif option == "Raw Data":
        show_raw_data(st.session_state.config)

def calculate_cost_with_battery(optimisation_results: BatteryDispatchResult, config: ElectricityConfig) -> float:
    if optimisation_results.total_cost is None:
        return 0.0
    battery_capex = optimisation_results.battery_size * config.BATTERY_COST_PER_KWH
    battery_opex = battery_capex * config.OPEX_PERCENT_OF_CAPEX
    total_battery_cost = (battery_capex + battery_opex) * (1 + config.TAX_RATE)
    return optimisation_results.total_cost + total_battery_cost

def show_cost_modelling(config):
    st.title("Cost Modelling")
    
    st.header("Configuration Parameters")
    on = st.toggle("Heat Battery Modelling", value=False)
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Usage & Rates")
        st.session_state.config.SOC_FACTOR = st.number_input(
            "SOC factor (Total usable capacity)", 
            value=config.SOC_FACTOR,
            step=0.05
        )
        st.session_state.config.FLAT_RATE_C_PER_KWH = st.number_input(
            "Flat Rate (c/kWh)", 
            value=config.FLAT_RATE_C_PER_KWH,
            step=1.0
        )
        st.session_state.config.GAS_HEATING_C_PER_KWH = st.number_input(
            "Gas Heating Cost (c/kWh)", 
            value=config.GAS_HEATING_C_PER_KWH,
            step=0.1,
            format="%.2f",
            help="Cost of gas heating per kWh."
        )
        st.session_state.config.GAS_CONVERSION_RATIO = st.number_input(
            "Gas Conversion Ratio", 
            value=config.GAS_CONVERSION_RATIO,
            step=0.1,
            format="%.2f",
            help=(
                "Conversion ratio from gas to heat energy from electricity. E.g., a value of 3.2 means "
                "3.2 kWh of gas produces 1 kWh of the heat energy produced by electricity."
            )
        )

    
    with col2:
        st.subheader("Battery Parameters")
        st.session_state.config.BATTERY_INEFFICIENCY_FACTOR = st.number_input(
            "Battery Inefficiency Factor", 
            value=config.BATTERY_INEFFICIENCY_FACTOR,
            step=0.01,
            format="%.2f",
            help=(
                "Value between 0 and 1 representing the efficiency of charging/discharging the battery. "
                "A value of 0.9 would mean 90% efficiency for charging and for discharging, leaing to a "
                "90% * 90% = 81% round-trip efficiency."
            )
        )
        st.session_state.config.BATTERY_SIZE_KWH = st.number_input(
            "Battery Size (kWh)", 
            value=config.BATTERY_SIZE_KWH,
            step=5,
            disabled=True
        )
        st.session_state.config.BATTERY_POWER = st.number_input(
            "Battery Power (kW)", 
            value=config.BATTERY_POWER,
            step=5,
            help="Maximum power the battery can charge or discharge at any one time before the efficiency factor is considered."
        )
   
    
    with col3:
        st.subheader("Battery Investment")
        st.session_state.config.BATTERY_COST_PER_KWH = st.number_input(
            "Battery Cost per kWh (€/kWh)", 
            value=config.BATTERY_COST_PER_KWH,
            step=10.0
        )
        st.session_state.config.OPEX_PERCENT_OF_CAPEX = st.number_input(
            "OPEX (% / 100 of CAPEX)", 
            value=config.OPEX_PERCENT_OF_CAPEX ,
            step=0.01,
            format="%.2f",
            disabled=True,
        )
        if st.session_state.config.OPEX_PERCENT_OF_CAPEX is not None:
            st.session_state.config.OPEX_PERCENT_OF_CAPEX = st.session_state.config.OPEX_PERCENT_OF_CAPEX 
        else:
            st.session_state.config.OPEX_PERCENT_OF_CAPEX = st.session_state.config.OPEX_PERCENT_OF_CAPEX
        st.session_state.config.INFLATION_RATE = st.number_input(
            "Inflation Rate (% / 100)", 
            value=config.INFLATION_RATE,
            step=0.01,
            format="%.2f"
        )
        st.session_state.investment_duration_years = st.number_input(
            "Investment Duration (years)", 
            value=st.session_state.config.INVESTMENT_DURATION_YEARS,
            step=1,
            min_value=1,
            help=(
                "Number of years over which to project the cost savings. To speed up calculations, we optimise the battery size over a single year "
                "using the total cost divided by the number of investment years. The full cost projection is then calculated separately for "
                "the total number of years."
            )
        )
    
    # Additional costs section
    st.subheader("Additional Costs (c/kWh)")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.session_state.config.NETWORK_USAGE = st.number_input(
            "Network Usage", 
            value=config.NETWORK_USAGE,
            step=0.1,
            format="%.2f"
        )
        st.session_state.config.TAX_RATE = st.number_input(
            "Tax Rate (% / 100)", 
            value=config.TAX_RATE,
            step=0.01,
            format="%.2f"
        )
    
    with col5:  
        st.session_state.config.ELECTRICITY_TAX = st.number_input(
            "Electricity Tax", 
            value=config.ELECTRICITY_TAX,
            step=0.1,
            format="%.2f"
        )
        st.session_state.config.ADDITIONAL_COST = st.number_input(
            "Additional Cost", 
            value=config.ADDITIONAL_COST,
            step=0.1,
            format="%.2f"
        )
    
    with col6:
        st.session_state.config.KONZESSION = st.number_input(
            "Konzession", 
            value=config.KONZESSION,
            step=0.1,
            format="%.2f"
        )
        st.session_state.config.CHP_SURCHARGE = st.number_input(
            "CHP Surcharge", 
            value=config.CHP_SURCHARGE,
            step=0.1,
            format="%.2f"
        )

    if st.button("Reset to Default"):
        st.session_state.config = ElectricityConfig() # resets to default values
        st.rerun()
    
    if st.button("Save Configuration", disabled=True):
        with open(f"config/config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml", "w") as f:
            yaml_data = {
                "ANNUAL_USAGE": st.session_state.config.ANNUAL_USAGE,
                "BASELINE_USAGE_MWH": st.session_state.config.BASELINE_USAGE_MWH,
                "FLAT_RATE_C_PER_KWH": st.session_state.config.FLAT_RATE_C_PER_KWH,
                "BATTERY_INEFFICIENCY_FACTOR": st.session_state.config.BATTERY_INEFFICIENCY_FACTOR,
                "NETWORK_USAGE": st.session_state.config.NETWORK_USAGE,
                "TAX_RATE": st.session_state.config.TAX_RATE,
                "ELECTRICITY_TAX": st.session_state.config.ELECTRICITY_TAX,
                "ADDITIONAL_COST": st.session_state.config.ADDITIONAL_COST,
                "KONZESSION": st.session_state.config.KONZESSION,
                "CHP_SURCHARGE": st.session_state.config.CHP_SURCHARGE,
                "BATTERY_COST_PER_KWH": st.session_state.config.BATTERY_COST_PER_KWH,
                "OPEX_PERCENT_OF_CAPEX": st.session_state.config.OPEX_PERCENT_OF_CAPEX,
                "BATTERY_SIZE_KWH": st.session_state.config.BATTERY_SIZE_KWH,
                "BATTERY_POWER": st.session_state.config.BATTERY_POWER,
                "SOC_FACTOR": st.session_state.config.SOC_FACTOR,
                "INFLATION_RATE": st.session_state.config.INFLATION_RATE
            }
            yaml.dump(yaml_data, f)
        st.success(f"Configuration saved to config/config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    
    st.markdown("---")
    
    # Cost calculations (using dummy values for now)
    st.header("Cost Analysis")

    if st.button("Optimize Costs"):
        with st.spinner("Running optimization. This will take a moment..."):

            st.session_state.optimisation_results : BatteryDispatchResult = solve_battery_dispatch_pulp( # type: ignore
                price=st.session_state.usage_data['c_variable_and_fixed_per_kwh'],
                demand=st.session_state.usage_data['scaled_kwh_usage'],
                years = st.session_state.investment_duration_years,
                config=st.session_state.config,
            )


            # Put total cost into €/kWh with tax
            st.session_state.optimisation_results.total_cost = (
                st.session_state.optimisation_results.total_cost / 100 * (1 + st.session_state.config.TAX_RATE) # type: ignore
            )   

            # This gives the cost without CAPEX in €; we already add tax in here
            st.session_state.usage_data['c_variable_total_cost_without_battery'] = (
                st.session_state.optimisation_results.grid * 
                st.session_state.usage_data['c_variable_and_fixed_per_kwh'] * 
                (1 + st.session_state.config.TAX_RATE) / 100
            )

            # Put the cost the optimised cost with battery usage into another variable
            st.session_state.usage_data['c_variable_total_cost_with_battery'] = (
                st.session_state.usage_data['c_variable_total_cost_without_battery'] # + battery_capex

            )

            # Project this date investment_period years into the future
            st.session_state.breakeven = calculate_projections(
                st.session_state.usage_data, 
                st.session_state.optimisation_results.battery_size, # type: ignore
                config=st.session_state.config,
            )

        # When we optimise for the cost whilst also optimising the battery size, this isn't needed 
        # st.session_state.optimisation_results_with_battery = calculate_cost_with_battery(
        #     st.session_state.optimisation_results, 
        #     st.session_state.config
        # )

        st.session_state.profile_with_battery = (
            pd.DataFrame({
                'datetime': st.session_state.usage_data['datetime'],
                'raw_kwh_usage': st.session_state.usage_data['raw_kwh_usage'],
                'grid_kwh_usage': st.session_state.optimisation_results.grid,
                'soc': st.session_state.optimisation_results.soc,
                'battery_charge_kwh': st.session_state.optimisation_results.charge,
                'battery_discharge_kwh': st.session_state.optimisation_results.discharge,   
            }
        ))

        st.session_state.inflation_adjusted_costs = calculate_inflation_adjusted_costs(
            usage_data = st.session_state.usage_data, 
            optimisation_results = st.session_state.optimisation_results,
            investment_duration_years = st.session_state.investment_duration_years, 
            config = st.session_state.config
        )

        st.success("Optimization complete!")

    st.write(f"Inflation adjusted cost for **{st.session_state.investment_duration_years} years**.")
    col_result1, col_result2 = st.columns(2)

    with col_result1:
        st.metric(
            label="Gas Heating Cost",
            value=(f"""
                {inflation_adjusted_cost(
                    st.session_state.usage_data['scaled_kwh_usage'].sum() * st.session_state.config.GAS_HEATING_C_PER_KWH * st.session_state.config.GAS_CONVERSION_RATIO / 100 * (1 + st.session_state.config.TAX_RATE),
                    st.session_state.investment_duration_years,
                    st.session_state.config.INFLATION_RATE
                ):,.0f} €"""
            ),
            delta=None,
            help="Usage of gas heating system in kWh over the period."
        )

    with col_result2:
        st.metric(
            label="Original Flat Rate Cost",
            value=(f"""{inflation_adjusted_cost(
                st.session_state.usage_data['c_total_flat_cost'].sum()/100, 
                st.session_state.investment_duration_years, 
                st.session_state.config.INFLATION_RATE
                ) :,.0f} €"""
            ),
            delta=None,
            help="This is the inflation adjusted electricity cost for the period based on a flat rate without any energy stored in the battery."
        )

    col_result1, col_result2 = st.columns(2)
    with col_result1:
        st.metric(
            label="Variable Rate Cost (No Battery)",
            value=(f"""{inflation_adjusted_cost(
                st.session_state.usage_data['c_total_variable_cost'].sum()/100, 
                st.session_state.investment_duration_years, 
                st.session_state.config.INFLATION_RATE
                ) :,.0f} €"""
            ),
            delta=None,
            help="This is the inflation adjusted variable cost for the period if no battery is purchased and the user simply switches to a variable tarrif"
        )

    with col_result2:
        st.metric(
            label="Variable Rate Cost (Incl. Battery CAPEX)",
            value=(
                f"{st.session_state.inflation_adjusted_costs['optimised_inflation_adjusted_with_battery_cost']:,.0f} €" 
                if st.session_state.inflation_adjusted_costs else "N/A"
            ),
            delta=None,
            help="This is the inflation adjusted variable cost for the period with energy stored in the battery."
        )

    col_result1, col_result2 = st.columns(2)

    with col_result1:
        st.metric(
            label="Optimised battery size (kWh)",
            value=(
                f"{st.session_state.optimisation_results.battery_size:,.0f} kWh" 
                if st.session_state.optimisation_results.battery_size is not None else "N/A"
            ),
            delta=None,
            help="This is the optimised battery size based on the current configuration. If the battery size is 0 kWh this indicates that a battery is not cost effective under the current parameters."
        )
    
    with col_result2:
        st.metric(
            label="Battery Capex",
            value=(
                f"{st.session_state.optimisation_results.battery_size * st.session_state.config.BATTERY_COST_PER_KWH * (1 + st.session_state.config.TAX_RATE):,.0f} €" 
                if st.session_state.optimisation_results.battery_size is not None else "N/A"
            ),
            delta=None,
            help="This is the capital expenditure (Capex) for the optimised battery size."
        )
        
    col_result1, col_result2 = st.columns(2)    
    with col_result1:
        if st.session_state.inflation_adjusted_costs:
            savings_vs_fixed_perc = (
                st.session_state.inflation_adjusted_costs['optimised_inflation_adjusted_vs_flat_cost_delta'] /
                inflation_adjusted_cost(
                    st.session_state.usage_data['c_total_flat_cost'].sum()/100, 
                    st.session_state.investment_duration_years, 
                    st.session_state.config.INFLATION_RATE
                    )
            )
        else:
            savings_vs_fixed_perc = "N/A"

        st.metric(
            label="**Savings:** Variable rate with battery vs flat cost",
            value=(f"{st.session_state.inflation_adjusted_costs['optimised_inflation_adjusted_vs_flat_cost_delta']:,.0f} €"
                   if st.session_state.inflation_adjusted_costs else "N/A"
                   ),
            delta=(
                f"{savings_vs_fixed_perc:,.2%} €" 
                if st.session_state.inflation_adjusted_costs else "N/A"
                )
                ,
                help="This shows the difference between the optimized variable cost with battery CAPEX included and the flat rate cost over the investment duration."
        )
    
    with col_result2:
        if st.session_state.inflation_adjusted_costs:
            savings_vs_variable_perc = (
                st.session_state.inflation_adjusted_costs['optimised_inflation_adjusted_vs_variable_cost_delta'] /
                inflation_adjusted_cost(
                    st.session_state.usage_data['c_total_variable_cost'].sum()/100, 
                    st.session_state.investment_duration_years, 
                    st.session_state.config.INFLATION_RATE
                    )
            )
        else:
            savings_vs_variable_perc = "N/A"
        st.metric(
            label="**Savings:** Variable rate with battery vs. variable rate alone",
            value=(
                    f"{st.session_state.inflation_adjusted_costs['optimised_inflation_adjusted_vs_variable_cost_delta']:,.0f} €"
                    if st.session_state.inflation_adjusted_costs else "N/A"
                ),
            delta=(
                    f"{savings_vs_variable_perc:,.2%} €"
                    if st.session_state.inflation_adjusted_costs else "N/A"
                ),
            help="This shows the difference between the optimized variable cost with battery CAPEX included and the variable rate cost alone over the investment duration."
        )
    
    st.markdown("---")
    
    # Breakeven graph
    st.header("Breakeven Analysis")
    flat_vs_battery_tab, variable_vs_battery_tab = st.tabs(["Flat Cost vs Variable With Battery", "Variable Cost vs Variable With Battery Optimised"])
    if st.session_state.breakeven is not None:

        projections = st.session_state.breakeven.copy()
        projections["c_variable_total_cost_with_battery_cumulative"] = projections["c_variable_total_cost_with_battery_cumulative"] 
        projections["c_total_variable_cost_cumulative"] = projections["c_total_variable_cost_cumulative"] /100
        projections["c_total_flat_cost_cumulative"] = projections["c_total_flat_cost_cumulative"]/100
        projections = projections[projections['datetime'] >= '2025-01-01']

        with flat_vs_battery_tab:
            breakeven_point = projections[
                projections.c_total_flat_cost_cumulative >=
                projections.c_variable_total_cost_with_battery_cumulative
            ]

            breakeven_date = None
            if len(breakeven_point) > 0:
                breakeven_date = breakeven_point.iloc[0]["datetime"]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=projections["datetime"],
                    y=projections["c_total_flat_cost_cumulative"],
                    mode="lines",
                    name="Fixed Cost Without Battery (Cumulative)",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}€<extra></extra>",
                    line=dict(width=2),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=projections["datetime"],
                    y=projections["c_variable_total_cost_with_battery_cumulative"],
                    mode="lines",
                    name="Variable Cost With Battery (Cumulative)",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}€<extra></extra>",
                    line=dict(width=2),
                )
            )

            if breakeven_date is not None:
                fig.add_vline(
                    x=breakeven_date.to_pydatetime(),
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                )

                fig.add_annotation(
                    x=breakeven_date.to_pydatetime(),
                    y=1,
                    yref="paper",
                    text=f"Breakeven ({breakeven_date.date()})",
                    showarrow=False,
                    xanchor="left",
                    font=dict(color="red"),
                )

            fig.update_layout(
                title="Fixed Rate vs Variable Rate with Battery Investment",
                xaxis_title="Year",
                yaxis_title="Cumulative Cost (€)",
                yaxis_tickformat=",",
                hovermode="x unified",
                template="simple_white",
                legend=dict(
                    x=0.98,
                    y=0.02,
                    xanchor="right",
                    yanchor="bottom",
                )
            )

            st.plotly_chart(fig, width="stretch")

        with variable_vs_battery_tab:
            breakeven_point = projections[
                projections.c_total_variable_cost_cumulative >=
                projections.c_variable_total_cost_with_battery_cumulative
            ]

            breakeven_date = None
            if len(breakeven_point) > 0:
                breakeven_date = breakeven_point.iloc[0]["datetime"]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=projections["datetime"],
                    y=projections["c_total_variable_cost_cumulative"],
                    mode="lines",
                    name="Variable Cost Without Battery (Cumulative)",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}€<extra></extra>",
                    line=dict(width=2),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=projections["datetime"],
                    y=projections["c_variable_total_cost_with_battery_cumulative"],
                    mode="lines",
                    name="Variable Cost With Battery (Cumulative)",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}€<extra></extra>",
                    line=dict(width=2),
                )
            )

            if breakeven_date is not None:
                fig.add_vline(
                    x=breakeven_date.to_pydatetime(),
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                )

                fig.add_annotation(
                    x=breakeven_date.to_pydatetime(),
                    y=1,
                    yref="paper",
                    text=f"Breakeven ({breakeven_date.date()})",
                    showarrow=False,
                    xanchor="left",
                    font=dict(color="red"),
                )

            fig.update_layout(
                title="Variable Rate: No Battery vs Battery Investment",
                xaxis_title="Year",
                yaxis_title="Cumulative Cost (€)",
                yaxis_tickformat=",",
                hovermode="x unified",
                template="simple_white",
                legend=dict(
                    x=0.98,
                    y=0.02,
                    xanchor="right",
                    yanchor="bottom",
                )
            )

            st.plotly_chart(fig, width="stretch")

def show_raw_data(config):
    st.title("Raw Data Analysis")
    
    st.write("Displaying raw electricity data with configured parameters.")
    
    # Graph calculations
    st.session_state.usage_data['month'] = st.session_state.usage_data['datetime'].dt.month
    st.session_state.usage_data['month_name'] = st.session_state.usage_data['datetime'].dt.strftime('%B')
    st.session_state.usage_data['hour_of_day'] = st.session_state.usage_data['datetime'].dt.hour

    # Calculate average price by hour and month
    hourly_monthly_avg_cost = st.session_state.usage_data.groupby(['hour_of_day', 'month_name'])['c_variable_and_fixed_per_kwh'].mean().reset_index()
    hourly_monthly_avg_usage = st.session_state.usage_data.groupby(['hour_of_day', 'month_name'])['raw_kwh_usage'].mean().reset_index()

    # Get unique months in chronological order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
    months_in_data = [m for m in month_order if m in hourly_monthly_avg_cost['month_name'].unique()]
    
    st.subheader("1. Hourly Electricity Price")
    st.write("This graph shows the average hourly electricity prices over a typical day for each month, highlighting peak and off-peak periods.")
    
    # Graph 1 - Hourly Electricity Price 
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    for month in months_in_data:
        month_data = hourly_monthly_avg_cost[hourly_monthly_avg_cost['month_name'] == month]
        ax1.plot(month_data['hour_of_day'], month_data['c_variable_and_fixed_per_kwh'], 
                marker='o', label=month, linewidth=2)

    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Average Price [c€/kWh]')
    ax1.set_title('Average Variable + Fixed Cost per kWh by Hour of Day and Month')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24))
    
    st.pyplot(fig1)
    
    st.markdown("---")
    
    # Graph 2 - Hourly Consumption Pattern
    st.subheader("2. Daily Consumption Pattern")
    st.write("This graph illustrates typical daily electricity consumption patterns, showing peak usage times.")
    
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    for month in months_in_data:
        month_data = hourly_monthly_avg_usage[hourly_monthly_avg_usage['month_name'] == month]
        ax2.plot(month_data['hour_of_day'], month_data['raw_kwh_usage'], 
                marker='o', label=month, linewidth=2)

    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Consumption (kWh)')
    ax2.set_title('Average Electricity Consumption by Hour of Day and Month')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24))
    
    st.pyplot(fig2)
    
    st.markdown("---")
    st.write("3. Combined electricity price and consumption patterns.")

    selected_months = st.multiselect(
        "Select months to display",
        options=months_in_data,
        default="January",
        help="Select which months you want to look at in the combined graph. By default, only January is shown, but you can select multiple months to compare."
    )

    fig = go.Figure()

    # --- Price traces (left y-axis)
    for month in months_in_data:
        df_price = hourly_monthly_avg_cost[
            hourly_monthly_avg_cost["month_name"] == month
        ]

        fig.add_trace(
            go.Scatter(
                x=df_price["hour_of_day"],
                y=df_price["c_variable_and_fixed_per_kwh"],
                name=f"{month} - Price",
                yaxis="y1",
                mode="lines+markers",
                line=dict(dash="dash", width=2),
                visible=month in selected_months,
                hovertemplate="Hour %{x}<br>%{y:.2f} c€/kWh<extra></extra>",
            )
        )

    # --- Consumption traces (right y-axis)
    for month in months_in_data:
        df_usage = hourly_monthly_avg_usage[
            hourly_monthly_avg_usage["month_name"] == month
        ]

        fig.add_trace(
            go.Scatter(
                x=df_usage["hour_of_day"],
                y=df_usage["raw_kwh_usage"],
                name=f"{month} – Consumption",
                yaxis="y2",
                mode="lines+markers",
                visible=month in selected_months,
                hovertemplate="Hour %{x}<br>%{y:.2f} kWh<extra></extra>",
            )
        )

    # --- Layout
    fig.update_layout(
        title="Hourly Electricity Price & Consumption by Month",
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear",
            tick0=0,
            dtick=1,
        ),
        yaxis=dict(
            title="Average Price [c€/kWh]",
            side="left",
        ),
        yaxis2=dict(
            title="Average Consumption [kWh]",
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        template="simple_white",
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
        ),
    )

    st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    # Graph 3
    if st.session_state.profile_with_battery is not None:
        st.subheader("4. Energy Profile with Battery Optimization")
        st.write("This graph shows the updated energy profile with battery optimization applied, comparing original and new grid usage.")

        start = st.date_input("Start date: yyyy/mm/dd", datetime(2025, 1, 1))
        end = st.date_input("End date: yyyy/mm/dd", datetime(2025, 1, 7))

         # Graph 3 - Grid Usage vs Raw Usage
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        temp_df = (
            st.session_state.profile_with_battery[
                (st.session_state.profile_with_battery['datetime'].dt.date >= start) & 
                (st.session_state.profile_with_battery['datetime'].dt.date <= end)]
        )
        # ax3.plot(temp_df['datetime'], temp_df['grid_kwh_usage'], label='Grid Usage')
        ax3.plot(temp_df['datetime'], temp_df['raw_kwh_usage'], label='Original Grid Usage')
        ax3.plot(
            temp_df['datetime'], 
            temp_df['grid_kwh_usage'], 
            label='New Grid Usage With Battery', 
            linestyle='dotted'
            )
        fig3.autofmt_xdate()
        ax3.legend()
        st.pyplot(fig3)

if __name__ == "__main__":
    main()
