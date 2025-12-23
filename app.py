import yaml
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
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

    
    with col2:
        st.subheader("Battery Parameters")
        st.session_state.config.BATTERY_INEFFICIENCY_FACTOR = st.number_input(
            "Battery Inefficiency Factor", 
            value=config.BATTERY_INEFFICIENCY_FACTOR,
            step=0.01,
            format="%.2f"
        )
        st.session_state.config.BATTERY_SIZE_KWH = st.number_input(
            "Battery Size (kWh)", 
            value=config.BATTERY_SIZE_KWH,
            step=5
        )
        st.session_state.config.BATTERY_POWER = st.number_input(
            "Battery Power (kW)", 
            value=config.BATTERY_POWER,
            step=5
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
            format="%.2f"
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
            min_value=1
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
    
    if st.button("Save Configuration"):
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

            battery_capex = (
                st.session_state.optimisation_results.battery_size  # type: ignore
                * st.session_state.config.BATTERY_COST_PER_KWH / 100
                * (1 + st.session_state.config.TAX_RATE)
            )

            # Put total cost into €/kWh with tax
            st.session_state.optimisation_results.total_cost = (
                st.session_state.optimisation_results.total_cost / 100 * (1 + st.session_state.config.TAX_RATE) # type: ignore
            )   

            # This gives the cost without CAPEX in €
            st.session_state.usage_data['c_variable_total_cost_without_battery'] = (
                st.session_state.optimisation_results.grid * 
                st.session_state.usage_data['c_variable_and_fixed_per_kwh'] * 
                (1 + st.session_state.config.TAX_RATE) / 100
            )

            # Put the cost the optimised cost with battery usage into another variable
            st.session_state.usage_data['c_variable_total_cost_with_battery'] = (
                st.session_state.usage_data['c_variable_total_cost_without_battery'] + battery_capex

            )
            logging.info(f"Optimised total cost with battery: {st.session_state.usage_data['c_variable_total_cost_with_battery'] }")

            # Project this date investment_period years into the future
            st.session_state.breakeven = calculate_projections(
                st.session_state.usage_data, 
                st.session_state.optimisation_results.battery_size, # type: ignore
                config=st.session_state.config
            )

        # When we optimise for the cost whilst also optimising the battery size, this isn't needed 
        st.session_state.optimisation_results_with_battery = calculate_cost_with_battery(
            st.session_state.optimisation_results, 
            st.session_state.config
        )

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
            optimised_total_cost = st.session_state.optimisation_results.total_cost,
            investment_duration_years = st.session_state.investment_duration_years, 
            config = st.session_state.config
        )

        st.success("Optimization complete!")

    st.write(f"Inflation adjusted cost for {st.session_state.investment_duration_years} years.")
    col_result1, col_result2, col_result3 = st.columns(3)

    with col_result1:
        st.metric(
            label="Gas Heating Cost",
            value=(f"""
                {inflation_adjusted_cost(
                    st.session_state.usage_data['scaled_kwh_usage'].sum() * config.GAS_HEATING_C_PER_KWH * config.GAS_CONVERSION_RATIO / 100,
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

    with col_result3:
        st.metric(
            label="Variable Rate Cost (No Battery)",
            value=(f"""{inflation_adjusted_cost(
                st.session_state.usage_data['c_total_variable_cost'].sum()/100, 
                st.session_state.investment_duration_years, 
                st.session_state.config.INFLATION_RATE
                ) :,.0f} €"""
            ),
            delta=None,
            help="This is the inflation adjuseted variable cost for the period without any energy stored in the battery."
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
        st.metric(
            label="Variable rate with battery vs flat cost (CAPEX & OPEX)",
            value=(
                f"{st.session_state.inflation_adjusted_costs['optimised_inflation_adjusted_with_battery_cost']:,.0f} €" 
                if st.session_state.inflation_adjusted_costs else "N/A"
                )
                ,
            delta=(f"{st.session_state.inflation_adjusted_costs['optimised_inflation_adjusted_vs_flat_cost_delta']:,.0f} €"
                   if st.session_state.inflation_adjusted_costs else "N/A"
                   ),
                help="This shows the difference between the optimized variable cost with battery and the flat rate cost over the investment duration."
        )
    
    with col_result2:
        st.metric(
            label="Variable rate with battery vs. variable rate alone (CAPEX & OPEX)",
            value=(
                    f"{st.session_state.inflation_adjusted_costs['optimised_inflation_adjusted_with_battery_cost']:,.0f} €"
                    if st.session_state.inflation_adjusted_costs else "N/A"
                ),
            delta=(
                    f"{st.session_state.inflation_adjusted_costs['optimised_inflation_adjusted_vs_variable_cost_delta']:,.0f} €"
                    if st.session_state.inflation_adjusted_costs else "N/A"
                ),
            help="This shows the difference between the optimized variable cost with battery and the variable rate cost alone over the investment duration."
        )
    
    st.markdown("---")
    
    # Breakeven graph
    st.header("Breakeven Analysis")
    flat_vs_battery_tab, variable_vs_battery_tab = st.tabs(["Flat Cost vs Battery Optimised", "Variable Cost vs Battery Optimised"])
    if st.session_state.breakeven is not None:
        battery_capex = (
            st.session_state.optimisation_results.battery_size # type: ignore
            * st.session_state.config.BATTERY_COST_PER_KWH 
            * (1 + st.session_state.config.TAX_RATE)
            )

        projections = st.session_state.breakeven.copy()
        projections["c_variable_total_cost_with_battery_cumulative"] = projections["c_variable_total_cost_with_battery_cumulative"] / 100
        projections["c_total_variable_cost_cumulative"] = projections["c_total_variable_cost_cumulative"]
        projections["c_total_flat_cost_cumulative"] = projections["c_total_flat_cost_cumulative"]/100
        projections = projections[projections['datetime'] >= '2025-01-01']

        with flat_vs_battery_tab:
            breakeven_point = projections[
                    projections.c_total_flat_cost_cumulative >= projections.c_variable_total_cost_with_battery_cumulative
                ]

            if len(breakeven_point) > 0:
                breakeven_date = breakeven_point.iloc[0]['datetime']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(projections["datetime"], projections["c_total_flat_cost_cumulative"], label='Fixed Without Battery (Cumulative)', linewidth=2)
            ax.plot(projections["datetime"], projections["c_variable_total_cost_with_battery_cumulative"], label='With Battery (Cumulative)', linewidth=2)
            
            if len(breakeven_point) > 0:
                ax.axvline(x=breakeven_date, color='red', linestyle='--', label=f'Breakeven Point ({breakeven_date.date()})')

            ax.set_xlabel('Years', fontsize=12)
            ax.set_ylabel('Cumulative Cost (€)', fontsize=12)
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{x:,.0f}€")
            )
            ax.set_title('Breakeven Analysis: Fixed Rate vs Battery Investment', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

        with variable_vs_battery_tab:
            breakeven_point = projections[
                projections.c_total_variable_cost_cumulative >= projections.c_variable_total_cost_with_battery_cumulative
            ]

            if len(breakeven_point) > 0:
                breakeven_date = breakeven_point.iloc[0]['datetime']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(projections["datetime"], projections["c_total_variable_cost_cumulative"], label='Without Battery (Cumulative)', linewidth=2)
            ax.plot(projections["datetime"], projections["c_variable_total_cost_with_battery_cumulative"], label='With Battery (Cumulative)', linewidth=2)
            
            if len(breakeven_point) > 0:
                ax.axvline(x=breakeven_date, color='red', linestyle='--', label=f'Breakeven Point ({breakeven_date.date()})')

            ax.set_xlabel('Years', fontsize=12)
            ax.set_ylabel('Cumulative Cost (€)', fontsize=12)
            ax.set_title('Breakeven Analysis: Variable Rate vs Battery Investment', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

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
    
    # Graph 3
    if st.session_state.profile_with_battery is not None:
        st.subheader("3. Energy Profile with Battery Optimization")
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
