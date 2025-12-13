from matplotlib import projections
import streamlit as st
import yaml
import matplotlib.pyplot as plt
import numpy as np
from src.data_processing import get_usage_data, ElectricityConfig
from src.pulp_optimiser import solve_battery_dispatch_pulp, BatteryDispatchResult
from src.cost_calculations import calculate_projections


# Load configuration from YAML
@st.cache_data
def load_config():
    return ElectricityConfig.from_yaml("config.yaml")

def main():
    st.set_page_config(page_title="Electricity Cost Model", layout="wide")
    if 'usage_data' not in st.session_state:
        _, st.session_state.usage_data = get_usage_data()
        st.session_state.usage_data = st.session_state.usage_data[st.session_state.usage_data.datetime < '2025-12-01']
    
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

    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select Option:", ["Cost Modelling", "Raw Data"])
    
    if option == "Cost Modelling":
        show_cost_modelling(st.session_state.config)
    elif option == "Raw Data":
        show_raw_data(st.session_state.config)

def calculate_cost_with_battery(optimisation_results, config: ElectricityConfig) -> float:
    battery_capex = config.BATTERY_SIZE_KWH * config.BATTERY_COST_PER_KWH
    battery_opex = battery_capex * config.OPEX_PERCENT_OF_CAPEX
    total_battery_cost = (battery_capex + battery_opex) * (1 + config.TAX_RATE)
    if optimisation_results.total_cost is None:
        return 0.0
    return optimisation_results.total_cost + total_battery_cost

def show_cost_modelling(config):
    st.title("Cost Modelling")
    
    st.header("Configuration Parameters")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Usage & Rates")
        soc_factor = st.number_input(
            "SOC factor", 
            value=config.SOC_FACTOR,
            step=0.1
        )
        flat_rate = st.number_input(
            "Flat Rate (c/kWh)", 
            value=config.FLAT_RATE_C_PER_KWH,
            step=1.0
        )
    
    with col2:
        st.subheader("Battery Parameters")
        battery_inefficiency = st.number_input(
            "Battery Inefficiency Factor", 
            value=config.BATTERY_INEFFICIENCY_FACTOR,
            step=0.01,
            format="%.2f"
        )
        inflation_rate = st.number_input(
            "Inflation Rate", 
            value=config.INFLATION_RATE,
            step=0.01,
            format="%.2f"
        )
    
    with col3:
        st.subheader("Battery Investment")
        battery_cost = st.number_input(
            "Battery Cost per kWh (€)", 
            value=config.BATTERY_COST_PER_KWH,
            step=10
        )
        opex_percent_input = st.number_input(
            "OPEX (% of CAPEX)", 
            value=config.OPEX_PERCENT_OF_CAPEX  * 100,
            step=1.0,
            format="%.1f"
        )
        if opex_percent_input is not None:
            opex_percent = opex_percent_input / 100
        else:
            opex_percent = config.OPEX_PERCENT_OF_CAPEX
    
    # Additional costs section
    st.subheader("Additional Costs (c/kWh)")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        network_usage = st.number_input(
            "Network Usage", 
            value=config.NETWORK_USAGE,
            step=0.1,
            format="%.2f"
        )
        tax_rate = st.number_input(
            "Tax Rate", 
            value=config.TAX_RATE,
            step=0.01,
            format="%.2f"
        )
    
    with col5:
        electricity_tax = st.number_input(
            "Electricity Tax", 
            value=config.ELECTRICITY_TAX,
            step=0.1,
            format="%.2f"
        )
        additional_cost = st.number_input(
            "Additional Cost", 
            value=config.ADDITIONAL_COST,
            step=0.1,
            format="%.2f"
        )
    
    with col6:
        konzession = st.number_input(
            "Konzession", 
            value=config.KONZESSION,
            step=0.1,
            format="%.2f"
        )
        chp_surcharge = st.number_input(
            "CHP Surcharge", 
            value=config.CHP_SURCHARGE,
            step=0.1,
            format="%.2f"
        )
    
    st.markdown("---")
    
    # Cost calculations (using dummy values for now)
    st.header("Cost Analysis")
    
    col_result1, col_result2, col_result3 = st.columns(3)

    if st.button("Optimize Costs"):
        with st.spinner("Running optimization..."):
            st.session_state.optimisation_results = solve_battery_dispatch_pulp(
                price=st.session_state.usage_data['c_variable_and_fixed_per_kwh'],
                demand=st.session_state.usage_data['scaled_kwh_usage'],
                config=st.session_state.config,
            )

            st.session_state.optimisation_results.total_cost = (
                st.session_state.optimisation_results.total_cost / 100 * (1 + st.session_state.config.TAX_RATE) # type: ignore
            )   

            st.session_state.usage_data['c_variable_total_cost_with_battery'] = (
                st.session_state.optimisation_results.grid * 
                st.session_state.usage_data['c_variable_and_fixed_per_kwh'] * 
                (1 + st.session_state.config.TAX_RATE) / 100
            )

        st.session_state.optimisation_results_with_battery = calculate_cost_with_battery(
            st.session_state.optimisation_results, 
            st.session_state.config
        )

        st.session_state.breakeven = calculate_projections(
            st.session_state.usage_data, 
            config=st.session_state.config
        )

        st.success("Optimization complete!")

    
    with col_result1:
        st.metric(
            label="Original Cost (without optimization)",
            value=f"€{st.session_state.original_cost:,.0f}",
            delta=None
        )
    
    with col_result2:
        st.metric(
            label="Optimised Cost (without battery)",
            value=(
                f"€{st.session_state.optimisation_results.total_cost:,.0f} €" 
                if st.session_state.optimisation_results.total_cost is not None else "N/A"
                )
                ,
            delta=(f"{st.session_state.original_cost - st.session_state.optimisation_results.total_cost:,.0f} €"
                   if st.session_state.optimisation_results.total_cost is not None else "N/A"
                   )
        )
    
    with col_result3:
        st.metric(
            label="Cost with Battery (CAPEX + OPEX)",
            value=(
                    f"{st.session_state.optimisation_results_with_battery:,.0f} €"
                    if st.session_state.optimisation_results_with_battery is not None else "N/A"
                ),
            delta=(
                    f"{st.session_state.original_cost - st.session_state.optimisation_results_with_battery:,.0f} €"
                    if st.session_state.optimisation_results_with_battery is not None else "N/A"
                )
        )
    
    st.markdown("---")
    
    # Breakeven graph
    st.header("Breakeven Analysis")
    if st.session_state.breakeven is not None:

        battery_capex = (
            st.session_state.config.BATTERY_SIZE_KWH * st.session_state.config.BATTERY_COST_PER_KWH * (1 + st.session_state.config.TAX_RATE))

        st.session_state.breakeven.loc[:, "c_variable_total_cost_with_battery_cumulative"] = (
            st.session_state.breakeven.loc[:, "c_variable_total_cost_with_battery_cumulative"]
            + battery_capex + (battery_capex * st.session_state.config.OPEX_PERCENT_OF_CAPEX)
        )

        projections = st.session_state.breakeven.copy()
        projections = projections[projections['datetime'] >= '2025-01-01']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(projections["datetime"], projections["c_total_variable_cost_cumulative"]/100, label='Without Battery (Cumulative)', linewidth=2)
        ax.plot(projections["datetime"], projections["c_variable_total_cost_with_battery_cumulative"], label='With Battery (Cumulative)', linewidth=2)
        # ax.axvline(x=5.5, color='red', linestyle='--', label='Breakeven Point (5.5 years)')
        ax.set_xlabel('Years', fontsize=12)
        ax.set_ylabel('Cumulative Cost (€)', fontsize=12)
        ax.set_title('Breakeven Analysis: Battery Investment', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

def show_raw_data(config):
    st.title("Raw Data Analysis")
    
    st.write("Displaying raw electricity data with configured parameters.")
    
    # Graph 1
    st.subheader("1. Hourly Electricity Price")
    st.write("This graph shows the hourly electricity prices over a typical day, highlighting peak and off-peak periods.")
    
    hours = np.arange(0, 24)
    prices = 20 + 15 * np.sin(hours * np.pi / 12) + np.random.normal(0, 2, 24)
    
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(hours, prices, marker='o', linewidth=2, markersize=4)
    ax1.fill_between(hours, prices, alpha=0.3)
    ax1.set_xlabel('Hour of Day', fontsize=11)
    ax1.set_ylabel('Price (c/kWh)', fontsize=11)
    ax1.set_title('Hourly Electricity Prices', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(hours)
    
    st.pyplot(fig1)
    
    st.markdown("---")
    
    # Graph 2
    st.subheader("2. Daily Consumption Pattern")
    st.write("This graph illustrates typical daily electricity consumption patterns, showing peak usage times.")
    
    consumption = 50 + 30 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 5, 24)
    consumption = np.maximum(consumption, 20)  # Ensure positive values
    
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(hours, consumption, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Hour of Day', fontsize=11)
    ax2.set_ylabel('Consumption (kWh)', fontsize=11)
    ax2.set_title('Daily Consumption Pattern', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(hours)
    
    st.pyplot(fig2)
    
    st.markdown("---")
    
    # Graph 3
    st.subheader("3. Potential Savings with Battery")
    st.write("This graph shows the potential savings by shifting electricity consumption from peak to off-peak hours using battery storage.")
    
    savings = prices * consumption * 0.1 * (1 - np.abs(np.sin(hours * np.pi / 12)))
    
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(hours, savings, marker='s', linewidth=2, markersize=4, color='green')
    ax3.fill_between(hours, savings, alpha=0.3, color='green')
    ax3.set_xlabel('Hour of Day', fontsize=11)
    ax3.set_ylabel('Potential Savings (€)', fontsize=11)
    ax3.set_title('Hourly Savings Potential with Battery Storage', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(hours)
    
    st.pyplot(fig3)

if __name__ == "__main__":
    main()
