import yaml
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter

import logging

from src.data_processing import get_usage_data, get_max_heat_demand, ElectricityConfig
from src.pulp_optimiser import solve_tes_dispatch_pulp, solve_tes_dispatch_pulp_fixed_battery, BatteryDispatchResult
from src.cost_calculations import calculate_projections, inflation_adjusted_cost, calculate_inflation_adjusted_costs
from src.breakeven_plots import fixed_vs_variable_breakeven, variable_vs_variable_optimised_breakeven
from src.raw_data_plots import plot_raw_data

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

def reload_data():
    st.session_state.config.SCALE_FACTOR = st.session_state.scale_factor
    logging.info(f"Reload triggered, scale factor: {st.session_state.scale_factor}")
    _, st.session_state.usage_data = get_usage_data(config=st.session_state.config)
    st.session_state.usage_data = st.session_state.usage_data[st.session_state.usage_data.datetime < '2025-12-01']
    st.session_state.usage_data['c_variable_total_cost_with_battery'] = 0.0

def manual_capex_update():
    st.session_state.config.HEAT_PUMP_COST_PER_KW = st.session_state.hp_cost_per_kw

    st.session_state.non_optimised_capex = (
        st.session_state.config.TES_SIZE_KWH * st.session_state.config.TES_COST_PER_KWH 
        + st.session_state.config.HEAT_PUMP_SIZE_KW * st.session_state.config.HEAT_PUMP_COST_PER_KW
    )


def main():
    st.set_page_config(page_title="Electricity Cost Model", layout="wide")
    if 'usage_data' not in st.session_state:
        _, st.session_state.usage_data = get_usage_data()
        st.session_state.usage_data = st.session_state.usage_data[st.session_state.usage_data.datetime < '2025-12-01']
        st.session_state.usage_data['c_variable_total_cost_with_battery'] = 0.0
    
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
        st.session_state.config.HEAT_PUMP_SIZE_KW = get_max_heat_demand(st.session_state.usage_data)
        st.session_state.config.TES_SIZE_KWH = 0.0
        st.session_state.config.TES_POWER = 0.0

    
    if 'original_cost' not in st.session_state:
        st.session_state.original_cost = (
            (st.session_state.usage_data['c_variable_and_fixed_per_kwh'] * 
            st.session_state.usage_data['scaled_kwh_usage']).sum()  / 100
        )
    
    if 'optimisation_results' not in st.session_state:
        st.session_state.optimisation_results = BatteryDispatchResult(
            grid=[],
            charge=[],
            discharge=[],
            soc=[],
            tes_size=None,
            heat_pump_size=None,
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

    if 'fixed_battery_size' not in st.session_state:
        st.session_state.fixed_battery_size = False

    if 'fixed_heat_pump_size' not in st.session_state:
        st.session_state.fixed_heat_pump_size = False

    if 'non_optimised_capex' not in st.session_state:
        st.session_state.non_optimised_capex = (
            st.session_state.config.TES_SIZE_KWH * st.session_state.config.TES_COST_PER_KWH 
            + st.session_state.config.HEAT_PUMP_SIZE_KW * st.session_state.config.HEAT_PUMP_COST_PER_KW
        )
    
    if "hp_cost_per_kw" not in st.session_state:
        st.session_state.hp_cost_per_kw = st.session_state.config.HEAT_PUMP_COST_PER_KW
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select Option:", ["Cost Modelling", "Raw Data"])
    
    if option == "Cost Modelling":
        show_cost_modelling(st.session_state.config)
    elif option == "Raw Data":
        show_raw_data()

def calculate_cost_with_battery(optimisation_results: BatteryDispatchResult, config: ElectricityConfig) -> float:
    if optimisation_results.total_cost is None:
        return 0.0
    battery_capex = optimisation_results.tes_size * config.TES_COST_PER_KWH # type: ignore
    battery_opex = battery_capex * config.OPEX_PERCENT_OF_CAPEX
    total_battery_cost = (battery_capex + battery_opex) 
    return optimisation_results.total_cost + total_battery_cost

def show_cost_modelling(config):
    st.title("Cost Modelling")
    
    st.header("Configuration Parameters")
    st.session_state.fixed_heat_pump_size = st.toggle("Fix Heat Pump Size", value=False, help="Sets a fixed heat pump size rather than optimising it. Will set it to the maximum heat demand.")
    st.session_state.fixed_battery_size = st.toggle("Fix TES Size", value=False, help="Enables you to set the TES size manually rather than optimising it.")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Usage & Rates")
        st.number_input(
            "Scale Factor", 
            value=config.SCALE_FACTOR,
            step=0.1,
            format="%.2f",
            on_change=reload_data,
            key="scale_factor",
            help="Factor to scale the entire energy usage profile by. A value of 1.0 means no scaling, 2.0 means double the usage, etc."
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
        st.subheader("Heat Pump Parameters")
        st.session_state.config.BATTERY_INEFFICIENCY_FACTOR = st.number_input(
            "TES Inefficiency Factor", 
            value=config.BATTERY_INEFFICIENCY_FACTOR,
            step=0.01,
            format="%.2f",
            help=(
                "Value between 0 and 1 representing the efficiency of charging/discharging the TES. "
                "A value of 0.9 would mean 90% efficiency for charging and for discharging, leaing to a "
                "90% * 90% = 81% round-trip efficiency."
            )
        )
        st.session_state.config.HEAT_PUMP_SIZE_KW = st.number_input(
            "Heat Pump Size kW", 
            value=config.HEAT_PUMP_SIZE_KW,
            step=10.0,
            help="The max size of the heat pump in kW. The default value is the maximum heat demand.",
            disabled= not st.session_state.fixed_heat_pump_size,
        )
        st.session_state.config.TES_SIZE_KWH = st.number_input(
            "TES Size (kWh)", 
            value=config.TES_SIZE_KWH,
            step=5.0,
            disabled= not st.session_state.fixed_battery_size,
        )
        st.session_state.config.TES_POWER = st.number_input(
            "TES Power (kW)", 
            value=config.TES_POWER,
            step=5.0,
            help="Maximum power the TES can charge or discharge within an hour. This is set to half the maximum heat demand by default.",
            disabled = not st.session_state.fixed_battery_size,
        )
   
    
    with col3:
        st.subheader("Investment Parameters")
        st.session_state.config.HEAT_PUMP_COST_PER_KW = st.number_input(
            "Heat Pump Cost per kW (€/kW)", 
            step=50,
            on_change=manual_capex_update,
            key="hp_cost_per_kw",
        )
        st.session_state.config.TES_COST_PER_KWH = st.number_input(
            "TES Cost per kWh (€/kWh)", 
            value=config.TES_COST_PER_KWH,
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
            format="%.2f",
            disabled=True,
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
                "TES_COST_PER_KWH": st.session_state.config.TES_COST_PER_KWH,
                "OPEX_PERCENT_OF_CAPEX": st.session_state.config.OPEX_PERCENT_OF_CAPEX,
                "TES_SIZE_KWH": st.session_state.config.TES_SIZE_KWH,
                "TES_POWER": st.session_state.config.TES_POWER,
                "INFLATION_RATE": st.session_state.config.INFLATION_RATE
            }
            yaml.dump(yaml_data, f)
        st.success(f"Configuration saved to config/config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    
    st.markdown("---")
    
    # Cost calculations (using dummy values for now)
    st.header("Cost Analysis")

    if st.button("Optimize Costs"):
        with st.spinner("Running optimization. This will take a moment..."):

            if not st.session_state.fixed_battery_size:
                st.session_state.optimisation_results : BatteryDispatchResult = solve_tes_dispatch_pulp( # type: ignore
                    price=st.session_state.usage_data['c_variable_and_fixed_per_kwh'],
                    demand=st.session_state.usage_data['scaled_kwh_usage'],
                    years = st.session_state.investment_duration_years,
                    config=st.session_state.config,
                )
            else:
                st.session_state.optimisation_results : BatteryDispatchResult = solve_tes_dispatch_pulp_fixed_battery( # type: ignore
                    price=st.session_state.usage_data['c_variable_and_fixed_per_kwh'],
                    demand=st.session_state.usage_data['scaled_kwh_usage'],
                    years = st.session_state.investment_duration_years,
                    config=st.session_state.config,
                )

            # Put total cost into €/kWh with tax
            st.session_state.optimisation_results.total_cost = (
                st.session_state.optimisation_results.total_cost / 100  # type: ignore
            )   

            # This gives the cost without CAPEX in €; we already add tax in here
            st.session_state.usage_data['c_variable_total_cost_without_battery'] = (
                st.session_state.optimisation_results.grid * 
                st.session_state.usage_data['c_variable_and_fixed_per_kwh'] / 100
            )

            # Put the cost the optimised cost with battery usage into another variable
            st.session_state.usage_data['c_variable_total_cost_with_battery'] = (
                st.session_state.usage_data['c_variable_total_cost_without_battery'] # + battery_capex

            )

            # Project this date investment_period years into the future
            st.session_state.breakeven = calculate_projections(
                st.session_state.usage_data, 
                st.session_state.optimisation_results.tes_size, # type: ignore
                st.session_state.optimisation_results.heat_pump_size, # type: ignore
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
                'scaled_kwh_usage': st.session_state.usage_data['scaled_kwh_usage'],
                'grid_kwh_usage': st.session_state.optimisation_results.grid,
                'soc': st.session_state.optimisation_results.soc,
                'battery_charge_kwh': st.session_state.optimisation_results.charge,
                'battery_discharge_kwh': st.session_state.optimisation_results.discharge,   
                'price_c_per_kwh': st.session_state.optimisation_results.price,
            }
        ))

        st.session_state.inflation_adjusted_costs = calculate_inflation_adjusted_costs(
            usage_data = st.session_state.usage_data, 
            optimisation_results = st.session_state.optimisation_results,
            investment_duration_years = st.session_state.investment_duration_years, 
            config = st.session_state.config
        )

        st.success("Optimization complete!")

    st.write(f"Inflation adjusted cost for **{st.session_state.investment_duration_years} years**. These figures do not include tax.")
    col_result1, col_result2 = st.columns(2)

    with col_result1:
        st.metric(
            label="Gas Heating Cost",
            value=(f"""
                {inflation_adjusted_cost(
                    st.session_state.usage_data['scaled_kwh_usage'].sum() * st.session_state.config.GAS_HEATING_C_PER_KWH * st.session_state.config.GAS_CONVERSION_RATIO / 100,
                    st.session_state.investment_duration_years,
                    st.session_state.config.INFLATION_RATE
                ):,.0f} €"""
            ),
            delta=None,
            help="Usage of gas heating system in kWh over the period."
        )

    with col_result2:
        base_cost = inflation_adjusted_cost(
            st.session_state.usage_data["c_total_flat_cost"].sum() / 100,
            st.session_state.investment_duration_years,
            st.session_state.config.INFLATION_RATE,
        )

        capex = st.session_state.non_optimised_capex

        st.metric(
            label="Flat Rate Cost with Heat Pump (Non Optimised)",
            value=f"{(base_cost + capex):,.0f} €",
            help="This is the inflation adjusted electricity cost for the period based on a flat rate without any optimisation.",
        )


    with col_result1:
        base_cost = inflation_adjusted_cost(
            st.session_state.usage_data["c_total_flat_cost"].sum() / 100,
            st.session_state.investment_duration_years,
            st.session_state.config.INFLATION_RATE,
        )

        if st.session_state.optimisation_results.heat_pump_size is None:
            st.metric(
                label="Flat Rate Cost with Heat Pump (Optimised)",
                value=f"N/A",
                help="This is the inflation adjusted electricity cost for the period based on a flat rate with the storage and heat pump optimised.",
            )
        else:
            capex = (
                st.session_state.optimisation_results.heat_pump_size
                * st.session_state.config.HEAT_PUMP_COST_PER_KW
            )
            st.metric(
                label="Flat Rate Cost with Heat Pump (Optimised)",
                value=f"{(base_cost + capex):,.0f} €",
                help="This is the inflation adjusted electricity cost for the period based on a flat rate with the storage and heat pump optimised.",
            )



    with col_result2:
        pass

    col_result1, col_result2 = st.columns(2)
    with col_result1:
        st.metric(
            label="Variable Rate Cost with Heat Pump (Non Optimised)",
            value=(f"""{inflation_adjusted_cost(
                st.session_state.usage_data['c_total_variable_cost'].sum()/100, 
                st.session_state.investment_duration_years, 
                st.session_state.config.INFLATION_RATE
                ) + st.session_state.non_optimised_capex :,.0f} €"""
            ),
            delta=None,
            help=(
                "This is the inflation adjusted variable cost for the period when a Heat Pump is purchased and the user "
                "simply switches to a variable tariff."
            )
        )

    with col_result2:
        st.metric(
            label="Variable Rate Cost with Heat Pump (Optimised)",
            value=(
                f"{st.session_state.inflation_adjusted_costs['optimised_inflation_adjusted_with_battery_cost']:,.0f} €" 
                if st.session_state.inflation_adjusted_costs else "N/A"
            ),
            delta=None,
            help="This is the inflation adjusted variable cost for the period with an optimised TES battery and heat pump."
        )

    col_result1, col_result2 = st.columns(2)

    with col_result1:
        st.metric(
            label="Heat Pump size (kW)",
            value=(
                f"{st.session_state.optimisation_results.heat_pump_size:,.0f} kW" 
                if st.session_state.optimisation_results.heat_pump_size is not None else 
                f"{st.session_state.config.HEAT_PUMP_SIZE_KW:,.0f} kW"
            ),
            delta=None,
            help="This is the Heat Pump size based on the current configuration. This is either optimised or fixed depending on the settings above."
        )
    
    with col_result2:
        st.metric(
            label="Heat Pump Capex",
            value=(
                f"{st.session_state.optimisation_results.heat_pump_size * st.session_state.config.HEAT_PUMP_COST_PER_KW:,.0f} €" 
                if st.session_state.optimisation_results.heat_pump_size is not None else 
                f"{st.session_state.config.HEAT_PUMP_SIZE_KW * st.session_state.config.HEAT_PUMP_COST_PER_KW:,.0f} €"
            ),
            delta=None,
            help="This is the capital expenditure (Capex) for the Heat Pump."
        )

    col_result1, col_result2 = st.columns(2)

    with col_result1:
        st.metric(
            label="Optimised TES size (kWh)",
            value=(
                f"{st.session_state.optimisation_results.tes_size:,.0f} kWh" 
                if st.session_state.optimisation_results.tes_size is not None else "N/A"
            ),
            delta=None,
            help="This is the optimised TES size based on the current configuration. If the TES size is 0 kWh this indicates that a TES is not cost effective under the current parameters."
        )
    
    with col_result2:
        st.metric(
            label="TES Capex",
            value=(
                f"{st.session_state.optimisation_results.tes_size * st.session_state.config.TES_COST_PER_KWH:,.0f} €" 
                if st.session_state.optimisation_results.tes_size is not None else "N/A"
            ),
            delta=None,
            help="This is the capital expenditure (Capex) for the optimised TES size."
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
                help=("This shows the difference between the optimized variable cost and optimised TES and Heat Pump CAPEX included and the flat "
                      "rate cost with non-optimised Heat Pump and TES sizes over the investment duration."
                )
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
            help=(
                "This shows the difference between the optimized variable cost with TES and Heat Pump CAPEX "
                "included over the investment duration."
            )
        )
    
    st.markdown("---")
    
    # Breakeven graph
    st.header("Breakeven Analysis", help=(
        "These graphs compare the optimised costs over a time window with the the non-optimised costs. Optimised here means that "
        "the TES size and heat pump size have been optimised to give the lowest cost over the investment duration. "
        "Non-optimised means that the default values from the configuration file are used without any optimisation. "
    ))
    flat_vs_battery_tab, variable_vs_battery_tab = st.tabs(
        ["Flat Cost (Non-Optimised) vs Variable (Optimised)", "Variable (Non-Optimised) vs Variable (Optimised)"]
        )
    if st.session_state.breakeven is not None:

        projections = st.session_state.breakeven.copy()
        projections["c_variable_total_cost_with_battery_cumulative"] = projections["c_variable_total_cost_with_battery_cumulative"] /100
        projections["c_total_variable_cost_cumulative"] = projections["c_total_variable_cost_cumulative"] /100
        projections["c_total_flat_cost_cumulative"] = projections["c_total_flat_cost_cumulative"]/100
        projections = projections[projections['datetime'] >= '2025-01-01']

        with flat_vs_battery_tab:
           fixed_vs_variable_breakeven(projections)

        with variable_vs_battery_tab:
            variable_vs_variable_optimised_breakeven(projections)

def show_raw_data():
    st.title("Raw Data Analysis")
    
    st.write("Displaying raw electricity data with configured parameters.")
    
    # Graph calculations
    st.session_state.usage_data['month'] = st.session_state.usage_data['datetime'].dt.month
    st.session_state.usage_data['month_name'] = st.session_state.usage_data['datetime'].dt.strftime('%B')
    st.session_state.usage_data['hour_of_day'] = st.session_state.usage_data['datetime'].dt.hour

    # Calculate average price by hour and month
    hourly_monthly_avg_cost = st.session_state.usage_data.groupby(['hour_of_day', 'month_name'])['c_variable_and_fixed_per_kwh'].mean().reset_index()
    hourly_monthly_avg_usage = st.session_state.usage_data.groupby(['hour_of_day', 'month_name'])['scaled_kwh_usage'].mean().reset_index()

    plot_raw_data(
        hourly_monthly_avg_cost=hourly_monthly_avg_cost,
        hourly_monthly_avg_usage=hourly_monthly_avg_usage,
        profile_with_battery=st.session_state.profile_with_battery,
    )
    

if __name__ == "__main__":
    main()
