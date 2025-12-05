import streamlit as st
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Load configuration from YAML
@st.cache_data
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def main():
    st.set_page_config(page_title="Electricity Cost Model", layout="wide")
    
    # Load default configuration
    config = load_config()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select Option:", ["Cost Modelling", "Raw Data"])
    
    if option == "Cost Modelling":
        show_cost_modelling(config)
    elif option == "Raw Data":
        show_raw_data(config)

def show_cost_modelling(config):
    st.title("Cost Modelling")
    
    st.header("Configuration Parameters")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Usage & Rates")
        annual_usage = st.number_input(
            "Annual Usage (kWh)", 
            value=config['ANNUAL_USAGE'],
            step=10000
        )
        flat_rate = st.number_input(
            "Flat Rate (c/kWh)", 
            value=config['FLAT_RATE_C_PER_KWH'],
            step=1.0
        )
    
    with col2:
        st.subheader("Battery Parameters")
        battery_inefficiency = st.number_input(
            "Battery Inefficiency Factor", 
            value=config['BATTERY_INEFFICIENCY_FACTOR'],
            step=0.01,
            format="%.2f"
        )
        electricity_shift = st.number_input(
            "Electricity Shift Factor", 
            value=config['ELECTRICITY_SHIFT_FACTOR'],
            step=0.01,
            format="%.2f"
        )
    
    with col3:
        st.subheader("Battery Investment")
        battery_cost = st.number_input(
            "Battery Cost per kWh (€)", 
            value=config['BATTERY_COST_PER_KWH'],
            step=10
        )
        opex_percent_input = st.number_input(
            "OPEX (% of CAPEX)", 
            value=config['OPEX_PERCENT_OF_CAPEX'] * 100,
            step=1.0,
            format="%.1f"
        )
        if opex_percent_input is not None:
            opex_percent = opex_percent_input / 100
        else:
            opex_percent = config['OPEX_PERCENT_OF_CAPEX']
    
    # Additional costs section
    st.subheader("Additional Costs (€/MWh)")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        network_usage = st.number_input(
            "Network Usage", 
            value=config['NETWORK_USAGE'],
            step=0.1,
            format="%.2f"
        )
        tax_rate = st.number_input(
            "Tax Rate", 
            value=config['TAX_RATE'],
            step=0.01,
            format="%.2f"
        )
    
    with col5:
        electricity_tax = st.number_input(
            "Electricity Tax", 
            value=config['ELECTRICITY_TAX'],
            step=0.1,
            format="%.2f"
        )
        additional_cost = st.number_input(
            "Additional Cost", 
            value=config['ADDITIONAL_COST'],
            step=0.1,
            format="%.2f"
        )
    
    with col6:
        konzession = st.number_input(
            "Konzession", 
            value=config['KONZESSION'],
            step=0.1,
            format="%.2f"
        )
        chp_surcharge = st.number_input(
            "CHP Surcharge", 
            value=config['CHP_SURCHARGE'],
            step=0.1,
            format="%.2f"
        )
    
    st.markdown("---")
    
    # Cost calculations (using dummy values for now)
    st.header("Cost Analysis")
    
    col_result1, col_result2, col_result3 = st.columns(3)
    
    with col_result1:
        st.metric(
            label="Original Cost (without optimization)",
            value="€75,000",
            delta=None
        )
    
    with col_result2:
        st.metric(
            label="Reduced Cost (without battery)",
            value="€68,500",
            delta="-€6,500"
        )
    
    with col_result3:
        st.metric(
            label="Cost with Battery (CAPEX + OPEX)",
            value="€72,000",
            delta="-€3,000"
        )
    
    st.markdown("---")
    
    # Breakeven graph
    st.header("Breakeven Analysis")
    
    # Dummy data for breakeven graph
    years = np.arange(0, 11)
    cost_without_battery = 68500 + years * 0  # Flat line
    cost_with_battery = 72000 + years * 3500  # Initial investment + yearly savings
    cumulative_savings = years * 6500
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, cost_without_battery + years * 68500, label='Without Battery (Cumulative)', linewidth=2)
    ax.plot(years, cost_with_battery + years * 68000, label='With Battery (Cumulative)', linewidth=2)
    ax.axvline(x=5.5, color='red', linestyle='--', label='Breakeven Point (5.5 years)')
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
