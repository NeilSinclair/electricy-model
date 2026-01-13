import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

def plot_raw_data(
        hourly_monthly_avg_cost: pd.DataFrame, 
        hourly_monthly_avg_usage: pd.DataFrame, 
        profile_with_battery: pd.DataFrame | None = None,
        ) -> None:
    """Plot raw data visualizations for electricity cost and usage.

    Args:
        hourly_monthly_avg_cost (pd.DataFrame): DataFrame containing hourly average cost data.
        hourly_monthly_avg_usage (pd.DataFrame): DataFrame containing hourly average usage data.
    """

    # Get unique months in chronological order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
    months_in_data = [m for m in month_order if m in hourly_monthly_avg_cost['month_name'].unique()]

    st.header("Raw Data Visualizations")
    st.write("This section presents various visualizations of the raw electricity cost and usage data to help understand patterns and trends.")

    # Graph 1 - Hourly Electricity Price
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
        ax2.plot(month_data['hour_of_day'], month_data['scaled_kwh_usage'], 
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
                y=df_usage["scaled_kwh_usage"],
                name=f"{month} - Consumption",
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
    if profile_with_battery is not None:
        st.subheader("4. Energy Profile with Battery Optimization")
        st.write("This graph shows the updated energy profile with battery optimization applied, comparing original and new grid usage.")

        start = st.date_input("Start date: yyyy/mm/dd", datetime(2025, 2, 1))
        end = st.date_input("End date: yyyy/mm/dd", datetime(2025, 2, 7))

         # Graph 3 - Grid Usage vs Raw Usage
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        temp_df = (
            profile_with_battery[
                (profile_with_battery['datetime'].dt.date >= start) & # type: ignore
                (profile_with_battery['datetime'].dt.date <= end)] # type: ignore
        )
        # ax3.plot(temp_df['datetime'], temp_df['grid_kwh_usage'], label='Grid Usage')
        ax3.plot(
            temp_df['datetime'], 
            temp_df['scaled_kwh_usage'] / st.session_state.config.HEAT_PUMP_COP, 
            label='Original Grid Usage'
            )
        ax3.plot(
            temp_df['datetime'], 
            temp_df['grid_kwh_usage'], 
            label='New Grid Usage With Battery', 
            linestyle='dotted'
            )
        ax3_right = ax3.twinx()
        ax3_right.plot(
            temp_df['datetime'], 
            temp_df['price_c_per_kwh'], 
            label='Price (c/kWh)', 
            linestyle='dashed',
            color='red', 
            alpha=0.5
        )
        ax3_right.set_ylabel('Price (c/kWh)', color='red')
        ax3_right.tick_params(axis='y', labelcolor='red')
        ax3_right.legend(loc='upper right')
        fig3.autofmt_xdate()
        ax3.legend()
        st.pyplot(fig3)