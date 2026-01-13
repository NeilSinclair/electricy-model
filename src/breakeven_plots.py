import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def fixed_vs_variable_breakeven(projections: pd.DataFrame) -> None:
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
            name="Fixed Cost, Non-Optimised (Cumulative)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}€<extra></extra>",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=projections["datetime"],
            y=projections["c_variable_total_cost_with_battery_cumulative"],
            mode="lines",
            name="Variable Cost, Optimised (Cumulative)",
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
        title="Fixed Rate Non-Optimised vs Variable Rate Optimised",
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

def variable_vs_variable_optimised_breakeven(projections: pd.DataFrame) -> None:
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
            name="Variable Cost, Non-Optimised (Cumulative)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}€<extra></extra>",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=projections["datetime"],
            y=projections["c_variable_total_cost_with_battery_cumulative"],
            mode="lines",
            name="Variable Cost, Optimised (Cumulative)",
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
        title="Variable Rate: Non-Optimised vs Optimised with Battery Investment",
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
