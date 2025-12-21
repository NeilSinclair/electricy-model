import pulp as pl
from src.data_processing import ElectricityConfig
import pandas as pd
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class BatteryDispatchResult:
    grid: list[float]
    charge: list[float]
    discharge: list[float]
    soc: list[float]
    mode: list[str]
    price: list[float]
    battery_size: float | None = None
    total_cost: float | None = None


def solve_battery_dispatch_pulp(
    price: pd.Series,
    demand: pd.Series,
    years: int = 1,
    config: ElectricityConfig | None = None,
) -> BatteryDispatchResult:
    """Fuction which solves an enery storage optimisation problem using MILP with PuLP.
    
    Args:
        price (obj:`pd.Series`): Series of electricity prices (€/kWh).
        demand (obj:`pd.Series`): Series of electricity demand (kWh).
        config (ElectricityConfig | None): Configuration object, if None loads from 'config.yaml
    
    Returns:
        BatteryDispatchResult: Object containing optimisation results.    
    """
    
    if config is None:
        config = ElectricityConfig.from_yaml("config/config.yaml")

    
    P_MAX = config.BATTERY_POWER
    DT=1.0
    ETA_C=config.BATTERY_INEFFICIENCY_FACTOR
    ETA_D=config.BATTERY_INEFFICIENCY_FACTOR

    # The battery cost needs to be in the same units (c) as the price series
    battery_cost_per_kwh_c = config.BATTERY_COST_PER_KWH * 100 / years

    # If SOC_FACTOR is 0.8, batthery charges from 10% to 90% of E_MAX
    alpha = (1-config.SOC_FACTOR)/2
    beta = 1 - alpha

    T = len(price)
    C_MAX = P_MAX * DT

    model = pl.LpProblem("Battery_Arbitrage", pl.LpMinimize)

    # ---- decision variables ----
    g = pl.LpVariable.dicts("grid", range(T), lowBound=0)
    c = pl.LpVariable.dicts("charge", range(T), lowBound=0)
    u = pl.LpVariable.dicts("discharge", range(T), lowBound=0)
    s = pl.LpVariable.dicts("soc", range(T), lowBound=0)
    b = pl.LpVariable("battery_size", lowBound=0, upBound=300, cat=pl.LpContinuous)

    # binary: 1 = charging allowed, 0 = discharging allowed
    y = pl.LpVariable.dicts("is_charging", range(T), cat="Binary")

    # ---- objective: minimise grid cost ----
    logging.info(f"Battery cost per kWh: {config.BATTERY_COST_PER_KWH}")
    model += (
        pl.lpSum(price[t] * g[t] for t in range(T))
        + battery_cost_per_kwh_c * b
    )

    for t in range(T):

        # --- meet demand (no export) ---
        model += g[t] + u[t] == demand[t] + c[t]

        # --- power limits ---
        model += c[t] <= C_MAX * y[t]
        model += u[t] <= C_MAX * (1 - y[t])

        # --- SOC limits ---
        model += s[t] >= alpha * b 
        model += s[t] <= beta * b

        # --- SOC dynamics ---
        if t == 0:
            model += s[t] == alpha * b + ETA_C * c[t] - (1 / ETA_D) * u[t]
        else:
            model += s[t] == s[t-1] + ETA_C * c[t] - (1 / ETA_D) * u[t]

    # ---- optional: end where you started (prevents horizon dumping) ----
    model += s[T-1] == s[0]

    # ---- solve ----
    # solver = pl.HiGHS_CMD(msg=False)
    model.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus[model.status]
    print("Solver status:", status)

    print("Status:", pl.LpStatus[model.status])
    print("b =", pl.value(b))
    capex = battery_cost_per_kwh_c * pl.value(b) 

    energy = sum(float(price.iloc[t]) * pl.value(g[t]) for t in range(T))
    obj = pl.value(model.objective)

    print("Energy term:", energy)
    print("Capex term:", capex)
    print("Objective:", obj)
    print("Energy + Capex:", energy + capex)

    # ---- extract solution ----
    result = BatteryDispatchResult(
        grid = [pl.value(g[t]) for t in range(T)],
        charge = [pl.value(c[t]) for t in range(T)],
        discharge = [pl.value(u[t]) for t in range(T)],
        soc = [pl.value(s[t]) for t in range(T)],
        mode = None, # ["charge" if pl.value(y[t]) > 0.5 else "discharge" for t in range(T)],
        price = price.tolist(),
        battery_size = pl.value(b),
        total_cost = pl.value(model.objective),
    )

    return result