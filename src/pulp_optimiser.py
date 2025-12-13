import pulp as pl
from src.data_processing import ElectricityConfig
import pandas as pd
from dataclasses import dataclass

@dataclass
class BatteryDispatchResult:
    grid: list[float]
    charge: list[float]
    discharge: list[float]
    soc: list[float]
    mode: list[str]
    price: list[float]
    total_cost: float | None


def solve_battery_dispatch_pulp(
    price: pd.Series,
    demand: pd.Series,
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
        config = ElectricityConfig.from_yaml("config.yaml")

    E_MAX = config.BATTERY_SIZE_KWH
    P_MAX = config.BATTERY_POWER
    DT=1.0
    ETA_C=config.BATTERY_INEFFICIENCY_FACTOR
    ETA_D=config.BATTERY_INEFFICIENCY_FACTOR
    soc_init=0.0
    SOC_MIN = (1-config.SOC_FACTOR) * E_MAX
    SOC_MAX = config.SOC_FACTOR * E_MAX
    T = len(price)
    C_MAX = P_MAX * DT

    model = pl.LpProblem("Battery_Arbitrage", pl.LpMinimize)

    # ---- decision variables ----
    g = pl.LpVariable.dicts("grid", range(T), lowBound=0)
    c = pl.LpVariable.dicts("charge", range(T), lowBound=0)
    u = pl.LpVariable.dicts("discharge", range(T), lowBound=0)
    s = pl.LpVariable.dicts("soc", range(T), lowBound=SOC_MIN, upBound=SOC_MAX)

    # binary: 1 = charging allowed, 0 = discharging allowed
    y = pl.LpVariable.dicts("is_charging", range(T), cat="Binary")

    # ---- objective: minimise grid cost ----
    model += pl.lpSum(price[t] * g[t] for t in range(T))

    for t in range(T):

        # --- meet demand (no export) ---
        model += g[t] + u[t] == demand[t] + c[t]

        # --- power limits ---
        model += c[t] <= C_MAX * y[t]
        model += u[t] <= C_MAX * (1 - y[t])

        # --- SOC dynamics ---
        if t == 0:
            model += s[t] == soc_init + ETA_C * c[t] - (1 / ETA_D) * u[t]
        else:
            model += s[t] == s[t-1] + ETA_C * c[t] - (1 / ETA_D) * u[t]

    # ---- optional: end where you started (prevents horizon dumping) ----
    model += s[T-1] == soc_init

    # ---- solve ----
    model.solve(pl.PULP_CBC_CMD(msg=False))

    # ---- extract solution ----
    result = BatteryDispatchResult(
        grid = [pl.value(g[t]) for t in range(T)],
        charge = [pl.value(c[t]) for t in range(T)],
        discharge = [pl.value(u[t]) for t in range(T)],
        soc = [pl.value(s[t]) for t in range(T)],
        mode = ["charge" if pl.value(y[t]) > 0.5 else "discharge" for t in range(T)],
        price = price.tolist(),
        total_cost = pl.value(model.objective),
    )

    return result