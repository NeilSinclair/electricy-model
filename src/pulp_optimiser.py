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
    price: list[float]
    tes_size: float | None = None
    heat_pump_size: float | None = None
    total_cost: float | None = None


def solve_tes_dispatch_pulp(
    price: pd.Series,
    demand: pd.Series,
    years: int = 1,
    config: ElectricityConfig | None = None,
) -> BatteryDispatchResult:
    """Fuction which solves a temperature energy storage optimisation problem using MILP with PuLP.
    
    Args:
        price (obj:`pd.Series`): Series of electricity prices (€/kWh).
        demand (obj:`pd.Series`): Series of heat energy demand (kWh).
        years (int): Number of years over which to annualise the TES cost.
        config (ElectricityConfig | None): Configuration object, if None loads from 'config.yaml.
    
    Returns:
        BatteryDispatchResult: Object containing optimisation results.    
    """
    
    if config is None:
        config = ElectricityConfig.from_yaml("config/config.yaml")

    # The battery cost needs to be in the same units (c) as the price series
    TES_COST_PER_KWH_c = config.TES_COST_PER_KWH * 100 / years
    HEAT_PUMP_COST_PER_KW_c = config.HEAT_PUMP_COST_PER_KW * 100 / years

    T = len(price)

    model = pl.LpProblem("TES_Arbitrage", pl.LpMinimize)

    # ---- decision variables ----
    grid = pl.LpVariable.dicts("grid", range(T), lowBound=0)
    charge = pl.LpVariable.dicts("charge", range(T), lowBound=0)
    discharge = pl.LpVariable.dicts("discharge", range(T), lowBound=0)
    soc = pl.LpVariable.dicts("soc", range(T), lowBound=0)
    tes_size = pl.LpVariable("tes_size", lowBound=0, upBound=300, cat=pl.LpContinuous)
    heat_pump_size = pl.LpVariable("heat_pump_size", lowBound=0, upBound=300, cat=pl.LpContinuous)

    # binary: 1 = charging allowed, 0 = discharging allowed
    y = pl.LpVariable.dicts("is_charging", range(T), cat="Binary")

    # ---- objective: minimise grid cost ----
    logging.info(f"TES cost per kWh: {config.TES_COST_PER_KWH}")
    model += (
        pl.lpSum(price[t] * grid[t] for t in range(T))
        + TES_COST_PER_KWH_c * tes_size
        + HEAT_PUMP_COST_PER_KW_c * heat_pump_size
    )

    for t in range(T):

        # --- meet demand (no export) ---
        # We scale the grid amount by the heat pump COP to get the heat output
        model += (grid[t] * config.HEAT_PUMP_COP) + discharge[t] == demand[t] + charge[t]

        # --- power limits: we assume the TES can only charge and discharge at the same capacity as the heat pump ---
        M = 500
        model += charge[t] <= heat_pump_size 
        model += charge[t] <= M * y[t]
        model += discharge[t] <= heat_pump_size 
        model += discharge[t] <= M * (1 - y[t])

        # --- SOC limits ---
        model += soc[t] >= 0
        model += soc[t] <= tes_size

        # The system can't draw more energy from the grid than it needs to meet the heat demand
        model += grid[t] * config.HEAT_PUMP_COP <= heat_pump_size

        # --- SOC dynamics: We assume heat pump charges the TES directly and there's no conversion loss ---
        if t == 0:
            model += soc[t] == 0.5 * tes_size + charge[t] - discharge[t]
        else:
            model += soc[t] == soc[t-1] + charge[t] - discharge[t]

    # ---- optional: end where you started (prevents horizon dumping) ----
    model += soc[T-1] == soc[0]

    # ---- solve ----
    # solver = pl.HiGHS_CMD(msg=False)
    model.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus[model.status]
    print("Solver status:", status)

    print("TES Size =", pl.value(tes_size))
    print(f"TES Capex: {round(pl.value(tes_size) * config.TES_COST_PER_KWH * (1 + config.TAX_RATE)):,.0f} €") # type: ignore
    print(f"Heat Pump Size {pl.value(heat_pump_size):.1f} kW")
    print(f"Heat Pump Capex: {round(pl.value(heat_pump_size) * config.HEAT_PUMP_COST_PER_KW * (1 + config.TAX_RATE)):,.0f} €") # type: ignore
    capex = (config.TES_COST_PER_KWH * pl.value(tes_size) + config.HEAT_PUMP_COST_PER_KW  * pl.value(heat_pump_size)) # type: ignore

    energy = sum(float(price.iloc[t]) * pl.value(grid[t]) for t in range(T))

    print(f"Energy term with tax for {years} years: {energy * years * (1 + config.TAX_RATE) / 100:,.0f} €")
    print(f"Capex term with tax over {years} years: {capex * (1 + config.TAX_RATE):,.0f} €")
    print(f"Energy + Capex with tax: {((energy * years / 100) + capex) * (1 + config.TAX_RATE):,.0f} €")
    print(f"Objective term raw: {pl.value(model.objective)/100:,.0f} €")
    print(f"Objective term raw over {years} years with tax: {pl.value(model.objective) * (1 + config.TAX_RATE) / 100:,.0f} €")

    # ---- extract solution ----
    result = BatteryDispatchResult(
        grid = [pl.value(grid[t]) for t in range(T)],
        charge = [pl.value(charge[t]) for t in range(T)],
        discharge = [pl.value(discharge[t]) for t in range(T)],
        soc = [pl.value(soc[t]) for t in range(T)],
        heat_pump_size = pl.value(heat_pump_size), # type: ignore
        price = price.tolist(),
        tes_size = pl.value(tes_size), # type: ignore
        total_cost = pl.value(model.objective),
    )

    return result



def solve_tes_dispatch_pulp_fixed_battery(
    price: pd.Series,
    demand: pd.Series,
    years: int = 1,
    config: ElectricityConfig | None = None,
) -> BatteryDispatchResult:
    """Fuction which solves a temperature energy storage optimisation problem using MILP with PuLP.
    
    Args:
        price (obj:`pd.Series`): Series of electricity prices (€/kWh).
        demand (obj:`pd.Series`): Series of heat energy demand (kWh).
        years (int): Number of years over which to annualise the TES cost.
        config (ElectricityConfig | None): Configuration object, if None loads from 'config.yaml.
    
    Returns:
        BatteryDispatchResult: Object containing optimisation results.    
    """
    
    if config is None:
        config = ElectricityConfig.from_yaml("config/config.yaml")

    # The battery cost needs to be in the same units (c) as the price series
    TES_COST_PER_KWH_c = config.TES_COST_PER_KWH * 100 / years
    HEAT_PUMP_COST_PER_KW_c = config.HEAT_PUMP_COST_PER_KW * 100 / years

    T = len(price)

    model = pl.LpProblem("TES_Arbitrage", pl.LpMinimize)

    # ---- decision variables ----
    grid = pl.LpVariable.dicts("grid", range(T), lowBound=0)
    charge = pl.LpVariable.dicts("charge", range(T), lowBound=0)
    discharge = pl.LpVariable.dicts("discharge", range(T), lowBound=0)
    soc = pl.LpVariable.dicts("soc", range(T), lowBound=0)

    # binary: 1 = charging allowed, 0 = discharging allowed
    y = pl.LpVariable.dicts("is_charging", range(T), cat="Binary")

    # ---- objective: minimise grid cost ----
    logging.info(f"TES cost per kWh: {config.TES_COST_PER_KWH}")
    model += (
        pl.lpSum(price[t] * grid[t] for t in range(T))
    )

    for t in range(T):

        # --- meet demand (no export) ---
        # We scale the grid amount by the heat pump COP to get the heat output
        model += (grid[t] * config.HEAT_PUMP_COP) + discharge[t] == demand[t] + charge[t]

        # --- power limits: we assume the TES can only charge and discharge at the same capacity as the heat pump ---
        M = 500
        model += charge[t] <= config.HEAT_PUMP_SIZE_KW
        model += charge[t] <= M * y[t]
        model += discharge[t] <= config.TES_POWER
        model += discharge[t] <= M * (1 - y[t])

        # --- SOC limits ---
        model += soc[t] >= 0
        model += soc[t] <= config.TES_SIZE_KWH

        # The system can't draw more energy from the grid than it needs to meet the heat demand
        model += grid[t] * config.HEAT_PUMP_COP <= config.HEAT_PUMP_SIZE_KW

        # --- SOC dynamics: We assume heat pump charges the TES directly and there's no conversion loss ---
        if t == 0:
            model += soc[t] == 0.5 * config.TES_SIZE_KWH + charge[t] - discharge[t]
        else:
            model += soc[t] == soc[t-1] + charge[t] - discharge[t]

    # ---- optional: end where you started (prevents horizon dumping) ----
    model += soc[T-1] == soc[0]

    # ---- solve ----
    # solver = pl.HiGHS_CMD(msg=False)
    model.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus[model.status]
    print("Solver status:", status)

    print("TES Size =", config.TES_SIZE_KWH)
    print(f"TES Capex: {round(config.TES_SIZE_KWH * config.TES_COST_PER_KWH * (1 + config.TAX_RATE)):,.0f} €") # type: ignore
    print(f"Heat Pump Size {config.HEAT_PUMP_SIZE_KW:.1f} kW")
    print(f"Heat Pump Capex: {round(config.HEAT_PUMP_SIZE_KW * config.HEAT_PUMP_COST_PER_KW * (1 + config.TAX_RATE)):,.0f} €") # type: ignore
    capex = (config.TES_COST_PER_KWH * config.TES_SIZE_KWH + config.HEAT_PUMP_COST_PER_KW  * config.HEAT_PUMP_SIZE_KW) # type: ignore
    energy = sum(float(price.iloc[t]) * pl.value(grid[t]) for t in range(T))

    print(f"Energy term with tax for {years} years: {energy * years * (1 + config.TAX_RATE) / 100:,.0f} €")
    print(f"Capex term with tax over {years} years: {capex * (1 + config.TAX_RATE):,.0f} €")
    print(f"Energy + Capex with tax: {((energy * years / 100) + capex) * (1 + config.TAX_RATE):,.0f} €")
    print(f"Objective term raw: {pl.value(model.objective)/100:,.0f} €")
    print(f"Objective term raw over {years} years with tax: {pl.value(model.objective) * (1 + config.TAX_RATE) / 100:,.0f} €")

    # ---- extract solution ----
    result = BatteryDispatchResult(
        grid = [pl.value(grid[t]) for t in range(T)],
        charge = [pl.value(charge[t]) for t in range(T)],
        discharge = [pl.value(discharge[t]) for t in range(T)],
        soc = [pl.value(soc[t]) for t in range(T)],
        heat_pump_size = config.HEAT_PUMP_SIZE_KW,
        price = price.tolist(),
        tes_size = config.TES_SIZE_KWH,
        total_cost = pl.value(model.objective),
    )

    return result