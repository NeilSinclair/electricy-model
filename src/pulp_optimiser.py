import pulp as pl

def solve_battery_dispatch_pulp(
    price,
    demand,
    E_MAX,
    P_MAX,
    DT=1.0,
    ETA_C=1.0,
    ETA_D=1.0,
    soc_init=0.0,
):
    T = len(price)
    C_MAX = P_MAX * DT

    model = pl.LpProblem("Battery_Arbitrage", pl.LpMinimize)

    # ---- decision variables ----
    g = pl.LpVariable.dicts("grid", range(T), lowBound=0)
    c = pl.LpVariable.dicts("charge", range(T), lowBound=0)
    u = pl.LpVariable.dicts("discharge", range(T), lowBound=0)
    s = pl.LpVariable.dicts("soc", range(T), lowBound=0, upBound=E_MAX)

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
    result = {
        "grid": [pl.value(g[t]) for t in range(T)],
        "charge": [pl.value(c[t]) for t in range(T)],
        "discharge": [pl.value(u[t]) for t in range(T)],
        "soc": [pl.value(s[t]) for t in range(T)],
        "mode": ["charge" if pl.value(y[t]) > 0.5 else "discharge" for t in range(T)],
        "total_cost": pl.value(model.objective),
    }

    return result