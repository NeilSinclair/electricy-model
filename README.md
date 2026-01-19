# Electricity Cost Model

A Streamlit-based application for modeling and optimizing electricity costs with thermal energy storage (TES) and heat pump systems. This tool helps analyze the economic viability of investing in TES and heat pump systems by comparing different pricing scenarios and optimizing system sizing.

## Overview

This application enables users to:
- Model electricity costs under different pricing structures (flat rate vs. variable rate)
- Optimize thermal energy storage (TES) and heat pump sizing to minimize long-term costs
- Analyze breakeven points for TES and heat pump investments
- Visualize electricity price patterns and consumption profiles
- Project costs over multiple years with inflation adjustments

## Features

### 1. Cost Modeling
- **Variable vs. Fixed Rate Analysis**: Compare costs between fixed and variable electricity pricing
- **System Optimization**: Automatically optimize TES size and heat pump capacity using Mixed Integer Linear Programming (MILP)
- **Manual Configuration**: Option to manually set TES and heat pump sizes
- **Long-term Projections**: Calculate inflation-adjusted costs over configurable investment periods
- **Comprehensive Cost Breakdown**: Includes network usage, taxes, surcharges, and other additional costs

### 2. Optimization Engine
- Uses PuLP optimization library with CBC solver
- Optimizes for minimum total cost including:
  - Energy costs from grid electricity
  - Capital expenditure (CAPEX) for TES and heat pump
- Considers heat pump coefficient of performance (COP)
- Enforces physical constraints (charge/discharge limits, state of charge)

### 3. Visualization & Analysis
- **Breakeven Analysis**: Interactive charts showing when optimized systems pay for themselves
- **Raw Data Exploration**: 
  - Hourly electricity price patterns by month
  - Daily consumption patterns
  - Combined price and consumption analysis
  - Grid usage comparison with battery optimization
- **Inflation-Adjusted Metrics**: All cost projections account for inflation

## Installation

### Prerequisites
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

This project uses `uv` for dependency management. Dependencies are defined in `requirements.txt` and will be installed automatically.

### Using uv (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd electricity-model

# Run the application (uv will handle dependencies automatically)
uv run streamlit run app.py
```

### Using pip
```bash
# Clone the repository
git clone <repository-url>
cd electricy-model

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Project Structure

```
electricity-model/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── README.md                   # This file
├── config/
│   └── config.yaml            # Configuration parameters
├── data/
│   ├── day_ahead_1yr.csv      # Day-ahead electricity prices
│   ├── heat_data.csv          # Hourly heat demand data
├── notebooks/
│   └── optimizer.ipynb        # Development notebook
└── src/
    ├── data_processing.py     # Data loading and processing
    ├── pulp_optimiser.py      # MILP optimization solver
    ├── cost_calculations.py   # Cost projection calculations
    ├── breakeven_plots.py     # Breakeven visualization
    └── raw_data_plots.py      # Raw data visualization
```

## Configuration

The application is configured via `config/config.yaml`. Key parameters include:

### Usage & Rates
- `SCALE_FACTOR`: Multiplier for energy usage (default: 1.0)
- `FLAT_RATE_C_PER_KWH`: Flat electricity rate in cents per kWh
- `GAS_HEATING_C_PER_KWH`: Gas heating cost for comparison

### Heat Pump Parameters
- `HEAT_PUMP_COP`: Coefficient of performance (default: 3.5)
- `HEAT_PUMP_COST_PER_KW`: Capital cost per kW capacity (€)
- `HEAT_PUMP_SIZE_KW`: Heat pump capacity

### Thermal Energy Storage (TES)
- `TES_SIZE_KWH`: Storage capacity in kWh
- `TES_COST_PER_KWH`: Capital cost per kWh storage (€)
- `TES_POWER`: Maximum charge/discharge rate (kW)

### Additional Costs (c/kWh)
- `NETWORK_USAGE`: Grid network usage fees
- `ELECTRICITY_TAX`: Government electricity tax
- `ADDITIONAL_COST`: Other miscellaneous costs
- `KONZESSION`: Concession fees
- `CHP_SURCHARGE`: Combined heat and power surcharge

### Financial Parameters
- `TAX_RATE`: VAT or sales tax rate (default: 0.19)
- `ELECTRICITY_INFLATION_RATE`: Annual inflation rate (default: 0.02)
- `INVESTMENT_DURATION_YEARS`: Analysis period (default: 10)

## Usage

### Running the Application

1. **Start the app**:
   ```bash
   uv run streamlit run app.py
   ```

2. **Navigate** using the sidebar:
   - **Cost Modelling**: Main analysis interface
   - **Raw Data**: Visualize electricity price and usage patterns

### Cost Modelling Workflow

1. **Configure Parameters**:
   - Adjust scale factor to model different usage scenarios
   - Set flat and variable electricity rates
   - Configure TES and heat pump costs
   - Choose whether to fix or optimize system sizes

2. **Optimize Costs**:
   - Click "Optimize Costs" button
   - The solver will determine optimal TES and heat pump sizes
   - View optimization results and cost metrics

3. **Analyze Results**:
   - Compare inflation-adjusted costs over investment period
   - Review breakeven analysis charts
   - Examine savings between different scenarios:
     - Fixed rate (non-optimized) vs. Variable rate (optimized)
     - Variable rate (non-optimized) vs. Variable rate (optimized)

4. **Save Configuration** (optional):
   - Save current parameters to YAML for future use

### Raw Data Analysis

- Explore hourly electricity price patterns by month
- Visualize daily consumption patterns
- Combine price and consumption views
- Compare original vs. optimized grid usage profiles

## Data Requirements

The application expects three CSV files in the `data/` directory:

### 1. `heat_data.csv`
- Columns: `Timestamp`, `Heat demand kWh (heat)`
- Hourly heat energy demand data
- Timestamp format: Unix timestamp (seconds)

### 2. `day_ahead_1yr.csv`
- Columns: `Start date`, `End date`, `Germany/Luxembourg [€/MWh] Original resolutions`
- Day-ahead electricity market prices
- Date format: "MMM DD, YYYY HH:MM AM/PM"
- Data source: https://www.smard.de/en/downloadcenter/download-market-data/ 
- Data info: German day ahead electricity prices from 01.01.2025 to 01.01.2026 at 15 minute frequency downloaded as .csv, renamed as day_ahead_1yr.csv and put into the `data/` folder

## Technical Details

### Optimization Algorithm

The application uses Mixed Integer Linear Programming (MILP) via PuLP with the CBC solver to optimize TES dispatch and sizing:

To make the computation tractable, the optimisation is limited to the first year of the investment period where the CAPEX costs as scaled by the number of years in the investment period. Concretely, if we have an investment timeline of 10 years and the heat pump and TES costs are 1000€ and 50€ per kW/kWh respectively, we optimise the first year with a heat pump and TES cost of 100€ and 5€ per kW/kWh respectively. When the final numbers are shown on the dashboard, we show the optimised grid usage times the variable electricity price - plus year-on-year inflation - and add the re-scaled heat pump and TES costs for the full CAPEX.

**Decision Variables:**
- Grid electricity draw at each hour
- TES charge/discharge at each hour
- State of charge (SOC) at each hour
- TES capacity (when optimizing size)
- Heat pump capacity (when optimizing size)

**Objective:**
Minimize total cost = energy costs + annualized CAPEX

**Constraints:**
- Energy balance: meet heat demand at every hour
- Charge/discharge cannot happen simultaneously (binary constraint)
- SOC within bounds [0, TES capacity]
- Power limits based on heat pump capacity
- Cyclic constraint: end SOC equals start SOC
- Heat pump COP conversion from electrical to thermal energy

### Cost Calculations

**Inflation-Adjusted Costs:**
```
PV = C × [(1 + r)^n - 1] / r
```
Where:
- C = annual cost
- r = inflation rate
- n = number of years

**Total System Cost:**
```
Total = Energy Cost + TES CAPEX + Heat Pump CAPEX
```

## Docker Support

Build and run using Docker:

```bash
# Build the image
docker build -t electricity-model .

# Run the container
docker run -p 8501:8501 electricity-model
```

Access the application at `http://localhost:8501`

## Dependencies

Key Python packages:
- `streamlit`: Web application framework
- `pandas`: Data manipulation
- `pulp`: Linear programming solver
- `plotly`: Interactive visualizations
- `matplotlib`: Static plots
- `pyyaml`: Configuration file parsing

See `requirements.txt` for complete list with versions.

