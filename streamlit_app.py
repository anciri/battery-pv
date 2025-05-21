import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt
import plotly.graph_objects as go

# ---------------------- Helper Functions ----------------------

def load_timeseries(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel timeseries with 'timestamp' column."""
    try:
        if uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    if 'timestamp' not in df.columns:
        st.error("Missing 'timestamp' column.")
        st.stop()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.replace(';', ',').split(',') if x.strip()]


# Prepare and detect columns
MANDATORY_ENERGY = {'pv_gen_kwh', 'load_kwh'}
MANDATORY_POWER  = {'pv_gen_kw',  'load_kw'}

def _detect_scheme(cols: pd.Index) -> str:
    has_energy = MANDATORY_ENERGY.issubset(cols)
    has_power  = MANDATORY_POWER.issubset(cols)
    if has_energy and has_power:
        st.error("Ambiguous file: contains both energy and power columns.")
        st.stop()
    if has_energy:
        return 'energy'
    if has_power:
        return 'power'
    st.error("Missing mandatory columns: pv_gen_kwh & load_kwh, or pv_gen_kw & load_kw.")
    st.stop()


def prepare_dataframe(df: pd.DataFrame, abort_on_nan: bool) -> tuple[pd.DataFrame, float]:
    scheme = _detect_scheme(df.columns)
    dt_seconds = df['timestamp'].diff().dt.total_seconds().dropna().median()
    dt_h = dt_seconds / 3600

    if scheme == 'energy':
        df['pv_gen_kw'] = df['pv_gen_kwh'] / dt_h
        df['load_kw']   = df['load_kwh']   / dt_h
    else:
        df['pv_gen_kwh'] = df['pv_gen_kw'] * dt_h
        df['load_kwh']   = df['load_kw']   * dt_h

    critical = ['pv_gen_kwh', 'load_kwh', 'pv_gen_kw', 'load_kw']
    if df[critical].isna().any().any():
        if abort_on_nan:
            st.error("NaN in critical columns.")
            st.stop()
        df[critical] = df[critical].fillna(0)
        st.warning("Filled NaN in critical columns with 0.")

    return df, dt_h


class BatteryParams:
    def __init__(self, e_nom, p_nom, soc_min, soc_max, eta_rt, self_loss_h):
        self.e_nom = float(e_nom)
        self.p_nom = float(p_nom)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.eta_c = self.eta_d = sqrt(float(eta_rt))
        self.self_loss_h = float(self_loss_h)
        self.soc_init = self.soc_max


def simulate_bess(df: pd.DataFrame, params: BatteryParams, dt_h: float,
                   limit_export_kw: float | None, price_col: str | None):
    df_r = df.copy()
    # initialize columns
    for col in ['soc_start_kwh','soc_end_kwh','charge_kwh','discharge_int_kwh',
                'served_kwh','exportable_kwh','spilled_kwh','imported_kwh','net_kw','loss_total']:
        df_r[col] = 0.0
    if price_col:
        df_r['revenue_interval'] = 0.0

    soc = params.soc_init * params.e_nom
    # KPI accumulators
    captured=exported=spilled=imported=served=0
    loss_c=loss_d=self_loss=0
    peak_charge=peak_discharge=peak_net=-np.inf
    revenue_total = 0.0 if price_col else None

    p_lim_kwh = params.p_nom * dt_h
    export_lim_kwh = (limit_export_kw*dt_h) if limit_export_kw else None

    for i,row in df_r.iterrows():
        df_r.at[i,'soc_start_kwh'] = soc
        pv=row['pv_gen_kwh']; load=row['load_kwh']
        surplus=max(pv-load,0); deficit=max(load-pv,0)
        # charge
        space=(params.soc_max*params.e_nom-soc)/params.eta_c
        c=min(surplus,p_lim_kwh,space)
        soc+=c*params.eta_c; surplus-=c
        df_r.at[i,'charge_kwh']=c; captured+=c; loss_c+=c*(1-params.eta_c)
        peak_charge=max(peak_charge,c/dt_h)
        # export/spill
        if export_lim_kwh is not None:
            exp=min(surplus,export_lim_kwh)
            sp=surplus-exp
        else:
            exp=0; sp=surplus
        df_r.at[i,'exportable_kwh']=exp; df_r.at[i,'spilled_kwh']=sp
        exported+=exp; spilled+=sp
        # revenue
        if price_col:
            rev=(c-sp)*row[price_col]; df_r.at[i,'revenue_interval']=rev
            revenue_total+=rev
        # discharge
        avail=(soc-params.soc_min*params.e_nom)
        d_int=min(deficit/params.eta_d,p_lim_kwh,avail)
        soc-=d_int; served_kw=d_int*params.eta_d; deficit-=served_kw
        df_r.at[i,'discharge_int_kwh']=d_int; df_r.at[i,'served_kwh']=served_kw
        served+=served_kw; loss_d+=d_int-served_kw
        peak_discharge=max(peak_discharge,served_kw/dt_h)
        # import
        df_r.at[i,'imported_kwh']=deficit; imported+=deficit
        # net
        net=load-pv-served_kw+exp/dt_h; df_r.at[i,'net_kw']=net
        peak_net=max(peak_net,net)
        # soc end and self-loss
        df_r.at[i,'soc_end_kwh']=soc
        auto=soc*params.self_loss_h*dt_h; soc-=auto; self_loss+=auto
        df_r.at[i,'loss_total']=auto + (d_int-served_kw) + c*(1-params.eta_c)

    # aggregate KPIs
    total_pv=df_r['pv_gen_kwh'].sum(); total_load=df_r['load_kwh'].sum()
    cycles=(captured+(served/params.eta_d))/(2*params.e_nom)
    kpis={'E_nom_kWh':params.e_nom,'P_nom_kW':params.p_nom,
          'total_PV_Generado_kWh':total_pv,'total_load':total_load,
          'captured_kWh':captured,'descargado_kWh':served,
          'export_kWh':exported,'vertido_kWh':spilled,'import_kWh':imported,
          'loss_charge_kWh':loss_c,'loss_discharge_kWh':loss_d,'self_loss_kwh':self_loss,
          'loss_total_kWh':loss_c+loss_d+self_loss,'cycles_eq':cycles,
          'peak_charge_kW':peak_charge,'peak_discharge_k':peak_discharge,'peak_net_kW':peak_net,
          'revenue_kWh':revenue_total}
    return df_r, kpis


def kpis_to_dataframe(kpis: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(kpis, orient='index', columns=['Value'])
    df.index.name='KPI'
    return df

# ---------------------- Plotting ----------------------

def plot_interactive_energy_flows(df):
    fig = go.Figure()
    for col,name in [('pv_gen_kwh','PV Generation'),('load_kwh','Load'),
                    ('charge_kwh','Charge'),('served_kwh','Discharge'),
                    ('soc_end_kwh','SOC')]:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], mode='lines', name=name))
    fig.update_layout(xaxis_title='Time', yaxis_title='kWh', hovermode='x unified')
    return fig


def plot_energy_sankey(kpis):
    vals=kpis
    direct=vals['total_PV_Generado_kWh']-vals['captured_kWh']-vals['export_kWh']-vals['vertido_kWh']
    labels=["PV","Grid Import","Battery","Load","Export","Spill","Charge Loss","Discharge Loss","Self Loss","Direct PV"
            ]
    idx={l:i for i,l in enumerate(labels)}
    src=[idx['PV'],idx['PV'],idx['PV'],idx['PV'],idx['Grid Import'],idx['Battery'],idx['Battery'],idx['Battery'],]
    tgt=[idx['Battery'],idx['Export'],idx['Spill'],idx['Direct PV'],idx['Load'],idx['Load'],idx['Charge Loss'],idx['Discharge Loss'],idx['Self Loss']]
    val=[vals['captured_kWh'],vals['export_kWh'],vals['vertido_kWh'],direct,vals['import_kWh'],
         vals['descargado_kWh'],vals['loss_charge_kWh'],vals['loss_discharge_kWh'],vals['self_loss_kwh']]
    fig=go.Figure(go.Sankey(node=dict(label=labels),link=dict(source=src,target=tgt,value=val)))
    return fig

# ---------------------- Streamlit App ----------------------

def main():
    st.title("🔋⚡ Battery PV-Energy Simulator v3.1")
    st.sidebar.header("Inputs & Parameters")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv','xls','xlsx'])
    abort_on_nan = st.sidebar.checkbox("Abort on NaN", value=True)

    if not uploaded_file:
        st.info("Please upload a time-series file to begin.")
        return

    df = load_timeseries(uploaded_file)
    df_prep, dt_h = prepare_dataframe(df, abort_on_nan)

    # Sidebar parameters
    e_nom = st.sidebar.number_input("Nominal Energy E_nom (kWh)", value=15000.0)
    p_nom = st.sidebar.number_input("Nominal Power P_nom (kW)", value=4000.0)
    soc_min = st.sidebar.slider("SOC min", 0.0, 1.0, 0.1)
    soc_max = st.sidebar.slider("SOC max", 0.0, 1.0, 0.9)
    eta_rt = st.sidebar.slider("Round-trip efficiency", 0.0, 1.0, 0.9)
    self_loss_h = st.sidebar.number_input("Self-discharge per hour (fraction)", value=0.0)
    limit_export_kw = st.sidebar.number_input("Export limit (kW, 0 for none)", value=0.0)
    limit_export_kw = None if limit_export_kw==0 else limit_export_kw
    price_col = st.sidebar.text_input("Price column name (optional)") or None

    if st.sidebar.button("Run Simulation"):
        params = BatteryParams(e_nom,p_nom,soc_min,soc_max,eta_rt,self_loss_h)
        results_df, kpis = simulate_bess(df_prep, params, dt_h, limit_export_kw, price_col)

        st.subheader("Key Performance Indicators")
        st.dataframe(kpis_to_dataframe(kpis))

        st.subheader("Energy Flows Over Time")
        fig1 = plot_interactive_energy_flows(results_df)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Sankey Diagram of Energy Flows")
        fig2 = plot_energy_sankey(kpis)
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == '__main__':
    main()
