import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, List
import snowflake.connector

# --- CONFIG ---
IN_FORCE_STATUS = 'FEX Inforce'
FILTER_START_DATE = '2024-W01'

st.set_page_config(
    page_title="AMS Policy Survival Analysis by Product (FEX Inforce Duration)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- SNOWFLAKE CONNECTION ---

@st.cache_resource
def get_snowflake_connection():
    """Create a cached Snowflake connection using Streamlit secrets."""
    cfg = st.secrets["snowflake"]
    return snowflake.connector.connect(
        account=cfg["account"],
        user=cfg["user"],
        password=cfg["password"],
        warehouse=cfg["warehouse"],
        database=cfg["database"],
        schema=cfg["schema"],
        role=cfg["role"],
    )


# --- DATA LOADING ---

@st.cache_data(show_spinner="Loading and cleaning policy lapse data...")
def _load_ams_events() -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    """Loads data from Snowflake and prepares it for analysis."""
    query = """
        SELECT
            t1.POLICY_ID,
            t1.POLICY_NUMBER,
            t1.STATUS_FROM,
            t1.STATUS_TO,
            t1.TASK_DATE_FROM,
            t1.TASK_DATE_TO,
            t3.NAME        AS PRODUCT_NAME,
            t4.CARRIER_NAME AS CARRIER_NAME
        FROM raw.ams.policy_status_update t1
        INNER JOIN raw.ams.policy_status   t2 ON t1.POLICY_ID  = t2.POLICY_ID
        INNER JOIN RAW.AMS.CARRIER_PRODUCTS t3 ON t2.PRODUCT_ID = t3.ID
        INNER JOIN raw.ams.carriers         t4 ON t3.CARRIER_ID = t4.ID
    """

    try:
        conn = get_snowflake_connection()
        df = pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"❌ Failed to query Snowflake: {e}")
        return pd.DataFrame(), [], {}

    if df.empty:
        st.warning("Query returned no data.")
        return pd.DataFrame(), [], {}

    # --- DIAGNOSTIC EXPANDER ---
    df.columns = [str(c).strip().upper() for c in df.columns]
    if 'STATUS_TO' in df.columns and 'STATUS_FROM' in df.columns:
        with st.expander("🔍 Data Diagnostic — expand to debug status values", expanded=True):
            st.markdown("**Unique STATUS_TO values (top 30):**")
            st.dataframe(
                df['STATUS_TO'].astype(str).str.strip().value_counts().head(30).rename_axis('STATUS_TO').reset_index(name='Count'),
                use_container_width=True, hide_index=True
            )
            st.markdown("**Unique STATUS_FROM values (top 30):**")
            st.dataframe(
                df['STATUS_FROM'].astype(str).str.strip().value_counts().head(30).rename_axis('STATUS_FROM').reset_index(name='Count'),
                use_container_width=True, hide_index=True
            )
            st.markdown(f"**Current `IN_FORCE_STATUS` setting:** `{IN_FORCE_STATUS}`")
            match_to   = (df['STATUS_TO'].astype(str).str.strip() == IN_FORCE_STATUS).sum()
            match_from = (df['STATUS_FROM'].astype(str).str.strip() == IN_FORCE_STATUS).sum()
            st.markdown(f"**Rows where STATUS_TO matches:** `{match_to:,}`")
            st.markdown(f"**Rows where STATUS_FROM matches:** `{match_from:,}`")
            if match_to == 0:
                st.error("❌ No STATUS_TO rows match IN_FORCE_STATUS — update the IN_FORCE_STATUS constant at the top of the file to match a value from the table above.")
            elif match_from == 0:
                st.warning("⚠️ No STATUS_FROM rows match IN_FORCE_STATUS — policies will never show as lapsed.")
            else:
                st.success(f"✅ Status matching OK — {match_to:,} entries into inforce, {match_from:,} exits.")

            st.markdown("---")
            st.markdown("**Date column inspection (before renaming):**")
            for date_col in ['TASK_DATE_FROM', 'TASK_DATE_TO']:
                if date_col in df.columns:
                    parsed = pd.to_datetime(df[date_col], errors='coerce')
                    null_count = parsed.isna().sum()
                    st.markdown(f"`{date_col}` — nulls: `{null_count:,}` / `{len(df):,}` | "
                                f"min: `{parsed.min()}` | max: `{parsed.max()}`")

            st.markdown("---")
            st.markdown("**Sample of raw rows where STATUS_TO = IN_FORCE_STATUS:**")
            sample_cols = [c for c in ['POLICY_NUMBER','STATUS_FROM','STATUS_TO','TASK_DATE_FROM','TASK_DATE_TO'] if c in df.columns]
            sample = df[df['STATUS_TO'].astype(str).str.strip().str.upper() == IN_FORCE_STATUS][sample_cols].head(20)
            st.dataframe(sample, use_container_width=True, hide_index=True)

            st.markdown("**Sample of raw rows where STATUS_FROM = IN_FORCE_STATUS:**")
            sample2 = df[df['STATUS_FROM'].astype(str).str.strip().str.upper() == IN_FORCE_STATUS][sample_cols].head(20)
            st.dataframe(sample2, use_container_width=True, hide_index=True)

    df.columns = [str(c).strip().upper() for c in df.columns]

    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    st.info(f"🗑️ Removed {initial_count - len(df)} duplicate rows. {len(df)} rows remaining.")

    if 'TASK_DATE_TO' in df.columns:
        df.rename(columns={'TASK_DATE_TO': 'UPDATED_AT'}, inplace=True)
    elif 'TASK_DATE_FROM' in df.columns:
        df.rename(columns={'TASK_DATE_FROM': 'UPDATED_AT'}, inplace=True)
    else:
        st.error("❌ No date column (TASK_DATE_FROM or TASK_DATE_TO) found.")
        return pd.DataFrame(), [], {}

    required = {'POLICY_NUMBER', 'STATUS_TO', 'STATUS_FROM', 'PRODUCT_NAME', 'CARRIER_NAME', 'UPDATED_AT'}
    missing = required - set(df.columns)
    if missing:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        return pd.DataFrame(), [], {}

    df['POLICY_NUMBER'] = df['POLICY_NUMBER'].astype(str).str.strip()
    df['STATUS_TO']     = df['STATUS_TO'].astype(str).str.strip().astype('category')
    df['STATUS_FROM']   = df['STATUS_FROM'].astype(str).str.strip().astype('category')
    df['PRODUCT_NAME']  = df['PRODUCT_NAME'].astype(str).str.strip().astype('category')
    df['CARRIER_NAME']  = df['CARRIER_NAME'].astype(str).str.strip().astype('category')

    initial_date_rows = len(df)
    df['UPDATED_AT'] = pd.to_datetime(df['UPDATED_AT'], errors='coerce')
    df = df[df['UPDATED_AT'].notna()]

    failed = initial_date_rows - len(df)
    if df.empty:
        st.warning("No rows remain after date cleaning.")
        return pd.DataFrame(), [], {}
    elif failed > 0:
        st.warning(f"⚠️ {failed} rows dropped due to invalid UPDATED_AT. {len(df)} rows remaining.")
    else:
        st.success(f"✅ Data loaded and cleaned. **{len(df)}** event rows.")

    all_carriers = sorted(df['CARRIER_NAME'].cat.categories.tolist())
    carrier_product_map = (
        df.groupby('CARRIER_NAME', observed=True)['PRODUCT_NAME']
          .unique()
          .apply(lambda x: sorted([str(p) for p in x]))
          .to_dict()
    )

    return df, all_carriers, carrier_product_map


# --- COHORT ANALYSIS ---

@st.cache_data(show_spinner="Calculating policy survival...")
def calculate_retention_matrix(
    df_hash: str,
    df: pd.DataFrame,
    in_force_status: str,
    filter_product: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.NaT

    if filter_product and filter_product != "All Products":
        df = df[df['PRODUCT_NAME'] == filter_product]
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.NaT

    df_events = df.sort_values(['POLICY_NUMBER', 'UPDATED_AT']).reset_index(drop=True)
    max_event_date = df_events['UPDATED_AT'].max()

    df_enter = df_events[df_events['STATUS_TO'] == in_force_status]
    if df_enter.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.NaT

    df_enter_agg = df_enter.groupby('POLICY_NUMBER', as_index=False).agg(
        EnrollmentDate=('UPDATED_AT', 'first'),
        LastEnterDate=('UPDATED_AT', 'last'),
        PRODUCT_NAME=('PRODUCT_NAME', 'first')
    )

    df_leave = df_events[df_events['STATUS_FROM'] == in_force_status]
    df_leave_agg = (
        df_leave.groupby('POLICY_NUMBER', as_index=False).agg(LastLeaveDate=('UPDATED_AT', 'last'))
        if not df_leave.empty
        else pd.DataFrame(columns=['POLICY_NUMBER', 'LastLeaveDate'])
    )

    df_policies = df_enter_agg.merge(df_leave_agg, on='POLICY_NUMBER', how='left')
    no_leave = df_policies['LastLeaveDate'].isna()
    df_policies['IsCurrentlyActive'] = no_leave | (df_policies['LastEnterDate'] > df_policies['LastLeaveDate'])
    df_policies['EndDate'] = pd.to_datetime(np.where(df_policies['IsCurrentlyActive'], pd.NaT, df_policies['LastLeaveDate']))
    df_policies['SurvivalEndDate'] = pd.to_datetime(np.where(df_policies['IsCurrentlyActive'], max_event_date, df_policies['EndDate']))

    df_last = df_events.groupby('POLICY_NUMBER', as_index=False).agg(
        MostRecentStatus=('STATUS_TO', 'last'),
        MostRecentDate=('UPDATED_AT', 'last')
    )
    df_policies = df_policies.merge(df_last, on='POLICY_NUMBER', how='left')
    df_policies.drop(columns=['LastEnterDate', 'LastLeaveDate'], inplace=True, errors='ignore')
    df_policies = df_policies[df_policies['SurvivalEndDate'].notna() & df_policies['EnrollmentDate'].notna()]
    df_policies = df_policies[df_policies['SurvivalEndDate'] >= df_policies['EnrollmentDate']]

    if df_policies.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.NaT

    df_policies['CohortWeek'] = df_policies['EnrollmentDate'].dt.to_period('W')
    df_policies['DurationWeeks'] = (
        (df_policies['SurvivalEndDate'] - df_policies['EnrollmentDate']).dt.days / 7
    ).astype(int)

    min_cohort_week = df_policies['CohortWeek'].min()
    max_period = int((max_event_date.to_period('W') - min_cohort_week).n)
    periods = np.arange(0, min(max_period + 1, 104))

    survival_data = []
    for cohort, group in df_policies.groupby('CohortWeek'):
        durations = group['DurationWeeks'].values
        for p in periods:
            c = np.sum(durations >= p)
            if c > 0:
                survival_data.append({'CohortWeek': cohort, 'CohortPeriod': p, 'PolicyCount': c})

    if not survival_data:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), max_event_date

    df_survival = pd.DataFrame(survival_data)
    survival_matrix = df_survival.pivot_table(index='CohortWeek', columns='CohortPeriod', values='PolicyCount')
    retention_percentages = survival_matrix.div(survival_matrix.iloc[:, 0], axis=0) * 100

    retention_percentages.index = retention_percentages.index.strftime('%Y-W%V')
    survival_matrix.index = survival_matrix.index.strftime('%Y-W%V')
    df_policies['CohortWeekStr'] = df_policies['CohortWeek'].dt.strftime('%Y-W%V')

    return retention_percentages, survival_matrix, df_policies, max_event_date


def get_df_hash(df: pd.DataFrame, filter_product: Optional[str] = None) -> str:
    return f"{len(df)}_{df['POLICY_NUMBER'].nunique()}_{filter_product}"


@st.cache_data(show_spinner=False)
def calculate_combined_retention_cached(
    df_hash: str, df: pd.DataFrame, carrier: str, products_tuple: Tuple,
    max_periods: int = 52, start_cohort: Optional[str] = None, end_cohort: Optional[str] = None
) -> pd.DataFrame:
    df_f = df[(df['CARRIER_NAME'] == carrier) & (df['PRODUCT_NAME'].isin(list(products_tuple)))]
    if df_f.empty:
        return pd.DataFrame()

    ret, _, _, _ = calculate_retention_matrix(get_df_hash(df_f), df_f, IN_FORCE_STATUS, None)
    if ret.empty:
        return pd.DataFrame()

    if start_cohort and end_cohort:
        ret = ret[ret.index.isin([c for c in ret.index if start_cohort <= c <= end_cohort])]
        if ret.empty:
            return pd.DataFrame()

    avg_data = [
        (p, ret[p].dropna().mean())
        for p in range(min(max_periods, ret.shape[1]))
        if p in ret.columns and len(ret[p].dropna()) >= 3
    ]
    if avg_data:
        periods, values = zip(*avg_data)
        avg_df = pd.DataFrame({'CohortPeriod': periods, 'AverageSurvival': values})
        if len(values) > 2 and values[-1] < values[-2] * 0.9:
            avg_df = avg_df.iloc[:-1]
        return avg_df
    return pd.DataFrame()


def calculate_combined_retention(df, carrier, products, start_cohort=None, end_cohort=None):
    rng = f"{start_cohort or 'all'}-{end_cohort or 'all'}"
    return calculate_combined_retention_cached(
        f"{len(df)}_{carrier}_{'-'.join(sorted(products))}_{rng}",
        df, carrier, tuple(products), 52, start_cohort, end_cohort
    )


@st.cache_data(show_spinner=False)
def get_all_available_cohorts_cached(df_hash: str, df: pd.DataFrame) -> List[str]:
    ret, _, _, _ = calculate_retention_matrix(get_df_hash(df), df, IN_FORCE_STATUS, None)
    return sorted(ret.index.tolist()) if not ret.empty else []


def get_all_available_cohorts(df):
    return get_all_available_cohorts_cached(f"all_cohorts_{len(df)}_{df['POLICY_NUMBER'].nunique()}", df)


# --- MONTHLY COHORT ---

@st.cache_data(show_spinner="Calculating monthly cohort analysis...")
def calculate_monthly_cohort_analysis(
    df_hash: str, df: pd.DataFrame, in_force_status: str,
    start_month: str = '2024-11', filter_product=None, filter_carrier=None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.NaT

    df_f = df.copy()
    if filter_product:
        df_f = df_f[df_f['PRODUCT_NAME'] == filter_product]
    if filter_carrier:
        df_f = df_f[df_f['CARRIER_NAME'] == filter_carrier]
    if df_f.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.NaT

    df_events = df_f.sort_values(['POLICY_NUMBER', 'UPDATED_AT']).reset_index(drop=True)
    max_event_date = df_events['UPDATED_AT'].max()

    df_enter = df_events[df_events['STATUS_TO'] == in_force_status]
    if df_enter.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.NaT

    df_enter_agg = df_enter.groupby('POLICY_NUMBER', as_index=False).agg(
        EnrollmentDate=('UPDATED_AT', 'first'), LastEnterDate=('UPDATED_AT', 'last'),
        PRODUCT_NAME=('PRODUCT_NAME', 'first'), CARRIER_NAME=('CARRIER_NAME', 'first')
    )
    df_leave = df_events[df_events['STATUS_FROM'] == in_force_status]
    df_leave_agg = (
        df_leave.groupby('POLICY_NUMBER', as_index=False).agg(LastLeaveDate=('UPDATED_AT', 'last'))
        if not df_leave.empty else pd.DataFrame(columns=['POLICY_NUMBER', 'LastLeaveDate'])
    )

    df_p = df_enter_agg.merge(df_leave_agg, on='POLICY_NUMBER', how='left')
    df_p['IsCurrentlyActive'] = df_p['LastLeaveDate'].isna() | (df_p['LastEnterDate'] > df_p['LastLeaveDate'])
    df_p['LapseDate'] = pd.to_datetime(np.where(df_p['IsCurrentlyActive'], pd.NaT, df_p['LastLeaveDate']))
    df_p['SurvivalEndDate'] = pd.to_datetime(np.where(df_p['IsCurrentlyActive'], max_event_date, df_p['LapseDate']))

    df_last = df_events.groupby('POLICY_NUMBER', as_index=False).agg(
        MostRecentStatus=('STATUS_TO', 'last'), MostRecentDate=('UPDATED_AT', 'last')
    )
    df_p = df_p.merge(df_last, on='POLICY_NUMBER', how='left')
    df_p.drop(columns=['LastEnterDate', 'LastLeaveDate'], inplace=True, errors='ignore')
    df_p = df_p[df_p['SurvivalEndDate'].notna() & df_p['EnrollmentDate'].notna()]
    df_p = df_p[df_p['SurvivalEndDate'] >= df_p['EnrollmentDate']]
    if df_p.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.NaT

    df_p['CohortMonth'] = df_p['EnrollmentDate'].dt.to_period('M')
    df_p = df_p[df_p['CohortMonth'] >= pd.Period(start_month, freq='M')]
    if df_p.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.NaT

    df_p['DurationMonths'] = ((df_p['SurvivalEndDate'] - df_p['EnrollmentDate']).dt.days / 30.44).astype(int)

    start_period = pd.Period(start_month, freq='M')
    max_months = int((max_event_date.to_period('M') - start_period).n) + 1
    periods = np.arange(0, min(max_months + 1, 25))

    survival_data = []
    for cohort, group in df_p.groupby('CohortMonth'):
        initial = len(group)
        durations = group['DurationMonths'].values
        cohort_end = (cohort + 1).to_timestamp() - pd.Timedelta(days=1)
        max_complete = int((max_event_date - cohort_end).days / 30.44)
        for p in periods:
            if p > max_complete:
                continue
            survived = np.sum(durations >= p)
            survival_data.append({
                'CohortMonth': cohort.strftime('%Y-%m'),
                'MonthsElapsed': p,
                'SurvivalPct': (survived / initial * 100) if initial > 0 else 0
            })

    if not survival_data:
        return pd.DataFrame(), df_p, max_event_date

    df_s = pd.DataFrame(survival_data)
    matrix = df_s.pivot_table(index='CohortMonth', columns='MonthsElapsed', values='SurvivalPct')
    return matrix, df_p, max_event_date


# --- VISUALIZATION ---

def plot_cohort_curves_with_selection(retention_percentages, selected_cohorts, title_suffix=""):
    if retention_percentages.empty:
        return go.Figure().update_layout(title="No Data Available")
    fig = go.Figure()
    period_values = {}
    for cohort, retention in retention_percentages.iterrows():
        valid = retention.dropna()
        if len(valid) > 2:
            valid = valid.iloc[:-1]
        if len(valid) > 1:
            sel = cohort in selected_cohorts
            fig.add_trace(go.Scatter(
                x=valid.index.astype(int), y=valid.values, mode='lines',
                name=f'Cohort {cohort}', opacity=0.6 if sel else 0.15,
                line=dict(width=1.5 if sel else 1), legendgroup='cohorts',
                hovertemplate=f"<b>%{{y:.1f}}%</b><br>Week: %{{x}}<br>Cohort: {cohort}<extra></extra>",
            ))
            if sel:
                for p, v in zip(valid.index.astype(int), valid.values):
                    period_values.setdefault(p, []).append(v)

    if period_values:
        avg_periods = sorted([p for p in period_values if len(period_values[p]) >= 2])
        if avg_periods:
            fig.add_trace(go.Scatter(
                x=avg_periods, y=[np.mean(period_values[p]) for p in avg_periods],
                mode='lines', name=f'Average ({len(selected_cohorts)} cohorts)',
                line=dict(color='black', width=4, dash='dot'),
                hovertemplate="<b>Average: %{y:.1f}%</b><br>Week: %{x}<extra></extra>",
            ))
    fig.update_layout(
        title=f"Policy Survival Curves (Duration of '{IN_FORCE_STATUS}'){title_suffix}",
        xaxis_title='Cohort Period (Weeks Since Enrollment)',
        yaxis_title='Survival Percentage (%)', yaxis_tickformat='.0f',
        yaxis_range=[0, 100], template='plotly_white', hovermode="closest"
    )
    return fig


@st.cache_data(show_spinner=False)
def plot_product_comparison_cached(df_hash, df, max_periods=52):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for idx, product in enumerate(df['PRODUCT_NAME'].unique()):
        ret, _, _, _ = calculate_retention_matrix(get_df_hash(df, product), df, IN_FORCE_STATUS, product)
        if not ret.empty:
            avg_data = [
                (p, ret[p].dropna().mean())
                for p in range(min(max_periods, ret.shape[1]))
                if p in ret.columns and len(ret[p].dropna()) >= 3
            ]
            if avg_data:
                periods, values = zip(*avg_data)
                if len(values) > 2 and values[-1] < values[-2] * 0.9:
                    periods, values = periods[:-1], values[:-1]
                fig.add_trace(go.Scatter(
                    x=list(periods), y=list(values), mode='lines+markers', name=str(product),
                    line=dict(color=colors[idx % len(colors)], width=3), marker=dict(size=6)
                ))
    fig.update_layout(
        title="Average Survival Curves by Product Type",
        xaxis_title='Weeks Since Enrollment', yaxis_title='Average Survival Percentage (%)',
        yaxis_tickformat='.0f', yaxis_range=[0, 100], template='plotly_white', hovermode="x unified",
        legend=dict(yanchor="top", y=-0.2, xanchor="center", x=0.5, orientation='h')
    )
    return fig


def plot_combination_comparison(comparison_data, cohort_info=""):
    fig = go.Figure()
    colors = px.colors.qualitative.Dark24
    for idx, (label, df_avg) in enumerate(comparison_data.items()):
        if not df_avg.empty:
            fig.add_trace(go.Scatter(
                x=df_avg['CohortPeriod'], y=df_avg['AverageSurvival'],
                mode='lines+markers', name=label,
                line=dict(color=colors[idx % len(colors)], width=4), marker=dict(size=7)
            ))
    title = "Custom Combination Survival Comparison"
    if cohort_info:
        title += f"<br><sub>{cohort_info}</sub>"
    fig.update_layout(
        title=title, xaxis_title='Weeks Since Enrollment',
        yaxis_title='Average Survival Percentage (%)', yaxis_tickformat='.0f',
        yaxis_range=[0, 100], template='plotly_white', hovermode="x unified",
        legend=dict(yanchor="top", y=-0.2, xanchor="center", x=0.5, orientation='h')
    )
    return fig


def plot_monthly_cohort_survival(survival_matrix):
    if survival_matrix.empty:
        return go.Figure().update_layout(title="No Data Available")
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for idx, (cohort, row) in enumerate(survival_matrix.iterrows()):
        valid = row.dropna()
        if len(valid) > 1:
            fig.add_trace(go.Scatter(
                x=valid.index.astype(int), y=valid.values, mode='lines+markers', name=cohort,
                line=dict(color=colors[idx % len(colors)], width=2), marker=dict(size=6),
                hovertemplate=f"<b>{cohort}</b><br>Month %{{x}}: %{{y:.1f}}%<extra></extra>"
            ))
    fig.update_layout(
        title="Monthly Cohort Survival Curves", xaxis_title='Months Since Policy Start',
        yaxis_title='Survival Percentage (%)', yaxis_tickformat='.0f',
        yaxis_range=[0, 105], xaxis=dict(dtick=1), template='plotly_white', hovermode="closest"
    )
    return fig


def plot_monthly_lapse_rates(survival_matrix):
    if survival_matrix.empty:
        return go.Figure().update_layout(title="No Data Available")
    lapse_data = []
    for cohort, row in survival_matrix.iterrows():
        valid = row.dropna()
        for i in range(1, len(valid)):
            prev = valid.iloc[i-1]
            if prev > 0:
                lapse_data.append({'Cohort': cohort, 'Month': int(valid.index[i]),
                                    'MonthlyLapsePct': ((prev - valid.iloc[i]) / prev) * 100})
    if not lapse_data:
        return go.Figure().update_layout(title="No Data Available")
    fig = px.bar(pd.DataFrame(lapse_data), x='Month', y='MonthlyLapsePct', color='Cohort',
                 barmode='group', title="Month-over-Month Lapse Rate by Cohort")
    fig.update_layout(xaxis_title='Month Number', yaxis_title='Lapse Rate (%)',
                      yaxis_tickformat='.1f', xaxis=dict(dtick=1), template='plotly_white')
    return fig


# --- HEATMAP HELPER ---

def style_retention_heatmap(retention_pct: pd.DataFrame, policy_counts: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Returns a Styler with a RdYlGn background gradient.
    Each cell displays 'XX.X% (N)' where N is the surviving policy count.
    """
    counts_aligned = policy_counts.reindex_like(retention_pct)

    def fmt(pct, cnt):
        if pd.isna(pct):
            return ""
        if pd.notna(cnt):
            return f"{pct:.1f}% ({int(cnt)})"
        return f"{pct:.1f}%"

    # Build a same-shape DataFrame of display strings
    display_df = pd.DataFrame(
        [[fmt(retention_pct.loc[r, c],
              counts_aligned.loc[r, c] if c in counts_aligned.columns else np.nan)
          for c in retention_pct.columns]
         for r in retention_pct.index],
        index=retention_pct.index,
        columns=retention_pct.columns,
    )

    # Apply gradient using the numeric retention_pct as the color map source
    styler = (
        display_df.style
        .background_gradient(cmap='RdYlGn', gmap=retention_pct.values, axis=None, vmin=0, vmax=100)
    )
    return styler


# --- DISPLAY FUNCTIONS ---

def display_overview(retention_percentages, policy_count_matrix, selected_product, start_cohort, end_cohort):
    product_suffix = f" - {selected_product}" if selected_product != "All Products" else ""
    date_range_suffix = f" ({start_cohort} to {end_cohort})"
    st.header(f"📊 1. Policy Survival Curves{product_suffix}{date_range_suffix}")

    available_cohorts = retention_percentages.index.tolist()
    if not available_cohorts:
        st.warning("No cohorts available.")
        return

    col1, col2 = st.columns(2)
    with col1:
        avg_start = st.selectbox("From Cohort", available_cohorts, index=0, key='avg_start_cohort')
    with col2:
        end_opts = [c for c in available_cohorts if c >= avg_start]
        avg_end = st.selectbox("To Cohort", end_opts, index=len(end_opts)-1, key='avg_end_cohort')

    selected_cohorts = [c for c in available_cohorts if avg_start <= c <= avg_end]
    st.caption(f"Average includes **{len(selected_cohorts)}** cohorts from {avg_start} to {avg_end}")
    st.plotly_chart(plot_cohort_curves_with_selection(retention_percentages, selected_cohorts, product_suffix), use_container_width=True)

    st.markdown("---")
    st.header(f"📈 2. Survival Percentage Table{product_suffix}{date_range_suffix}")
    st.dataframe(
        style_retention_heatmap(retention_percentages, policy_count_matrix),
        use_container_width=True
    )

    st.markdown("---")
    st.header(f"🔢 3. Raw Policy Count Table{product_suffix}{date_range_suffix}")
    st.dataframe(policy_count_matrix.astype('Int64'), use_container_width=True)


def display_cohort_deep_dive(df_policies_detail, policy_count_matrix, max_event_date):
    st.title("🔎 Cohort Deep Dive")
    st.markdown("---")
    cohort_options = policy_count_matrix.index.tolist()
    if not cohort_options:
        st.warning("No cohorts available.")
        return

    cohort_dates = df_policies_detail.groupby('CohortWeekStr')['EnrollmentDate'].min().sort_index()
    cohort_labels = {
        cw: f"{cw} (starts {cohort_dates[cw].strftime('%Y-%m-%d')})" if cw in cohort_dates else cw
        for cw in cohort_options
    }
    selected_cohort = st.selectbox(
        "Select Cohort Week", cohort_options, index=len(cohort_options)-1,
        format_func=lambda x: cohort_labels.get(x, x), key='deep_dive_cohort_select'
    )

    cp = df_policies_detail[df_policies_detail['CohortWeekStr'] == selected_cohort].copy()
    cp['DurationWeeks'] = ((cp['SurvivalEndDate'] - cp['EnrollmentDate']).dt.days / 7).round(1)
    initial_size = len(cp)
    active = cp['IsCurrentlyActive'].sum()
    st.info(f"Cohort Size: **{initial_size}** | Active: **{active}** | Terminated: **{initial_size - active}**")

    pb = cp['PRODUCT_NAME'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Product Distribution")
        st.dataframe(pb.to_frame(name="Count"))
    with col2:
        st.plotly_chart(px.pie(values=pb.values, names=pb.index, title="Product Mix"), use_container_width=True)

    if selected_cohort in policy_count_matrix.index:
        counts = policy_count_matrix.loc[selected_cohort].dropna()
        if len(counts) > 2:
            counts = counts.iloc[:-1]
        ret = (counts / counts.iloc[0]) * 100
        fig = go.Figure(go.Scatter(x=ret.index.astype(int), y=ret.values, mode='lines+markers',
                                    line=dict(color='blue', width=3)))
        fig.update_layout(title=f"Survival Curve for {selected_cohort}", xaxis_title='Weeks Since Enrollment',
                           yaxis_title='Survival %', yaxis_range=[0, 100], template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### Policy Details for **{selected_cohort}**")
    display_cols = ['POLICY_NUMBER', 'PRODUCT_NAME', 'EnrollmentDate', 'EndDate', 'IsCurrentlyActive',
                    'MostRecentStatus', 'MostRecentDate', 'SurvivalEndDate', 'DurationWeeks']
    disp = cp[display_cols].copy()
    for col in ['EnrollmentDate', 'EndDate', 'MostRecentDate', 'SurvivalEndDate']:
        disp[col] = disp[col].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A')
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(f"Data cutoff: {max_event_date.strftime('%Y-%m-%d') if pd.notna(max_event_date) else 'N/A'}")


def display_product_comparison(df_ams):
    st.title("📊 Product Comparison")
    st.markdown("---")
    st.plotly_chart(plot_product_comparison_cached(get_df_hash(df_ams), df_ams), use_container_width=True)
    st.header("Product Statistics")
    rows = []
    for product in df_ams['PRODUCT_NAME'].unique():
        ret, _, _, _ = calculate_retention_matrix(get_df_hash(df_ams, product), df_ams, IN_FORCE_STATUS, product)
        if not ret.empty:
            avg = ret.mean(axis=0)
            def gs(a, p):
                return a[p] if p in a.index else (a.iloc[p] if p < len(a) else None)
            rows.append({
                'Product': product,
                'Total Policies': df_ams[df_ams['PRODUCT_NAME'] == product]['POLICY_NUMBER'].nunique(),
                **{f'Week {w} %': (f"{gs(avg,w):.1f}%" if gs(avg,w) is not None else "N/A") for w in [4,13,26,52]}
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def display_persistency_trends(df_ams, df_policies_detail, retention_percentages):
    st.title("📈 Persistency Trends Over Time")
    st.markdown("---")
    if df_policies_detail.empty or retention_percentages.empty:
        st.warning("No data available.")
        return

    available_cohorts = sorted(retention_percentages.index.tolist())
    if len(available_cohorts) < 4:
        st.warning("Need at least 4 cohorts.")
        return

    cohort_dates = df_policies_detail.groupby('CohortWeekStr')['EnrollmentDate'].min().sort_index()
    col1, col2, col3 = st.columns(3)
    with col1:
        window_months = st.selectbox("Cohort Window (months)", [1,2,4,8], index=2)
    with col2:
        step_months = st.selectbox("Step Size (months)", [1,2,3,6], index=0)
    with col3:
        max_months_display = st.selectbox("Max Months Display", [3,6,9,12,18,24], index=3)

    max_weeks_display = max_months_display * 4

    # ── Milestone checkboxes (single set, no duplicates) ──────────────────────
    st.markdown("**Select milestone months:**")
    mcols = st.columns(6)
    selected_milestones_months = []
    for i in range(12):
        with mcols[i % 6]:
            if st.checkbox(f"Mo {i+1}", value=True, key=f"ms_mo_{i+1}"):
                selected_milestones_months.append(i + 1)
    # ──────────────────────────────────────────────────────────────────────────

    selected_milestones = [int(m * 4.33) for m in sorted(selected_milestones_months)]
    st.markdown("---")

    valid_cohorts = [(cw, cohort_dates[cw]) for cw in available_cohorts if cw in cohort_dates]
    if not valid_cohorts:
        st.warning("Could not parse cohort dates.")
        return

    min_date = min(d for _, d in valid_cohorts)
    max_date = max(d for _, d in valid_cohorts)
    obs_points, cur = [], min_date + pd.DateOffset(months=window_months)
    while cur <= max_date + pd.DateOffset(months=1):
        obs_points.append(cur)
        cur += pd.DateOffset(months=step_months)

    if len(obs_points) < 2:
        st.warning("Not enough data range.")
        return

    with st.spinner("Calculating snapshots..."):
        snapshot_curves, snapshot_milestones, snapshot_counts = {}, {}, {}
        for obs in obs_points:
            ws = obs - pd.DateOffset(months=window_months)
            included = [cw for cw, dt in valid_cohorts if ws <= dt < obs]
            if len(included) < 3:
                continue
            filtered = retention_percentages[retention_percentages.index.isin(included)]
            if filtered.empty:
                continue
            avg_curve = {p: filtered[p].dropna().mean()
                         for p in range(max_weeks_display + 1)
                         if p in filtered.columns and len(filtered[p].dropna()) >= 3}
            if avg_curve:
                lbl = obs.strftime('%Y-%m')
                snapshot_curves[lbl] = avg_curve
                snapshot_counts[lbl] = len(included)
                snapshot_milestones[lbl] = {m: avg_curve[m] for m in selected_milestones if m in avg_curve}

    if not snapshot_curves:
        st.warning("Could not generate snapshots.")
        return

    st.success(f"Generated **{len(snapshot_curves)}** snapshots")
    snapshot_dates = list(snapshot_curves.keys())

    col1, col2 = st.columns([3,1])
    with col1:
        mode = st.radio("Comparison Mode", ["Latest vs Historical","Custom Selection","Show All"], horizontal=True)
    with col2:
        show_counts = st.checkbox("Show cohort counts", value=True)

    if mode == "Latest vs Historical":
        sel_snaps = sorted(set([snapshot_dates[-1]] +
            [snapshot_dates[max(0, len(snapshot_dates)-1-(m//step_months))] for m in [3,6,12]]))
    elif mode == "Custom Selection":
        sel_snaps = st.multiselect("Select observation points", snapshot_dates,
                                    default=[snapshot_dates[0], snapshot_dates[-1]])
    else:
        sel_snaps = snapshot_dates

    if sel_snaps:
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for idx, obs in enumerate(sel_snaps):
            curve = snapshot_curves[obs]
            ps = sorted(curve.keys())
            lbl = f"{obs} ({snapshot_counts[obs]} cohorts)" if show_counts else obs
            fig.add_trace(go.Scatter(
                x=[p/4.33 for p in ps], y=[curve[p] for p in ps],
                mode='lines+markers', name=lbl,
                line=dict(color=colors[idx % len(colors)], width=3), marker=dict(size=5)
            ))
        fig.update_layout(
            title=f"Survival Curves by Observation Point ({window_months}-month window)",
            xaxis_title='Months Since Enrollment', yaxis_title='Average Survival %',
            yaxis_range=[0,100], template='plotly_white',
            legend=dict(yanchor="top", y=-0.15, xanchor="center", x=0.5, orientation='h')
        )
        st.plotly_chart(fig, use_container_width=True)

    if selected_milestones_months:
        st.header("Milestone Persistency Trends")
        md = [{'Observation': obs, 'Month': f'Month {round(w/4.33)}', 'Survival %': v}
              for obs, ms in snapshot_milestones.items() for w, v in ms.items()]
        if md:
            fig_m = px.line(pd.DataFrame(md), x='Observation', y='Survival %', color='Month',
                            markers=True, title=f"Persistency Milestones ({window_months}-month window)")
            fig_m.update_layout(yaxis_range=[0,100], template='plotly_white',
                                  legend=dict(yanchor="top", y=-0.15, xanchor="center", x=0.5, orientation='h'))
            st.plotly_chart(fig_m, use_container_width=True)


def display_combination_comparison(df_ams_raw, all_carriers, carrier_product_map):
    st.title("⚖️ Custom Combination Survival Comparison")
    st.markdown("---")
    all_cohorts = get_all_available_cohorts(df_ams_raw)
    if not all_cohorts:
        st.warning("No cohorts available.")
        return

    default_start_idx = next((i for i, c in enumerate(all_cohorts) if c >= FILTER_START_DATE), 0)
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        combo_start = st.selectbox("Start Cohort", all_cohorts, index=default_start_idx, key='combo_start_cohort')
    with col2:
        end_opts = [c for c in all_cohorts if c >= combo_start]
        combo_end = st.selectbox("End Cohort", end_opts, index=len(end_opts)-1, key='combo_end_cohort')
    with col3:
        use_filter = st.checkbox("Apply Filter", value=True, key='use_combo_cohort_filter')

    filtered_count = len([c for c in all_cohorts if combo_start <= c <= combo_end])
    st.info(f"📊 Using **{filtered_count if use_filter else len(all_cohorts)}** cohorts")
    st.markdown("---")

    num_groups = st.selectbox("Number of Groups (2-8)", list(range(2,9)), index=0)
    st.subheader("Define Comparison Groups")
    num_cols = min(num_groups, 4)
    input_keys = []

    for g in range(num_groups):
        if g % num_cols == 0:
            cols = st.columns(num_cols)
        with cols[g % num_cols]:
            st.markdown(f"**Group {g+1}**")
            custom_name = st.text_input("Name (optional)", value="", key=f'name_group_{g}', placeholder="Optional")
            sel_carrier = st.selectbox("Carrier", all_carriers, key=f'carrier_group_{g}')
            prods = carrier_product_map.get(sel_carrier, [])
            st.multiselect("Products", prods, default=prods[:min(3,len(prods))], key=f'products_group_{g}')
            input_keys.append((f'name_group_{g}', f'carrier_group_{g}', f'products_group_{g}', f"Group {g+1}"))

    st.markdown("---")
    if st.button("Generate Comparison Chart 🚀", type="primary", use_container_width=True):
        comparison_data = {}
        start_p = combo_start if use_filter else None
        end_p = combo_end if use_filter else None
        with st.spinner("Calculating..."):
            for nk, ck, pk, gl in input_keys:
                products = st.session_state.get(pk, [])
                carrier = st.session_state.get(ck, "")
                if not products or not carrier:
                    continue
                label = st.session_state.get(nk, "").strip() or f"{carrier} - {gl}"
                avg_df = calculate_combined_retention(df_ams_raw, carrier, products, start_p, end_p)
                if not avg_df.empty:
                    comparison_data[label] = avg_df
                else:
                    st.warning(f"❌ No valid data for {gl}: {carrier}")

        if comparison_data:
            ci = f"Cohorts: {combo_start} to {combo_end} ({filtered_count} cohorts)" if use_filter else "All cohorts"
            st.plotly_chart(plot_combination_comparison(comparison_data, ci), use_container_width=True)
            rows = []
            for label, df_avg in comparison_data.items():
                row = {'Group': label}
                for w in [4,13,26,39,52]:
                    d = df_avg[df_avg['CohortPeriod'] == w]
                    row[f'Week {w}'] = f"{d['AverageSurvival'].values[0]:.1f}%" if not d.empty else "N/A"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.error("No valid data for any group.")


def display_monthly_cohort_analysis(df_ams_raw, all_carriers):
    st.title("📅 Monthly Cohort Analysis")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_carrier = st.selectbox("Filter by Carrier", ["All Carriers"]+all_carriers, key='monthly_carrier')
    with col2:
        prods = sorted(df_ams_raw['PRODUCT_NAME'].unique().tolist()) if sel_carrier == "All Carriers" \
            else sorted(df_ams_raw[df_ams_raw['CARRIER_NAME']==sel_carrier]['PRODUCT_NAME'].unique().tolist())
        sel_product = st.selectbox("Filter by Product", ["All Products"]+prods, key='monthly_product')
    with col3:
        start_month = st.selectbox("Start Month",
            ['2024-11','2024-10','2024-09','2024-08','2024-07','2024-06','2024-01','2023-01'], key='monthly_start')

    fc = None if sel_carrier == "All Carriers" else sel_carrier
    fp = None if sel_product == "All Products" else sel_product
    df_hash = f"monthly_{len(df_ams_raw)}_{sel_carrier}_{sel_product}_{start_month}"
    matrix, df_p, max_dt = calculate_monthly_cohort_analysis(df_hash, df_ams_raw, IN_FORCE_STATUS, start_month, fp, fc)

    if matrix.empty:
        st.warning("No data available.")
        return

    total = len(df_p)
    active = df_p['IsCurrentlyActive'].sum()
    c = st.columns(4)
    c[0].metric("Total Policies", f"{total:,}")
    c[1].metric("Currently Active", f"{active:,}", delta=f"{active/total*100:.1f}%")
    c[2].metric("Lapsed", f"{total-active:,}", delta=f"-{(total-active)/total*100:.1f}%")
    c[3].metric("Monthly Cohorts", len(matrix))
    st.markdown("---")

    # For the monthly heatmap we need counts alongside the survival pct matrix.
    # Reconstruct counts from the survival pct matrix + cohort sizes from df_p.
    cohort_sizes = df_p.groupby(df_p['EnrollmentDate'].dt.to_period('M').dt.strftime('%Y-%m'))['POLICY_NUMBER'].count()
    monthly_counts = matrix.copy()
    for col in matrix.columns:
        for row in matrix.index:
            sz = cohort_sizes.get(row, np.nan)
            pct = matrix.loc[row, col]
            monthly_counts.loc[row, col] = round(pct / 100 * sz) if pd.notna(pct) and pd.notna(sz) else np.nan

    st.header("1. Monthly Cohort Survival Curves")
    st.plotly_chart(plot_monthly_cohort_survival(matrix), use_container_width=True)
    st.markdown("---")
    st.header("2. Survival Percentage Table")
    st.dataframe(
        style_retention_heatmap(matrix, monthly_counts),
        use_container_width=True
    )
    st.subheader("Lapse Percentage Table")
    lapse_matrix = 100 - matrix
    st.dataframe(
        lapse_matrix.style
            .format('{:.1f}%', na_rep="—")
            .background_gradient(cmap='RdYlGn_r', axis=None, vmin=0, vmax=100),
        use_container_width=True
    )
    st.markdown("---")
    st.header("3. Month-over-Month Lapse Rates")
    st.plotly_chart(plot_monthly_lapse_rates(matrix), use_container_width=True)


# --- MAIN ---

def main():
    st.title("AMS Policy Survival Analysis by Product")
    st.markdown(f"Analyzes **weekly policy survival** based on duration in **'{IN_FORCE_STATUS}'** status.")

    df_ams_raw, all_carriers_list, carrier_product_map = _load_ams_events()
    if df_ams_raw.empty:
        st.stop()

    st.sidebar.header("Global Filters")
    sel_carrier = st.sidebar.selectbox("Select Carrier", ["All Carriers"] + all_carriers_list)
    df_ams = df_ams_raw if sel_carrier == "All Carriers" else df_ams_raw[df_ams_raw['CARRIER_NAME'] == sel_carrier]
    if df_ams.empty:
        st.warning(f"No data for carrier: **{sel_carrier}**")
        st.stop()

    products = ["All Products"] + sorted(df_ams['PRODUCT_NAME'].unique().tolist())
    sel_product = st.sidebar.selectbox("Select Product Type", products)

    hash_key = get_df_hash(df_ams, None if sel_product == "All Products" else sel_product)
    ret, counts, df_detail, max_dt = calculate_retention_matrix(
        hash_key, df_ams, IN_FORCE_STATUS,
        None if sel_product == "All Products" else sel_product
    )
    if ret.empty:
        st.warning("Could not generate survival matrix.")
        st.stop()

    available_cohorts = sorted(ret.index.tolist())
    cohort_dates = df_detail.groupby('CohortWeekStr')['EnrollmentDate'].min().sort_index()
    cohort_labels = {cw: f"{cw} ({cohort_dates[cw].strftime('%Y-%m-%d')})" if cw in cohort_dates else cw
                     for cw in available_cohorts}

    default_start_idx = next((i for i, c in enumerate(available_cohorts) if c >= FILTER_START_DATE), 0)
    st.sidebar.markdown("### Cohort Date Range")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_cohort = st.selectbox("Start", available_cohorts, index=default_start_idx,
                                     format_func=lambda x: cohort_labels.get(x, x), key='overview_start_cohort')
    with c2:
        end_opts = [c for c in available_cohorts if c >= start_cohort]
        end_cohort = st.selectbox("End", end_opts, index=len(end_opts)-1,
                                   format_func=lambda x: cohort_labels.get(x, x), key='overview_end_cohort')

    if start_cohort in cohort_dates and end_cohort in cohort_dates:
        st.sidebar.info(f"📅 {cohort_dates[start_cohort].strftime('%Y-%m-%d')} → {cohort_dates[end_cohort].strftime('%Y-%m-%d')}")

    filtered_ret = ret[(ret.index >= start_cohort) & (ret.index <= end_cohort)]
    filtered_counts = counts[(counts.index >= start_cohort) & (counts.index <= end_cohort)]
    if filtered_ret.empty:
        st.warning("No data for selected cohort range.")
        st.stop()

    tabs = st.tabs(["Cohort Overview", "Cohort Deep Dive", "Product Comparison",
                    "Persistency Trends", "📅 Monthly Cohorts", "Combination Comparison"])
    with tabs[0]:
        display_overview(filtered_ret, filtered_counts, sel_product, start_cohort, end_cohort)
    with tabs[1]:
        display_cohort_deep_dive(df_detail, counts, max_dt)
    with tabs[2]:
        display_product_comparison(df_ams)
    with tabs[3]:
        display_persistency_trends(df_ams, df_detail, ret)
    with tabs[4]:
        display_monthly_cohort_analysis(df_ams_raw, all_carriers_list)
    with tabs[5]:
        display_combination_comparison(df_ams_raw, all_carriers_list, carrier_product_map)


if __name__ == '__main__':
    main()
