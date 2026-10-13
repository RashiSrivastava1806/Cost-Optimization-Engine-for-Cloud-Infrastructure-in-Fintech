
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "optimization"

st.set_page_config(
    page_title="FinOpt | Cloud Cost Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#101A33"
PANEL = "#182542"
BLUE = "#638CFF"
CYAN = "#43D9D0"
GREEN = "#47D7A0"
ORANGE = "#FFB86B"
RED = "#FF7185"
TEXT = "#F3F6FF"
MUTED = "#A7B4D0"

# --------------------------------------------------
# 2. STYLING
# --------------------------------------------------

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {NAVY};
        color: {TEXT};
    }}

    [data-testid="stSidebar"] {{
        background: #0B1328;
        border-right: 1px solid #293957;
    }}

    h1, h2, h3 {{
        color: {TEXT} !important;
        letter-spacing: -0.4px;
    }}

    p, label, .stMarkdown {{
        color: {MUTED};
    }}

    [data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid #2B3C60;
        padding: 18px;
        border-radius: 14px;
    }}

    [data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}

    [data-testid="stMetricValue"] {{
        color: {TEXT};
    }}

    div[data-testid="stTabs"] button {{
        color: {MUTED};
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid #2B3C60;
        border-radius: 10px;
    }}

    .hero {{
        background: linear-gradient(120deg, #1B2D54, #172342);
        padding: 25px 28px;
        border-radius: 18px;
        border: 1px solid #344A73;
        margin-bottom: 20px;
    }}

    .eyebrow {{
        color: {CYAN};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }}

    .hero-subtitle {{
        color: {MUTED};
        font-size: 14px;
    }}

    .notice {{
        background: #332B22;
        border: 1px solid #8A693D;
        color: #FFE3B5;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0 20px 0;
    }}

    .section-note {{
        color: {MUTED};
        font-size: 13px;
    }}

    #MainMenu, footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_csv(path, modified_time):
    return pd.read_csv(path)


def load_first_existing(paths):
    for path in paths:
        if path.exists():
            try:
                return load_csv(str(path), path.stat().st_mtime), path
            except Exception:
                continue
    return pd.DataFrame(), None


billing, billing_path = load_first_existing([
    DATA / "cleaned_cloud_billing_data_canonical.csv",
    DATA / "cleaned_cloud_billing_data.csv",
    DATA / "cloud_billing_data.csv",
])

recommendations, recommendations_path = load_first_existing([
    OUTPUT / "all_recommendations.csv",
])

if billing.empty:
    st.error(
        "Billing data could not be loaded. Check the CSV files "
        "inside the project's data/ folder."
    )
    st.stop()

# Standardize date and cost columns.
if "Usage_Start_Date" in billing.columns:
    billing["Usage_Start_Date"] = pd.to_datetime(
        billing["Usage_Start_Date"],
        errors="coerce",
    )
    DATE_COL = "Usage_Start_Date"
else:
    DATE_COL = None

if "Total_Cost_INR" in billing.columns:
    COST_COL = "Total_Cost_INR"
elif "Rounded_Cost" in billing.columns:
    COST_COL = "Rounded_Cost"
else:
    st.error("No recognized cost column was found in billing data.")
    st.stop()

billing[COST_COL] = pd.to_numeric(
    billing[COST_COL],
    errors="coerce",
)

billing = billing.dropna(subset=[COST_COL]).copy()
billing = billing[billing[COST_COL] >= 0].copy()

if DATE_COL:
    billing = billing.dropna(subset=[DATE_COL]).copy()

# --------------------------------------------------
# 4. CHART THEME
# --------------------------------------------------

def style_chart(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hoverlabel=dict(
            bgcolor=PANEL,
            font_color=TEXT,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#344462",
    )
    fig.update_yaxes(
        gridcolor="#293957",
        zeroline=False,
    )
    return fig


def money(value):
    return f"₹{value:,.0f}"


# --------------------------------------------------
# 5. SIDEBAR AND FILTERS
# --------------------------------------------------

with st.sidebar:
    st.markdown("## FINOPT")
    st.caption("CLOUD COST INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Cost Analytics",
            "Forecasting",
            "Optimization Center",
            "Data & Methodology",
        ],
    )

    st.divider()
    st.markdown("### Dashboard filters")

    filtered = billing.copy()

    if DATE_COL and not filtered.empty:
        minimum = filtered[DATE_COL].min().date()
        maximum = filtered[DATE_COL].max().date()

        date_range = st.date_input(
            "Usage date range",
            value=(minimum, maximum),
            min_value=minimum,
            max_value=maximum,
        )

        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                filtered[DATE_COL].dt.date.between(
                    start_date, end_date
                )
            ]

    if "Service_Name" in filtered.columns:
        services = sorted(
            filtered["Service_Name"]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_services = st.multiselect(
            "Cloud services",
            services,
            default=services,
        )

        filtered = filtered[
            filtered["Service_Name"].astype(str).isin(
                selected_services
            )
        ]

    st.caption("Currency: INR (₹)")
    st.caption("Source: project billing dataset")

# --------------------------------------------------
# 6. PAGE HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">FINANCIAL OPERATIONS / CLOUD INTELLIGENCE</div>
        <h1>Cloud Cost Intelligence</h1>
        <div class="hero-subtitle">
            Cost visibility · Anomaly investigation · Forecasting ·
            Risk-aware optimization
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
    <strong>Research prototype:</strong> SLA compliance is not verified
    when contractual targets and operational evidence are unavailable.
    Recommendations require manual approval. No cloud resources are
    modified automatically. Savings are illustrative, not guaranteed.
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No billing records match the selected filters.")
    st.stop()

total_cost = filtered[COST_COL].sum()
record_count = len(filtered)
resource_count = (
    filtered["Resource_ID"].nunique()
    if "Resource_ID" in filtered.columns
    else record_count
)

daily_cost = None

if DATE_COL:
    daily_cost = (
        filtered.groupby(filtered[DATE_COL].dt.date)[COST_COL]
        .sum()
        .sort_index()
    )

    average_daily = (
        daily_cost.mean() if not daily_cost.empty else 0
    )
else:
    average_daily = 0

# --------------------------------------------------
# 7. EXECUTIVE OVERVIEW
# --------------------------------------------------

if page == "Executive Overview":

    st.subheader("Executive overview")
    st.caption("A consolidated view of cloud expenditure and optimization signals.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total expenditure", money(total_cost))
    c2.metric("Average daily cost", money(average_daily))
    c3.metric("Resources represented", f"{resource_count:,}")

    if not recommendations.empty:
        savings_col = "Illustrative_10pct_Savings_INR"

        savings = (
            pd.to_numeric(
                recommendations[savings_col],
                errors="coerce",
            ).fillna(0).sum()
            if savings_col in recommendations.columns
            else 0
        )

        c4.metric("Illustrative savings", money(savings))
    else:
        c4.metric("Illustrative savings", "N/A")

    left, right = st.columns([1.7, 1])

    with left:
        st.markdown("### Spending trend")

        if DATE_COL:
            trend = (
                filtered.groupby(
                    filtered[DATE_COL].dt.date
                )[COST_COL]
                .sum()
                .reset_index()
            )
            trend.columns = ["Date", "Cost"]

            fig = px.area(
                trend,
                x="Date",
                y="Cost",
                markers=True,
                color_discrete_sequence=[BLUE],
            )
            fig.update_traces(
                fillcolor="rgba(99,140,255,0.15)"
            )
            fig.update_yaxes(title="Cost (INR)")
            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )
        else:
            st.info("A usable date column is required for a spending trend.")

    with right:
        st.markdown("### Cost by service")

        if "Service_Name" in filtered.columns:
            service_cost = (
                filtered.groupby("Service_Name")[COST_COL]
                .sum()
                .sort_values(ascending=False)
                .head(8)
                .reset_index()
            )

            fig = px.pie(
                service_cost,
                names="Service_Name",
                values=COST_COL,
                hole=0.65,
                color_discrete_sequence=[
                    BLUE, CYAN, GREEN, ORANGE, RED,
                    "#9B8AFB", "#67B7E8", "#F18AB5",
                ],
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )

    st.markdown("### Optimization signals")

    if not recommendations.empty:
        a, b, c = st.columns(3)

        a.metric(
            "Recommendation records",
            f"{len(recommendations):,}",
        )

        if "Recommended_Action" in recommendations.columns:
            anomaly_count = recommendations[
                recommendations["Recommended_Action"]
                .astype(str)
                .str.contains("anomaly", case=False, na=False)
            ].shape[0]
        else:
            anomaly_count = 0

        b.metric("Anomaly investigations", f"{anomaly_count:,}")

        if "Recommended_Action" in recommendations.columns:
            rightsizing_count = recommendations[
                recommendations["Recommended_Action"]
                .astype(str)
                .str.contains("rightsizing", case=False, na=False)
            ].shape[0]
        else:
            rightsizing_count = 0

        c.metric("Rightsizing reviews", f"{rightsizing_count:,}")

        st.caption(
            "These are recommendation records, not necessarily unique resources."
        )
    else:
        st.info(
            "Optimization recommendations are unavailable. "
            "Run the optimization engine to generate them."
        )

# --------------------------------------------------
# 8. COST ANALYTICS
# --------------------------------------------------

elif page == "Cost Analytics":

    st.subheader("Cost analytics")

    left, right = st.columns(2)

    with left:
        st.markdown("### Service cost concentration")

        if "Service_Name" in filtered.columns:
            service_cost = (
                filtered.groupby("Service_Name")[COST_COL]
                .sum()
                .sort_values()
                .tail(15)
                .reset_index()
            )

            fig = px.bar(
                service_cost,
                x=COST_COL,
                y="Service_Name",
                orientation="h",
                color=COST_COL,
                color_continuous_scale=["#273B68", BLUE, CYAN],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(title="Total cost (INR)")
            fig.update_yaxes(title="")
            st.plotly_chart(
                style_chart(fig, 430),
                use_container_width=True,
            )

    with right:
        st.markdown("### Utilization versus cost")

        if (
            "CPU_Utilization" in filtered.columns
            and "Memory_Utilization" in filtered.columns
        ):
            scatter_data = filtered.copy()

            scatter_data["CPU_Utilization"] = pd.to_numeric(
                scatter_data["CPU_Utilization"],
                errors="coerce",
            )
            scatter_data["Memory_Utilization"] = pd.to_numeric(
                scatter_data["Memory_Utilization"],
                errors="coerce",
            )

            fig = px.scatter(
                scatter_data,
                x="CPU_Utilization",
                y="Memory_Utilization",
                size=COST_COL,
                color=COST_COL,
                hover_name=(
                    "Resource_ID"
                    if "Resource_ID" in scatter_data.columns
                    else None
                ),
                color_continuous_scale=["#273B68", BLUE, CYAN],
                labels={
                    "CPU_Utilization": "CPU utilization (%)",
                    "Memory_Utilization": "Memory utilization (%)",
                    COST_COL: "Cost (INR)",
                },
            )
            st.plotly_chart(
                style_chart(fig, 430),
                use_container_width=True,
            )
        else:
            st.info("CPU and memory columns are needed for this chart.")

    st.markdown("### Billing records")

    display_columns = [
        col for col in [
            "Resource_ID",
            "Service_Name",
            "Region_Zone",
            DATE_COL,
            "CPU_Utilization",
            "Memory_Utilization",
            COST_COL,
        ]
        if col and col in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered billing data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_cloud_billing.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# 9. FORECASTING
# --------------------------------------------------

elif page == "Forecasting":

    st.subheader("Cost forecasting")
    st.caption(
        "Forecast files are loaded when available. "
        "The chart does not generate a new prediction."
    )

    forecast_candidates = list(
        (DATA / "canonical_run").rglob("*.csv")
    ) if (DATA / "canonical_run").exists() else []

    forecast_candidates = [
        path for path in forecast_candidates
        if "forecast" in path.name.lower()
        and any(
            key in path.name.lower()
            for key in ["prediction", "future", "forecast"]
        )
    ]

    selected_forecast = None

    for path in forecast_candidates:
        try:
            candidate = pd.read_csv(path)
            date_columns = [
                col for col in candidate.columns
                if "date" in col.lower()
            ]
            prediction_columns = [
                col for col in candidate.columns
                if any(
                    key in col.lower()
                    for key in ["forecast", "prediction", "predicted"]
                )
            ]

            if date_columns and prediction_columns:
                selected_forecast = candidate
                break
        except Exception:
            continue

    if selected_forecast is not None:
        date_col = next(
            col for col in selected_forecast.columns
            if "date" in col.lower()
        )
        pred_col = next(
            col for col in selected_forecast.columns
            if any(
                key in col.lower()
                for key in ["forecast", "prediction", "predicted"]
            )
        )

        selected_forecast[date_col] = pd.to_datetime(
            selected_forecast[date_col],
            errors="coerce",
        )
        selected_forecast[pred_col] = pd.to_numeric(
            selected_forecast[pred_col],
            errors="coerce",
        )

        selected_forecast = selected_forecast.dropna(
            subset=[date_col, pred_col]
        )

        if not selected_forecast.empty:
            fig = px.line(
                selected_forecast,
                x=date_col,
                y=pred_col,
                markers=True,
                color_discrete_sequence=[CYAN],
                labels={
                    date_col: "Forecast date",
                    pred_col: "Predicted cost (INR)",
                },
            )

            st.plotly_chart(
                style_chart(fig, 430),
                use_container_width=True,
            )

            st.caption(
                "Confirm the forecast file's date range and units "
                "before interpreting the projection."
            )
        else:
            st.info("The detected forecast file has no usable rows.")
    else:
        st.info(
            "No compatible forecast CSV was found automatically. "
            "Your forecasting scripts and research figures remain "
            "available in the project. We can connect the exact "
            "forecast output once its filename and columns are confirmed."
        )

    st.markdown("### Historical daily expenditure")

    if daily_cost is not None and not daily_cost.empty:
        st.line_chart(daily_cost)

    st.warning(
        "The current forecasting experiment has substantial prediction "
        "error. Treat projections as planning estimates, not precise "
        "financial commitments. The existing 30-day projection is "
        "dated August 30 to September 28, 2024, so it is historical "
        "relative to the current date."
    )

# --------------------------------------------------
# 10. OPTIMIZATION CENTER
# --------------------------------------------------

elif page == "Optimization Center":

    st.subheader("Optimization center")

    if recommendations.empty:
        st.warning(
            "Recommendations not found. Run "
            "Optimization Engine/optimization_engine.py first."
        )
    else:
        rec = recommendations.copy()

        if "Priority" in rec.columns:
            priority_filter = st.multiselect(
                "Filter by priority",
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )
            rec = rec[rec["Priority"].isin(priority_filter)]

        if "Recommended_Action" in rec.columns:
            action_filter = st.multiselect(
                "Filter by recommended action",
                sorted(rec["Recommended_Action"].dropna().unique()),
            )

            if action_filter:
                rec = rec[
                    rec["Recommended_Action"].isin(action_filter)
                ]

        k1, k2, k3 = st.columns(3)

        k1.metric("Visible recommendations", len(rec))

        savings_column = "Illustrative_10pct_Savings_INR"

        savings = (
            pd.to_numeric(
                rec[savings_column],
                errors="coerce",
            ).fillna(0).sum()
            if savings_column in rec.columns
            else 0
        )

        k2.metric("Illustrative savings", money(savings))
        k3.metric("Automatic changes", "Disabled")

        preferred_columns = [
            "Resource_ID",
            "Service_Name",
            "Current_Cost_INR",
            "CPU_Utilization",
            "Memory_Utilization",
            "Recommended_Action",
            "Priority",
            "Reason",
            "Illustrative_10pct_Savings_INR",
            "SLA_Compliance_Status",
            "SLA_Evidence_Status",
            "Manual_Approval_Required",
        ]

        visible_columns = [
            col for col in preferred_columns if col in rec.columns
        ]

        st.dataframe(
            rec[visible_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download recommendations",
            data=rec.to_csv(index=False).encode("utf-8"),
            file_name="optimization_recommendations.csv",
            mime="text/csv",
        )

        st.markdown("### Decision safeguards")
        st.markdown(
            """
            - Investigate anomaly flags before proposing changes.
            - Verify contractual SLA targets and operational telemetry.
            - Review workload criticality, security, and compliance.
            - Validate cost estimates and pricing assumptions.
            - Obtain human approval before any resource modification.
            """
        )

# --------------------------------------------------
# 11. DATA AND METHODOLOGY
# --------------------------------------------------

elif page == "Data & Methodology":

    st.subheader("Data and methodology")

    st.markdown("### Data source")

    st.write(f"Billing dataset: `{billing_path}`")
    st.write(f"Billing rows currently displayed: {len(filtered):,}")
    st.write(f"Cost field used: `{COST_COL}`")
    st.write(f"Date field used: `{DATE_COL or 'Unavailable'}`")

    if recommendations_path:
        st.write(f"Recommendation dataset: `{recommendations_path}`")
    else:
        st.write("Recommendation dataset: Not found")

    st.markdown("### Pipeline components")

    components = [
        ("Data preparation", "Standardization, date parsing and chronological splitting"),
        ("Feature engineering", "Cost, duration, utilization and benchmark features"),
        ("Anomaly detection", "Statistical and contextual investigation flags"),
        ("Forecasting", "Held-out evaluation against a baseline"),
        ("SLA assessment", "Evidence-aware risk screening and decision safeguards"),
        ("Optimization", "Advisory recommendations and illustrative savings scenarios"),
    ]

    st.dataframe(
        pd.DataFrame(
            components,
            columns=["Component", "Purpose"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Forecast evaluation")

    st.write("Held-out MAE: ₹902,218.80")
    st.write("Held-out RMSE: ₹1,180,874.39")
    st.write("Held-out WAPE: 40.71%")
    st.write("Baseline WAPE: 49.09%")

    st.caption(
        "These are the recorded experiment results. Confirm the evaluation "
        "artifacts and modeling assumptions before presenting them as final."
    )

    st.markdown("### Important limitations")

    st.markdown(
        """
        - SLA compliance cannot be certified from the current evidence.
        - Anomaly flags are not proof of billing errors.
        - The savings scenario is not realized or guaranteed savings.
        - Historical forecast dates must not be presented as current forecasts.
        - Recommendations are advisory and never automatically modify resources.
        """
    )

# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FINOPT · FinTech Cloud Cost Optimization Research Prototype · "
    "INR currency · Manual approval required"
)
