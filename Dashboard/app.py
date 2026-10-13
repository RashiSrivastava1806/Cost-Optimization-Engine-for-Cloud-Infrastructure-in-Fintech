
from pathlib import Path
import io
import html
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FinOpt | Cloud Cost Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent

# Your requested palette
COLORS = {
    "teal": "#72B0AB",
    "mint": "#BCDDDC",
    "cream": "#FFEDD1",
    "peach": "#FDC1B4",
    "coral": "#FE9179",
    "sand": "#CFB97E",
    "gold": "#B89D47",
    "forest": "#355E58",
    "deep": "#053229",
}

PLOT_COLORS = [
    COLORS["teal"],
    COLORS["coral"],
    COLORS["sand"],
    COLORS["forest"],
    COLORS["gold"],
    COLORS["peach"],
    COLORS["mint"],
]

# =========================================================
# 2. APPLICATION STYLING
# =========================================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {COLORS["cream"]};
        color: {COLORS["deep"]};
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: {COLORS["deep"]};
    }}

    [data-testid="stSidebar"] * {{
        color: {COLORS["cream"]};
    }}

    h1, h2, h3 {{
        color: {COLORS["deep"]} !important;
        letter-spacing: -0.5px;
    }}

    p, label, [data-testid="stCaptionContainer"] {{
        color: {COLORS["forest"]};
    }}

    [data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid {COLORS["mint"]};
        border-radius: 14px;
        padding: 17px;
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS["forest"]};
    }}

    [data-testid="stMetricValue"] {{
        color: {COLORS["deep"]};
    }}

    .hero {{
        background: linear-gradient(
            115deg,
            {COLORS["deep"]},
            {COLORS["forest"]}
        );
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 20px;
        color: {COLORS["cream"]};
    }}

    .hero h1 {{
        color: {COLORS["cream"]} !important;
        font-size: 36px;
        margin: 10px 0;
    }}

    .hero p {{
        color: {COLORS["mint"]};
        margin-bottom: 0;
    }}

    .eyebrow {{
        color: {COLORS["teal"]};
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 700;
    }}

    .notice {{
        background: #FFFFFF;
        border-left: 5px solid {COLORS["teal"]};
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0 20px 0;
    }}

    .warning {{
        background: {COLORS["peach"]};
        border-left: 5px solid {COLORS["coral"]};
        border-radius: 8px;
        padding: 15px;
        margin: 12px 0;
    }}

    div.stButton > button,
    div.stDownloadButton > button {{
        background: {COLORS["deep"]};
        color: #FFFFFF;
        border: 1px solid {COLORS["forest"]};
        border-radius: 8px;
    }}

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {{
        background: {COLORS["forest"]};
        color: #FFFFFF;
        border-color: {COLORS["teal"]};
    }}

    div[data-testid="stTabs"] button {{
        color: {COLORS["deep"]};
    }}

    hr {{
        border-color: {COLORS["mint"]};
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. GENERAL HELPERS
# =========================================================

def money(value):
    if pd.isna(value):
        return "N/A"
    return f"₹{value:,.2f}"


def read_uploaded_file(uploaded_file):
    """Read CSV or Excel without modifying the uploaded file."""
    if uploaded_file is None:
        return None

    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        except Exception:
            return pd.read_csv(io.BytesIO(raw), encoding_errors="replace")

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))

    raise ValueError("Upload a CSV or Excel file.")


def clean_numeric(series):
    """Convert common currency-formatted values to numbers."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("INR", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def find_column(columns, keywords):
    """Suggest a likely matching column without silently assuming it."""
    for column in columns:
        normalized = str(column).strip().lower().replace(" ", "_")
        if normalized in keywords:
            return column
    return None


def style_chart(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color=COLORS["deep"], family="Arial"),
        margin=dict(l=10, r=10, t=45, b=10),
        colorway=PLOT_COLORS,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=COLORS["mint"],
    )
    fig.update_yaxes(
        gridcolor=COLORS["mint"],
        zeroline=False,
    )
    return fig


def download_csv(df, filename, label):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# =========================================================
# 4. HEADER AND NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("## FINOPT")
    st.caption("CLOUD COST INTELLIGENCE")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Cost Analytics",
            "Optimization",
            "SLA Evidence",
            "Export Report",
        ],
    )

    st.divider()
    st.caption("Client data is processed for this session.")
    st.caption("No cloud resources are modified automatically.")

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">FINANCIAL OPERATIONS / CLOUD INTELLIGENCE</div>
        <h1>Cloud Cost Intelligence</h1>
        <p>
            Understand spending. Identify opportunities.
            Make evidence-based optimization decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 5. CLIENT FILE UPLOAD
# =========================================================

with st.expander("Upload billing data", expanded=True):
    st.write(
        "Upload a cloud billing export in CSV or Excel format. "
        "You will map its columns before analysis."
    )

    billing_file = st.file_uploader(
        "Cloud billing sheet",
        type=["csv", "xlsx", "xls"],
        key="billing_upload",
    )

if billing_file is None:
    st.markdown(
        """
        <div class="notice">
        <strong>Welcome to FinOpt.</strong><br>
        Upload your billing sheet to begin. Cost analysis requires a
        cost column. Additional insights become available when resource,
        date, service, region and utilization data are supplied.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Recommended fields: resource ID, service, region, usage date, "
        "total cost, CPU utilization and memory utilization."
    )

    st.stop()

try:
    raw_df = read_uploaded_file(billing_file)
except Exception as error:
    st.error(f"Could not read this file: {error}")
    st.stop()

if raw_df is None or raw_df.empty:
    st.error("The uploaded file is empty or could not be parsed.")
    st.stop()

raw_df.columns = [
    str(column).strip() for column in raw_df.columns
]

if raw_df.columns.duplicated().any():
    st.error(
        "The file contains duplicate column names. Rename duplicate "
        "headers and upload the file again."
    )
    st.stop()

columns = list(raw_df.columns)

# =========================================================
# 6. COLUMN MAPPING
# =========================================================

st.subheader("Map your billing columns")
st.caption(
    "Suggested matches are provided for convenience. "
    "Please verify them before continuing."
)

def select_column(label, keywords, required=False):
    suggestion = find_column(columns, keywords)

    options = ["— Not available —"] + columns

    default_index = (
        options.index(suggestion)
        if suggestion in columns
        else 0
    )

    chosen = st.selectbox(
        label + (" *" if required else ""),
        options,
        index=default_index,
        key=f"mapping_{label}",
    )

    return None if chosen == "— Not available —" else chosen


with st.form("column_mapping_form"):
    m1, m2, m3 = st.columns(3)

    with m1:
        cost_col = select_column(
            "Total cost",
            [
                "total_cost_inr", "total_cost", "cost",
                "amount", "billed_cost", "unrounded_cost",
                "rounded_cost", "cost_inr", "line_item_unblended_cost",
            ],
            required=True,
        )

        date_col = select_column(
            "Usage date",
            [
                "usage_start_date", "date", "usage_date",
                "billing_date", "usage_start_time",
                "date_time", "usagestartdate",
            ],
        )

    with m2:
        resource_col = select_column(
            "Resource identifier",
            [
                "resource_id", "resourceid", "resource_name",
                "instance_id", "asset_id",
            ],
        )

        service_col = select_column(
            "Service",
            [
                "service_name", "service", "product_name",
                "service_type", "meter_category",
            ],
        )

    with m3:
        region_col = select_column(
            "Region",
            [
                "region_zone", "region", "availability_zone",
                "location", "region_name",
            ],
        )

        cpu_col = select_column(
            "CPU utilization",
            [
                "cpu_utilization", "cpu_utilization_percent",
                "cpu_percent", "cpu_usage_percent",
            ],
        )

        memory_col = select_column(
            "Memory utilization",
            [
                "memory_utilization", "memory_utilization_percent",
                "memory_percent", "memory_usage_percent",
            ],
        )

    currency_col = select_column(
        "Currency column (optional)",
        ["currency", "currency_code", "billing_currency"],
    )

    submitted = st.form_submit_button("Apply mapping and analyze")

if not submitted:
    st.info("Verify the mappings and select **Apply mapping and analyze**.")
    st.stop()

if cost_col is None:
    st.error("A total cost column is required.")
    st.stop()

# =========================================================
# 7. NORMALIZE THE CLIENT DATA
# =========================================================

df = pd.DataFrame(index=raw_df.index)

df["Cost"] = clean_numeric(raw_df[cost_col])

if date_col:
    df["Date"] = pd.to_datetime(
        raw_df[date_col],
        errors="coerce",
        dayfirst=False,
        utc=True,
    ).dt.tz_convert(None)

if resource_col:
    df["Resource"] = raw_df[resource_col].astype("string")

if service_col:
    df["Service"] = raw_df[service_col].astype("string")

if region_col:
    df["Region"] = raw_df[region_col].astype("string")

if cpu_col:
    df["CPU"] = clean_numeric(raw_df[cpu_col])

if memory_col:
    df["Memory"] = clean_numeric(raw_df[memory_col])

if currency_col:
    df["Currency"] = raw_df[currency_col].astype("string")

valid_cost = df["Cost"].notna()
negative_cost = df["Cost"] < 0

invalid_count = int((~valid_cost).sum())
negative_count = int(negative_cost.sum())

df = df.loc[valid_cost & ~negative_cost].copy()

if df.empty:
    st.error(
        "No valid non-negative cost records were found. "
        "Check your cost column mapping and numeric format."
    )
    st.stop()

if "Currency" in df.columns:
    currencies = df["Currency"].dropna().unique().tolist()
    if len(currencies) > 1:
        st.warning(
            "Multiple currencies were detected. Costs are not "
            "converted into a common currency. Filter or convert "
            "them before interpreting totals."
        )
    elif len(currencies) == 1 and str(currencies[0]).upper() not in [
        "INR", "RS", "₹"
    ]:
        st.warning(
            f"The uploaded file reports currency as {currencies[0]}. "
            "Values will be shown in the uploaded currency, not INR."
        )

st.markdown(
    f"""
    <div class="notice">
        <strong>File loaded:</strong>
        {html.escape(billing_file.name)}<br>
        Valid cost records: {len(df):,} |
        Rows without valid costs: {invalid_count:,} |
        Negative-cost rows excluded: {negative_count:,}
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 8. FILTERS
# =========================================================

st.subheader("Analysis filters")

filtered = df.copy()

f1, f2, f3 = st.columns(3)

with f1:
    if "Date" in filtered.columns and filtered["Date"].notna().any():
        min_date = filtered["Date"].min().date()
        max_date = filtered["Date"].max().date()

        date_range = st.date_input(
            "Usage period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                filtered["Date"].dt.date.between(start_date, end_date)
            ]

with f2:
    if "Service" in filtered.columns:
        service_options = sorted(
            filtered["Service"].dropna().astype(str).unique().tolist()
        )
        selected_services = st.multiselect(
            "Services",
            service_options,
            default=service_options,
        )
        filtered = filtered[
            filtered["Service"].astype(str).isin(selected_services)
        ]

with f3:
    if "Region" in filtered.columns:
        region_options = sorted(
            filtered["Region"].dropna().astype(str).unique().tolist()
        )
        selected_regions = st.multiselect(
            "Regions",
            region_options,
            default=region_options,
        )
        filtered = filtered[
            filtered["Region"].astype(str).isin(selected_regions)
        ]

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

total_cost = float(filtered["Cost"].sum())

# =========================================================
# 9. BASIC OPTIMIZATION SIGNALS
# =========================================================

has_cpu_memory = (
    "CPU" in filtered.columns
    and "Memory" in filtered.columns
)

if has_cpu_memory:
    low_utilization = (
        filtered["CPU"].between(0, 100)
        & filtered["Memory"].between(0, 100)
        & (filtered["CPU"] < 30)
        & (filtered["Memory"] < 30)
    )
else:
    low_utilization = pd.Series(False, index=filtered.index)

resource_cost = (
    filtered.groupby("Resource")["Cost"].sum()
    if "Resource" in filtered.columns
    else pd.Series(dtype=float)
)

# =========================================================
# 10. OVERVIEW
# =========================================================

if page == "Overview":
    st.subheader("Executive overview")
    st.caption("A financial summary of the uploaded billing period.")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total expenditure", money(total_cost))
    k2.metric("Billing records", f"{len(filtered):,}")

    k3.metric(
        "Resources",
        f"{filtered['Resource'].nunique():,}"
        if "Resource" in filtered.columns
        else "Unavailable",
    )

    k4.metric(
        "Low-utilization candidates",
        f"{int(low_utilization.sum()):,}"
        if has_cpu_memory
        else "Data needed",
    )

    left, right = st.columns([1.7, 1])

    with left:
        st.markdown("### Spending over time")

        if "Date" in filtered.columns and filtered["Date"].notna().any():
            trend = (
                filtered.dropna(subset=["Date"])
                .groupby(filtered["Date"].dt.date)["Cost"]
                .sum()
                .reset_index()
            )
            trend.columns = ["Date", "Cost"]

            fig = px.area(
                trend,
                x="Date",
                y="Cost",
                markers=True,
                color_discrete_sequence=[COLORS["teal"]],
            )

            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )
        else:
            st.info("Upload or map a valid usage date column for trend analysis.")

    with right:
        st.markdown("### Cost distribution")

        category = (
            "Service"
            if "Service" in filtered.columns
            else "Region"
            if "Region" in filtered.columns
            else "Resource"
            if "Resource" in filtered.columns
            else None
        )

        if category:
            breakdown = (
                filtered.groupby(category, dropna=False)["Cost"]
                .sum()
                .nlargest(8)
                .reset_index()
            )

            fig = px.pie(
                breakdown,
                names=category,
                values="Cost",
                hole=0.62,
                color_discrete_sequence=PLOT_COLORS,
            )

            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )
        else:
            st.info("Map a service, region or resource column to show a breakdown.")

    st.markdown("### Highest-cost resources")

    if "Resource" in filtered.columns:
        top_resources = (
            filtered.groupby("Resource")["Cost"]
            .sum()
            .nlargest(10)
            .reset_index()
        )

        fig = px.bar(
            top_resources.sort_values("Cost"),
            x="Cost",
            y="Resource",
            orientation="h",
            color_discrete_sequence=[COLORS["forest"]],
        )

        st.plotly_chart(
            style_chart(fig, 360),
            use_container_width=True,
        )
    else:
        st.info("Map a resource identifier to rank individual resources.")

# =========================================================
# 11. COST ANALYTICS
# =========================================================

elif page == "Cost Analytics":
    st.subheader("Cost analytics")

    left, right = st.columns(2)

    with left:
        st.markdown("### Spending by service")

        if "Service" in filtered.columns:
            by_service = (
                filtered.groupby("Service")["Cost"]
                .sum()
                .nlargest(15)
                .sort_values()
                .reset_index()
            )

            fig = px.bar(
                by_service,
                x="Cost",
                y="Service",
                orientation="h",
                color_discrete_sequence=[COLORS["teal"]],
            )

            st.plotly_chart(
                style_chart(fig, 430),
                use_container_width=True,
            )
        else:
            st.info("Service data is unavailable.")

    with right:
        st.markdown("### Spending by region")

        if "Region" in filtered.columns:
            by_region = (
                filtered.groupby("Region")["Cost"]
                .sum()
                .nlargest(15)
                .reset_index()
            )

            fig = px.pie(
                by_region,
                names="Region",
                values="Cost",
                hole=0.55,
                color_discrete_sequence=PLOT_COLORS,
            )

            st.plotly_chart(
                style_chart(fig, 430),
                use_container_width=True,
            )
        else:
            st.info("Region data is unavailable.")

    st.markdown("### Cost versus utilization")

    if has_cpu_memory:
        scatter = filtered.copy()

        valid_cpu_memory = (
            scatter["CPU"].between(0, 100)
            & scatter["Memory"].between(0, 100)
        )

        scatter = scatter.loc[valid_cpu_memory]

        if not scatter.empty:
            fig = px.scatter(
                scatter,
                x="CPU",
                y="Memory",
                size="Cost",
                color="Cost",
                hover_name=(
                    "Resource"
                    if "Resource" in scatter.columns
                    else None
                ),
                color_continuous_scale=[
                    COLORS["mint"],
                    COLORS["teal"],
                    COLORS["forest"],
                ],
                labels={
                    "CPU": "CPU utilization (%)",
                    "Memory": "Memory utilization (%)",
                    "Cost": "Cost",
                },
            )

            st.plotly_chart(
                style_chart(fig, 430),
                use_container_width=True,
            )
        else:
            st.info("No valid CPU and memory measurements are available.")
    else:
        st.info(
            "Map CPU and memory columns to analyze utilization. "
            "Cost data alone cannot establish underutilization."
        )

    st.markdown("### Billing records")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    download_csv(
        filtered,
        "filtered_billing.csv",
        "Download filtered billing records",
    )

# =========================================================
# 12. OPTIMIZATION
# =========================================================

elif page == "Optimization":
    st.subheader("Optimization opportunities")

    st.markdown(
        """
        Recommendations below are screening signals. They are not
        automatic instructions to modify production resources.
        """
    )

    if not has_cpu_memory:
        st.warning(
            "CPU and memory measurements are missing. "
            "Upload or map both fields to assess low-utilization candidates."
        )

    opportunities = filtered.loc[low_utilization].copy()

    if has_cpu_memory and not opportunities.empty:
        opportunities["Screening_Reason"] = (
            "CPU and memory are both below 30%; verify workload demand, "
            "performance and SLA requirements before rightsizing."
        )

        # A scenario, not a forecast or guaranteed saving.
        opportunities["Illustrative_10pct_Savings"] = (
            opportunities["Cost"] * 0.10
        )

        # Group by resource if possible, to avoid listing the same
        # resource repeatedly as if each row were a distinct resource.
        if "Resource" in opportunities.columns:
            opportunities = (
                opportunities.groupby("Resource", dropna=False)
                .agg(
                    Current_Cost=("Cost", "sum"),
                    CPU=("CPU", "mean"),
                    Memory=("Memory", "mean"),
                    Screening_Reason=("Screening_Reason", "first"),
                    Illustrative_10pct_Savings=(
                        "Illustrative_10pct_Savings", "sum"
                    ),
                )
                .reset_index()
            )
        else:
            opportunities = opportunities.rename(
                columns={
                    "Cost": "Current_Cost",
                    "CPU": "CPU",
                    "Memory": "Memory",
                }
            )

        a, b = st.columns(2)

        a.metric(
            "Resources for review",
            f"{len(opportunities):,}",
        )

        b.metric(
            "Illustrative 10% scenario",
            money(opportunities["Illustrative_10pct_Savings"].sum()),
        )

        st.dataframe(
            opportunities,
            use_container_width=True,
            hide_index=True,
        )

        download_csv(
            opportunities,
            "optimization_opportunities.csv",
            "Download optimization opportunities",
        )
    else:
        st.info(
            "No resources currently qualify for this screening rule, "
            "or the required utilization data is unavailable."
        )

    st.markdown("### Anomalous spending")

    st.caption(
        "This is a simple statistical screening method, not a trained "
        "production anomaly model. Investigate flags before taking action."
    )

    if len(filtered) >= 5:
        q1 = filtered["Cost"].quantile(0.25)
        q3 = filtered["Cost"].quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            threshold = q3 + 1.5 * iqr
            anomalies = filtered[filtered["Cost"] > threshold].copy()

            st.metric("Records flagged for review", len(anomalies))
            st.write(f"Screening threshold: {money(threshold)}")

            if not anomalies.empty:
                st.dataframe(
                    anomalies,
                    use_container_width=True,
                    hide_index=True,
                )

                download_csv(
                    anomalies,
                    "billing_anomalies_for_review.csv",
                    "Download flagged records",
                )
        else:
            st.info(
                "The interquartile range is zero. "
                "This screening rule cannot distinguish unusual records."
            )
    else:
        st.info("At least five valid cost records are required for this screen.")

    st.markdown(
        """
        <div class="warning">
        <strong>Before approving a recommendation:</strong>
        verify the service owner, resource criticality, contractual SLA,
        monitoring evidence, workload demand, security and compliance.
        Every change requires human approval.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 13. SLA EVIDENCE WORKSPACE
# =========================================================

elif page == "SLA Evidence":
    st.subheader("SLA evidence verification")
    st.write(
        "Upload contractual targets and operational measurements separately. "
        "Billing records alone cannot prove SLA compliance."
    )

    st.markdown("### 1. Upload SLA targets")

    targets_file = st.file_uploader(
        "SLA targets CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="sla_targets",
    )

    st.caption(
        "Suggested fields: Service_ID, Metric, Operator, Target_Value, "
        "Unit, Measurement_Window, Valid_From, Valid_To."
    )

    st.markdown("### 2. Upload actual measurements")

    metrics_file = st.file_uploader(
        "Monitoring metrics CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="sla_metrics",
    )

    st.caption(
        "Suggested fields: Service_ID, Metric, Timestamp, "
        "Observed_Value, Unit, Source."
    )

    if targets_file is None or metrics_file is None:
        st.info(
            "Upload both files to compare actual measurements with targets. "
            "Until then, SLA status remains NOT_VERIFIABLE."
        )
    else:
        try:
            targets = read_uploaded_file(targets_file)
            metrics = read_uploaded_file(metrics_file)
        except Exception as error:
            st.error(f"Could not read SLA evidence files: {error}")
            st.stop()

        if targets.empty or metrics.empty:
            st.error("One or both SLA evidence files are empty.")
            st.stop()

        st.markdown("### Preview target data")
        st.dataframe(targets.head(10), use_container_width=True)

        st.markdown("### Preview monitoring data")
        st.dataframe(metrics.head(10), use_container_width=True)

        required_targets = {
            "Service_ID", "Metric", "Operator",
            "Target_Value", "Unit",
        }

        required_metrics = {
            "Service_ID", "Metric", "Timestamp",
            "Observed_Value", "Unit",
        }

        missing_targets = required_targets - set(targets.columns)
        missing_metrics = required_metrics - set(metrics.columns)

        if missing_targets or missing_metrics:
            st.warning(
                "The evidence files do not match the expected schema. "
                "Rename or map the required columns before verification."
            )

            if missing_targets:
                st.write("Missing target columns:", sorted(missing_targets))

            if missing_metrics:
                st.write("Missing measurement columns:", sorted(missing_metrics))
        else:
            targets = targets.copy()
            metrics = metrics.copy()

            targets["Target_Value"] = pd.to_numeric(
                targets["Target_Value"],
                errors="coerce",
            )

            metrics["Observed_Value"] = pd.to_numeric(
                metrics["Observed_Value"],
                errors="coerce",
            )

            metrics["Timestamp"] = pd.to_datetime(
                metrics["Timestamp"],
                errors="coerce",
                utc=True,
            )

            targets["Service_ID"] = (
                targets["Service_ID"].astype(str).str.strip()
            )
            metrics["Service_ID"] = (
                metrics["Service_ID"].astype(str).str.strip()
            )

            targets["Metric"] = (
                targets["Metric"].astype(str).str.strip().str.lower()
            )
            metrics["Metric"] = (
                metrics["Metric"].astype(str).str.strip().str.lower()
            )

            targets["Unit"] = (
                targets["Unit"].astype(str).str.strip().str.lower()
            )
            metrics["Unit"] = (
                metrics["Unit"].astype(str).str.strip().str.lower()
            )

            metrics["Measurement_Window"] = (
                metrics["Measurement_Window"].astype(str).str.strip().str.lower()
                if "Measurement_Window" in metrics.columns
                else ""
            )

            results = []

            for _, target in targets.iterrows():
                service = target["Service_ID"]
                metric = target["Metric"]
                unit = target["Unit"]
                operator = str(target["Operator"]).strip()
                target_value = target["Target_Value"]

                matching = metrics[
                    (metrics["Service_ID"] == service)
                    & (metrics["Metric"] == metric)
                    & (metrics["Unit"] == unit)
                ].copy()

                if pd.isna(target_value):
                    status = "EVIDENCE_INSUFFICIENT"
                    observed = None
                    reason = "Target value is missing or invalid."

                elif matching.empty:
                    status = "NOT_VERIFIABLE"
                    observed = None
                    reason = "No matching operational measurements."

                else:
                    # This MVP only evaluates a simple numeric metric.
                    # It does not certify availability or contractual SLA
                    # compliance from arbitrary point measurements.
                    observed = matching["Observed_Value"].dropna()

                    if observed.empty:
                        status = "EVIDENCE_INSUFFICIENT"
                        observed = None
                        reason = "Matching measurements contain no valid values."

                    elif metric in {
                        "availability",
                        "availability_percent",
                        "uptime",
                    }:
                        status = "NOT_VERIFIABLE"
                        observed = float(observed.mean())
                        reason = (
                            "Availability requires a complete measurement "
                            "window and contract-specific calculation. "
                            "A simple average of point measurements is insufficient."
                        )

                    else:
                        observed = float(observed.mean())

                        # Verify simple supported numeric operators.
                        if operator == ">=":
                            passed = observed >= target_value
                        elif operator == ">":
                            passed = observed > target_value
                        elif operator == "<=":
                            passed = observed <= target_value
                        elif operator == "<":
                            passed = observed < target_value
                        elif operator == "==":
                            passed = observed == target_value
                        else:
                            passed = None

                        if passed is None:
                            status = "EVIDENCE_INSUFFICIENT"
                            reason = f"Unsupported operator: {operator}"
                        else:
                            status = (
                                "TARGET_MET"
                                if passed
                                else "TARGET_NOT_MET"
                            )
                            reason = (
                                "Numeric comparison performed. "
                                "Confirm reporting window, coverage, "
                                "source authenticity and contract applicability."
                            )

                results.append({
                    "Service_ID": service,
                    "Metric": metric,
                    "Operator": operator,
                    "Target_Value": target_value,
                    "Observed_Summary": observed,
                    "Unit": unit,
                    "Status": status,
                    "Interpretation": reason,
                    "Automatic_Approval_Allowed": False,
                })

            result_df = pd.DataFrame(results)

            st.markdown("### Evidence assessment")

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )

            st.warning(
                "This is a preliminary evidence screening tool, not a "
                "contractual SLA certification. It does not yet validate "
                "complete measurement windows, contractual exclusions, "
                "sampling coverage, or monitoring-source authenticity. "
                "Availability is deliberately left NOT_VERIFIABLE by this "
                "simple evaluator."
            )

            download_csv(
                result_df,
                "sla_evidence_assessment.csv",
                "Download SLA evidence assessment",
            )

# =========================================================
# 14. EXPORT REPORT
# =========================================================

elif page == "Export Report":
    st.subheader("Export client report")

    st.write(
        "Download the filtered billing data and analysis outputs "
        "for offline review."
    )

    report_summary = pd.DataFrame([{
        "Billing_File": billing_file.name,
        "Records_Analyzed": len(filtered),
        "Total_Cost": total_cost,
        "Currency": (
            ", ".join(filtered["Currency"].dropna().astype(str).unique())
            if "Currency" in filtered.columns
            else "As supplied; not identified"
        ),
        "Resources": (
            filtered["Resource"].nunique()
            if "Resource" in filtered.columns
            else None
        ),
        "Date_From": (
            str(filtered["Date"].min())
            if "Date" in filtered.columns
            else None
        ),
        "Date_To": (
            str(filtered["Date"].max())
            if "Date" in filtered.columns
            else None
        ),
        "SLA_Certification": "Not performed by billing analysis",
        "Automatic_Modification": "Disabled",
    }])

    st.dataframe(
        report_summary,
        use_container_width=True,
        hide_index=True,
    )

    download_csv(
        report_summary,
        "client_cost_summary.csv",
        "Download executive summary",
    )

    download_csv(
        filtered,
        "client_filtered_billing.csv",
        "Download analyzed billing data",
    )

    st.info(
        "For privacy, avoid uploading credentials, secrets or unrelated "
        "personal information. This prototype processes uploaded files "
        "in the current app session; it is not a complete production "
        "security or data-retention system."
    )

# =========================================================
# 15. FOOTER
# =========================================================

st.divider()

st.caption(
    "FINOPT | Cloud Cost Intelligence | "
    "Advisory analytics only | Human approval required"
)
