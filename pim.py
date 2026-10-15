# ============================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================
import re
from io import BytesIO
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import pandas as pd
import plotly.express as px
import streamlit as st
 
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISIT_PREFIX = "Total Number of Visits Per Month"
STANDARD_PREFIX = "Meeting Minimum Standards By Grade?"
PRIORITY_PREFIX = "Teacher's Priority Area (0, 1, 2, or 3)"
PRIORITY_STUB = "Teacher's Priority Area (0, 1, 2, or 3)"
 
# Original district/year/grade targets from the supplied application.
VISIT_TARGET_RULES = {
    ("Jhalakathi", 1, 1): 2,
    ("Jhalakathi", 2, 1): 2, 
    ("Jhalakathi", 2, 2): 2,
    ("Jhalakathi", 3, 1): 2, 
    ("Jhalakathi", 3, 2): 2,
    ("Jhalakathi", 4, 1): 2, 
    ("Jhalakathi", 4, 2): 2,
    ("Jamalpur", 1, 1): 1, 
    ("Jamalpur", 1, 2): 1,
    ("Jamalpur", 2, 1): 1, 
    ("Jamalpur", 2, 2): 1,
    ("Jamalpur", 3, 1): 1, 
    ("Jamalpur", 3, 2): 1,
    ("Jamalpur", 4, 1): 1, 
    ("Jamalpur", 4, 2): 1,
    ("Moulvibazar", 1, 1): 2,
    ("Moulvibazar", 2, 1): 2, 
    ("Moulvibazar", 2, 2): 2,
    ("Moulvibazar", 3, 1): 1, 
    ("Moulvibazar", 3, 2): 2,
    ("Moulvibazar", 4, 1): 1, 
    ("Moulvibazar", 4, 2): 1,
    ("Narail", 1, 1): 1, 
    ("Narail", 1, 2): 1,
    ("Narail", 2, 1): 1, 
    ("Narail", 2, 2): 1, 
    ("Narail", 2, 3): 1,
    ("Narail", 3, 1): 1, 
    ("Narail", 3, 2): 1, 
    ("Narail", 3, 3): 1,
}
 
# ============================================================
# 2. DATA PROCESSING FUNCTIONS
# ============================================================
def data_clean(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build column names from the two header rows and clean field values."""
    if raw_df is None or raw_df.empty or len(raw_df) < 2:
        raise ValueError("The selected worksheet does not contain the two required header rows.")
 
    df = raw_df.copy()
    main_header = df.iloc[0].ffill()
    month_header = df.iloc[1]
    columns = []
 
    for main, month in zip(main_header, month_header):
        main_text = str(main).strip() if pd.notna(main) else "Unnamed"
        if pd.notna(month) and str(month).strip():
            columns.append(f"{main_text}_{str(month).strip()}")
        else:
            columns.append(main_text)
 
    df.columns = columns
    df = df.iloc[2:].reset_index(drop=True)
 
    def normalize_month_column(column):
        if "_" not in str(column):
            return str(column).strip()
        prefix, suffix = str(column).rsplit("_", 1)
        try:
            parsed = pd.to_datetime(suffix, errors="raise")
            return f"{prefix}_{parsed.strftime('%b')}"
        except (ValueError, TypeError, OverflowError):
            # Also support headers already containing month names.
            month = suffix[:3].title()
            return f"{prefix}_{month}" if month in MONTH_ORDER else str(column).strip()
 
    df.columns = [normalize_month_column(col) for col in df.columns]
    df = df.dropna(axis=1, how="all")
 
    if "School Name" not in df.columns:
        raise ValueError("Could not find a 'School Name' column. Please check the worksheet and header rows.")
    df = df.dropna(subset=["School Name"])
    if "S/N" in df.columns:
        df = df.drop(columns=["S/N"])
 
    standard_cols = [c for c in df.columns if c.startswith(STANDARD_PREFIX)]
    priority_cols = [c for c in df.columns if c.startswith(PRIORITY_PREFIX)]
    visit_cols = [c for c in df.columns if c.startswith(VISIT_PREFIX)]
 
    yes_no = {"No": 0, "Yes": 1, 0: 0, 1: 1}
    priority_map = {
        "0: No Priority Areas Achieved": 0,
        "1: Mastered Instructional Routine": 1,
        "2: Mastered Basic Skills": 2,
        "3: Mastered Advanced Skills": 3,
        0: 0, 1: 1, 2: 2, 3: 3,
    }
    for col in standard_cols:
        df[col] = df[col].map(yes_no)
    for col in priority_cols:
        df[col] = df[col].map(priority_map)
    for col in visit_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
 
    # Normalize numeric identifier columns where Excel may have stored numbers as text.
    for col in ["Year of Support", "Grade"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    return df.reset_index(drop=True)
 
 
def df_long(df: pd.DataFrame) -> pd.DataFrame:
    stubs = [STANDARD_PREFIX, VISIT_PREFIX, PRIORITY_STUB]
    id_columns = ["Field Office", "District", "RtR Staff Name", "Project ID", "School Name","Teacher Name", "Year of Support", "Grade", "Class",]
 
    missing = [c for c in id_columns if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns for monthly analysis: " + ", ".join(missing))
    available_stubs = [stub for stub in stubs if any(c.startswith(stub + "_") for c in df.columns)]
    if not available_stubs:
        raise ValueError("No monthly visit, standards, or priority columns were found.")
 
    df_long = pd.wide_to_long(df.copy(), 
                              stubnames=available_stubs, 
                              i=id_columns, 
                              j="Month", 
                              sep="_", suffix="[A-Za-z]+").reset_index()
    visit_column = f"{VISIT_PREFIX}"
    if visit_column not in df_long.columns:
        raise ValueError(f"Expected visit column '{visit_column}' was not found.")
    # Preserve the actual monthly standards values when present. Only use the
    # visit-count rule as a fallback when the workbook has no standards columns.
    standards_column = STANDARD_PREFIX
    if standards_column not in df_long.columns:
        df_long[standards_column] = np.where(
            pd.to_numeric(df_long[visit_column], errors="coerce") >= 2, 1, 0
        )
    else:
        df_long[standards_column] = pd.to_numeric(
            df_long[standards_column], errors="coerce"
        )
    return df_long
 
#========================================================================================================================================================
#========================================================================================================================================================
 
 
def calculate_dashboard_metrics(df: pd.DataFrame) -> dict:
    visit_cols = [c for c in df.columns if c.startswith(VISIT_PREFIX)]
    standard_cols = [c for c in df.columns if c.startswith(STANDARD_PREFIX)]
    priority_cols = [c for c in df.columns if c.startswith(PRIORITY_PREFIX)]

    total_staff = df["RtR Staff Name"].nunique()
    total_school = df["Project ID"].nunique()
    total_teachers = df["Teacher Name"].nunique()
    total_visits = float(df[visit_cols].sum().sum())
    standard_rate = round(df[standard_cols].stack().value_counts().loc[1]/(df[standard_cols].stack().value_counts().loc[0]+df[standard_cols].stack().value_counts().loc[1])*100,1)
    avg_priority = float((round(df[priority_cols].stack().mean(),1)))
    metrics = {"total_staff" : total_staff,
               "total_schools": total_school,
               "total_teachers": total_teachers,
               "total_visits" : total_visits,
               "standard_rate" : standard_rate,
               "avg_priority" :  avg_priority}
    return metrics 
 
def monthly_visit_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(VISIT_PREFIX)]
    if not cols:
        return pd.DataFrame({"Month": [], "Total_Visit": []})
    summary = df[cols].sum().rename_axis("Month").reset_index(name="Total_Visit")
    summary["Month"] = summary["Month"].str.replace(f"^{re.escape(VISIT_PREFIX)}_", "", regex=True)
    summary["Month"] = pd.Categorical(summary["Month"], categories=MONTH_ORDER, ordered=True)
    return summary.sort_values("Month").reset_index(drop=True)
 
 
def create_monthly_visit_chart(df: pd.DataFrame):
    monthly = monthly_visit_summary(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    # Modern dark theme
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1F2937")
 
    # Line chart
    ax.plot(
        monthly["Month"],
        monthly["Total_Visit"],
        color="#38BDF8",
        marker="o",
        linewidth=3,
        markersize=8,
        markerfacecolor="#FFFFFF",
        markeredgecolor="#38BDF8",
        markeredgewidth=2,
        zorder=3
    )
 
    # Add value labels
    for x, y in zip(monthly["Month"], monthly["Total_Visit"]):
        ax.annotate(
            f"{y:,.0f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )
 
    # Titles and axis labels
    ax.set_title(
        "Monthly Total Visits",
        color="white",
        fontsize=16,
        fontweight="bold",
        pad=20
    )
    ax.set_xlabel("Month", color="#D1D5DB", labelpad=10)
    ax.set_ylabel("Total Visits", color="#D1D5DB", labelpad=10)
 
    # Grid and axis styling
    ax.set_ylim(bottom=0)
    ax.grid(
        axis="y",
        linestyle="--",
        color="#374151",
        alpha=0.8
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors="#D1D5DB")
    ax.tick_params(axis="y", colors="#D1D5DB")
 
    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)
 
    fig.tight_layout()
    return fig

#### Home Page 3 graph

def render_monthly_visits_chart(filtered_df,key_prefix="home",show_values=True):
 
    monthly = monthly_visit_summary(filtered_df).dropna(subset=["Month"]).copy()

    if monthly.empty:
        st.info("No monthly school visit data is available.")
        return

    monthly["Month"] = monthly["Month"].astype(str)

    chart_type = st.selectbox("Visits chart",["Line", "Bar"],key=f"{key_prefix}_visit_chart_type")

    if chart_type == "Line":
        fig = px.line(
                      monthly,
                      x="Month",
                      y="Total_Visit",
                      markers=True,
                      title="Monthly School Visits",
                      template="plotly_white")

        fig.update_traces(line=dict(color="#0284C7", width=3),
                          marker=dict(size=8),
                          text=monthly["Total_Visit"] if show_values else None,
                          texttemplate="%{text:,.0f}" if show_values else None,
                          textposition="top center",
                          hovertemplate=("Month: %{x}<br>""Visits: %{y:,.0f}<extra></extra>"))

    else:
        fig = px.bar(monthly,
                     x="Month",
                     y="Total_Visit",
                     title="Monthly School Visits",
                     template="plotly_white",
                     text="Total_Visit" if show_values else None,
                     color_discrete_sequence=["#0284C7"])

        fig.update_traces(texttemplate="%{text:,.0f}" if show_values else None,
            hovertemplate=("Month: %{x}<br>""Visits: %{y:,.0f}<extra></extra>"))

    fig.update_layout(height=390,
                      autosize=True,
                      margin=dict(l=20, r=20, t=65, b=35),
                      title=dict(x=0.03),
                      showlegend=False,
                      xaxis=dict(categoryorder="array",categoryarray=MONTH_ORDER),
                      yaxis=dict(title="Visits",rangemode="tozero",gridcolor="#E2E8F0"),
                      font=dict(family="Arial", color="#0F172A"),
                      hovermode="x unified")

    st.plotly_chart(fig,use_container_width=True,config={"responsive": True,"displaylogo": False,"scrollZoom": True},key=f"{key_prefix}_monthly_visits_chart")


def render_standards_chart(filtered_df,key_prefix="home",show_values=True):
 
    standards = standard_month(filtered_df).reset_index()
    if standards.empty:
        st.info("No monthly standards data is available.")
        return

    standards["Month"] = standards["Month"].astype(str)

    required_columns = ["Meets Standard", "Does Not Meet"]
    missing = [col for col in required_columns if col not in standards.columns]

    if missing:
        st.warning(f"Missing standards columns: {missing}")
        return

    standards_long = standards.melt(id_vars="Month",
                                    value_vars=required_columns,
                                    var_name="Status",
                                    value_name="Percentage")

    chart_mode = st.selectbox("Standards chart",["Grouped bars", "Stacked bars", "Line"],key=f"{key_prefix}_standards_chart_mode")
    colors = {"Meets Standard": "#059669","Does Not Meet": "#F97316"}

    if chart_mode == "Line":
        fig = px.line(standards_long,
                      x="Month",
                      y="Percentage",
                      color="Status",
                      markers=True,
                      title="Minimum Standards by Month",
                      template="plotly_white",
                      color_discrete_map=colors)

    else:
        fig = px.bar(standards_long,
                     x="Month",
                     y="Percentage",
                     color="Status",
                     barmode=("group" if chart_mode == "Grouped bars" else "stack"),
                     title="Standards Met by Month",
                     template="plotly_white",
                     color_discrete_map=colors)

    if show_values:
        fig.update_traces(texttemplate="%{y:.0f}%",
            textposition=("outside" if chart_mode != "Line" else "top center"))

    fig.update_traces(
        hovertemplate=("Month: %{x}<br>""%{fullData.name}: %{y:.1f}%<extra></extra>"))

    fig.update_layout(height=390,
                      autosize=True,
                      margin=dict(l=20, r=20, t=65, b=35),
                      title=dict(x=0.03),
                      yaxis=dict(title="Percentage (%)",range=[0, 105 if show_values else 100],gridcolor="#E2E8F0"),
                      xaxis=dict(categoryorder="array",categoryarray=MONTH_ORDER),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                      font=dict(family="Arial", color="#0F172A"),
                      hovermode="x unified")

    st.plotly_chart(fig,
                    use_container_width=True,
                    config={"responsive": True,"displaylogo": False,"scrollZoom": True},key=f"{key_prefix}_standards_chart")


def render_priority_chart(filtered_df,key_prefix="home"):
 
    priority_table = priority_teacher_table(filtered_df)
    if priority_table.empty:
        st.info("No teacher priority data is available.")
        return

    priority_values = priority_table.copy()
    priority_cols = [f"Priority_{i}" for i in range(4)]

    for col in priority_cols:
        if col not in priority_values.columns:
            priority_values[col] = 0

        priority_values[col] = pd.to_numeric(priority_values[col].astype(str).str.replace("%", "", regex=False),errors="coerce").fillna(0)

    priority_values.index.name = "Month"
    priority_values = priority_values.reset_index()
    priority_values["Month"] = priority_values["Month"].astype(str)

    priority_values["_month_order"] = (priority_values["Month"].map({month: i for i, month in enumerate(MONTH_ORDER)}))
    priority_values = (priority_values.sort_values("_month_order").drop(columns="_month_order"))
 
    control1, control2, control3 = st.columns(3)

    with control1:
        priority_view = st.selectbox("Chart style",
                                     ["Trend lines", "Stacked area", "Stacked bars"],
                                     key=f"{key_prefix}_priority_view")

    with control2:
        priority_focus = st.selectbox("Priority area",
                                      ["All priorities",
                                       "Priority 0",
                                       "Priority 1",
                                       "Priority 2",
                                       "Priority 3"],
                                      key=f"{key_prefix}_priority_focus")

    with control3:
        priority_metric = st.selectbox("Display metric",["Percentage (%)", "Teacher records"],key=f"{key_prefix}_priority_metric")

    priority_labels = {"Priority_0": "Priority 0",
                       "Priority_1": "Priority 1",
                       "Priority_2": "Priority 2",
                       "Priority_3": "Priority 3"}

    priority_long_df = priority_values.melt(id_vars="Month",
                                            value_vars=priority_cols,
                                            var_name="Priority",
                                            value_name="Percentage")

    priority_long_df["Priority"] = (priority_long_df["Priority"].map(priority_labels))

    if priority_metric == "Teacher records":
        month_counts = {}

        for month in MONTH_ORDER:
            col = f"{PRIORITY_STUB}_{month}"

            if col in filtered_df.columns:
                month_counts[month] = int(pd.to_numeric(filtered_df[col],errors="coerce").isin([0, 1, 2, 3]).sum())
            else:
                month_counts[month] = 0

        priority_long_df["Records"] = (priority_long_df.apply(lambda row: (row["Percentage"]* month_counts.get(row["Month"], 0)/ 100),axis=1))

        y_col = "Records"
        y_title = "Number of Teacher"

    else:
        y_col = "Percentage"
        y_title = "Teachers (%)"

    if priority_focus != "All priorities":
        priority_long_df = priority_long_df[priority_long_df["Priority"] == priority_focus]

    if priority_long_df.empty:
        st.info("No priority values are available for this selection.")
        return

    colors = {"Priority 0": "#64748B",
              "Priority 1": "#0EA5E9",
              "Priority 2": "#8B5CF6",
              "Priority 3": "#F59E0B"}

    if priority_view == "Stacked area" and priority_focus == "All priorities":
        fig = px.area(priority_long_df,
                      x="Month",
                      y=y_col,
                      color="Priority",
                      title="Monthly Distribution of Teacher Priority Areas",
                      template="plotly_white",
                      category_orders={"Month": MONTH_ORDER},
                      color_discrete_map=colors)

    elif priority_view == "Stacked bars" and priority_focus == "All priorities":
        fig = px.bar(priority_long_df,
                     x="Month",
                     y=y_col,
                     color="Priority",
                     barmode="stack",
                     title="Monthly Distribution of Teacher Priority Areas",
                     template="plotly_white",
                     category_orders={"Month": MONTH_ORDER},
                     color_discrete_map=colors)

    else:
        fig = px.line(priority_long_df,
                      x="Month",
                      y=y_col,
                      color="Priority",
                      markers=True,
                      title="Monthly Teacher Priority Trends",
                      template="plotly_white",
                      category_orders={"Month": MONTH_ORDER},
                      color_discrete_map=colors)

    fig.update_layout(height=440,
                      autosize=True,
                      margin=dict(l=20, r=20, t=70, b=35),
                      title=dict(x=0.5),
                      xaxis=dict(title="Month",categoryorder="array",categoryarray=MONTH_ORDER),
                      yaxis=dict(title=y_title,gridcolor="#E2E8F0",rangemode="tozero"),
                      legend=dict(title="Priority area",orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0),
                      font=dict(family="Arial", color="#0F172A"),
                      hovermode="x unified")

    st.plotly_chart(fig,use_container_width=True,
        config={"responsive": True,"displaylogo": False,"scrollZoom": True},key=f"{key_prefix}_priority_chart")


# ============================================================
# 3. VISITS ANALYSIS FUNCTIONS
# ============================================================
def table_visit(df: pd.DataFrame) -> pd.DataFrame:
    visit_cols = [c for c in df.columns if c.startswith(VISIT_PREFIX)]
    if not visit_cols:
        raise ValueError("No monthly visit columns were found.")
    staff_summary = df.pivot_table(index="RtR Staff Name", 
                                   values=["Project ID", "Teacher Name"],
                                   aggfunc={"Project ID": "nunique", "Teacher Name": "count"}, fill_value=0).rename(columns={"Project ID": "Total Number of Schools","Teacher Name": "Total Number of Teachers",})
    target_df = df.copy()
    target_df["target"] = [VISIT_TARGET_RULES.get((district, year, grade), 0) for district, year, grade in zip(target_df["District"], target_df["Year of Support"], target_df["Grade"])]
    target_df["Target_Visit"] = target_df["target"] * len(visit_cols)
    targets = target_df.groupby("RtR Staff Name")["Target_Visit"].sum()
    visited = target_df.groupby("RtR Staff Name")[visit_cols].sum().sum(axis=1).rename("Total_visited")
    result = pd.concat([staff_summary, targets, visited], axis=1).fillna(0)
    result["Gap of Visit"] = result["Target_Visit"] - result["Total_visited"]
    for grade in [1, 2]:
        grade_visits = target_df[target_df["Grade"] == grade].groupby("RtR Staff Name")[visit_cols].sum().sum(axis=1)
        result[f"Visit Grade_{grade}"] = grade_visits
    result = result.fillna(0)
    result = result[["Total Number of Schools", "Total Number of Teachers", "Target_Visit", "Total_visited","Gap of Visit", "Visit Grade_1", "Visit Grade_2",]]
    total_row = result.sum(numeric_only=True).to_frame().T
    total_row.index = ["Total"]
    result = pd.concat([result, total_row])
    result.index.name = "RtR Staff Name"
    return result
 
 
def create_visit_distribution_chart(long_df: pd.DataFrame):
    col = VISIT_PREFIX
    if col not in long_df.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.text(0.5, 0.5, "No visit data available", ha="center", va="center")
        ax.axis("off")
        return fig
    values = pd.to_numeric(long_df[col], errors="coerce").dropna()
    counts = values.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(counts.index.astype(str), counts.values, label="Observations", width=0.5)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Visits recorded per month")
    ax.set_ylabel("Number of observations")
    ax.set_title("Monthly Visit Distribution")
    ax.legend()
    fig.tight_layout()
    return fig
 
 
def create_visit_gap_chart(visit_table: pd.DataFrame):
    chart_df = visit_table.drop(index="Total", errors="ignore")
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(chart_df.index.astype(str), chart_df["Gap of Visit"], label="Visit Gap", width=0.55)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Staff Name")
    ax.set_ylabel("Target visits minus actual visits")
    ax.set_title("School Visit Gap by Staff")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    return fig
 
# ============================================================
# 4. STANDARDS ANALYSIS FUNCTIONS
# ============================================================
def standard_grade_lf(df: pd.DataFrame) -> pd.DataFrame:
    stnd_grd_lf = df_long(df)
    stnd_grd_lf = pd.pivot_table(stnd_grd_lf,
                                 index = "RtR Staff Name",
                                 values="Meeting Minimum Standards By Grade?",
                                 columns = "Grade",
                                 aggfunc="sum",
                                 margins=True,
                                 margins_name="Total Standard Meet").astype(int)
    stnd_grd_lf = stnd_grd_lf.rename(columns={1:"Total Standard Meet_Grade-1", 2: "Total Standard Meet_Grade-2"})
    return stnd_grd_lf
 
def stnadard_LF_graph(df: pd.DataFrame):
    stnd_LF = standard_grade_lf(df)
    stnd_LF = stnd_LF[:-1]
    fig, ax = plt.subplots(figsize=(9,6))
    bars = ax.bar(stnd_LF.index.astype(str),stnd_LF["Total Standard Meet"], label = "Total Standard Meet", width = 0.55)
    ax.bar_label(bars, padding = 3, fontsize = 8)
    ax.set_xlabel("staff")
    ax.set_ylabel("Total Standard Meet")
    ax.set_title("Staff wise Standard Meet")
    ax.tick_params(axis = "x", rotation =45)
    ax.legend()
    fig.tight_layout()
    return fig
 
def standard_month(df: pd.DataFrame) -> pd.DataFrame:
    stnd_month = df_long(df)
    stnd_month = pd.pivot_table(
        stnd_month,
        index="Month",
        values="Project ID",
        columns="Meeting Minimum Standards By Grade?",
        aggfunc="count",
        margins=True,
        margins_name="TOTAL",
    )
    # Ensure both status columns exist even if the selected data contains
    # only "meets" or only "does not meet" observations.
    for status in [0, 1]:
        if status not in stnd_month.columns:
            stnd_month[status] = 0
    stnd_month = stnd_month.fillna(0)
    stnd_month = stnd_month.rename(columns={0: "Does Not Meet", 1: "Meets Standard"})
    stnd_month["Does Not Meet"] = (stnd_month["Does Not Meet"] / stnd_month["TOTAL"] * 100).round(0)
    stnd_month["Meets Standard"] = (stnd_month["Meets Standard"] / stnd_month["TOTAL"] * 100).round(0)
    stnd_month = stnd_month.drop(columns="TOTAL")
    stnd_month = stnd_month.iloc[:-1]
    stnd_month = stnd_month.reset_index()
    stnd_month["Month"] = pd.Categorical(stnd_month["Month"],categories=MONTH_ORDER,ordered=True)
    stnd_month = stnd_month.sort_values("Month")
    stnd_month = stnd_month.set_index("Month")
    return stnd_month
 
def standard_month_gph(df: pd.DataFrame):
    st_mn_gh = standard_month(df)
    x = np.arange(len(st_mn_gh.index))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, st_mn_gh["Does Not Meet"], width,label="Does Not Meet")
    bars2 = ax.bar(x + width/2, st_mn_gh["Meets Standard"], width,label="Meets Standard")
    ax.bar_label(bars1, fmt="%.0f%%", padding=3)
    ax.bar_label(bars2, fmt="%.0f%%", padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(st_mn_gh.index)
    ax.set_xlabel("Month")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Meeting Minimum Standards")
    ax.legend()
    ax.set_ylim(0, 110)
    fig.tight_layout()
    return fig
    
 
def standard_graph(df: pd.DataFrame):
    stnd_gph = df_long(df)
    counts_by_standard = (stnd_gph["Meeting Minimum Standards By Grade?"].value_counts())
    meets = counts_by_standard.get(1, 0)
    does_not_meet = counts_by_standard.get(0, 0)
    categories = ["Meets Standards", "Does Not Meet"]
    counts = [meets, does_not_meet]
    total = sum(counts)
    if total == 0:
        return None
    percentages = [count / total * 100 for count in counts]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1F2937")
    colors = ["#34D399", "#F87171"]
    bars = ax.bar(categories,
                  percentages,
                  width=0.4,
                  color=colors,
                  edgecolor="#FFFFFF",
                  linewidth=0.8,
                  zorder=3)
    ax.bar_label(bars,
                 labels=[f"{p:.1f}%" for p in percentages],
                 padding=6,
                 color="white",
                 fontsize=12,
                 fontweight="bold")
    ax.set_title("Percentage of Meeting Minimum Standards",
                  color="white",
                  fontsize=15,
                  fontweight="bold",
                  pad=20)
    ax.set_xlabel("Performance Category", color="#D1D5DB", labelpad=10)
    ax.set_ylabel("Percentage (%)", color="#D1D5DB", labelpad=10)
    ax.set_ylim(0, 110)
    ax.set_axisbelow(True)
    ax.grid(axis="y",
            linestyle="--",
            color="#374151",
            alpha=0.8)
    ax.tick_params(axis="x", colors="#D1D5DB")
    ax.tick_params(axis="y", colors="#D1D5DB")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig
 
 
# ============================================================
# 5. PRIORITY ANALYSIS FUNCTIONS
# ============================================================
def priority_long(df: pd.DataFrame) -> pd.DataFrame:
    """Return the common long-format dataset used by priority views."""
    id_columns = ["Field Office", "District", "RtR Staff Name", "Project ID", "School Name","Teacher Name", "Year of Support", "Grade", "Class",]
    stubs = [STANDARD_PREFIX, VISIT_PREFIX, PRIORITY_STUB]
    available_stubs = [stub for stub in stubs if any(c.startswith(stub + "_") for c in df.columns)]
    return pd.wide_to_long(df.copy(), stubnames=available_stubs, i=id_columns, j="Month", sep="_", suffix="[A-Za-z]+").reset_index()
 
def stndwisepriority(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col.startswith(VISIT_PREFIX):
            rename_map[col] = col.replace(VISIT_PREFIX, "Visits", 1)
        elif col.startswith(STANDARD_PREFIX):
            rename_map[col] = col.replace(STANDARD_PREFIX, "Minimum Standard", 1)
        elif col.startswith(PRIORITY_PREFIX):
            rename_map[col] = col.replace(PRIORITY_PREFIX, "PriorityArea", 1)
    df_temp = df.rename(columns=rename_map)
    id_cols = [c for c in ["RtR Staff Name", "Project ID", "Teacher Name", "Grade"] if c in df_temp.columns]
    long_df = pd.wide_to_long(df_temp, stubnames=["Visits", "Minimum Standard", "PriorityArea"],i=id_cols, j="Month", sep="_", suffix="[A-Za-z]+").reset_index()
    long_df["PriorityArea"] = pd.to_numeric(long_df["PriorityArea"], errors="coerce").astype("Int64")
    long_df["Month"] = pd.Categorical(long_df["Month"], categories=MONTH_ORDER, ordered=True)
    long_df["Minimum Standard"] = long_df["Minimum Standard"].map({0: "No", 1: "Yes"})
    long_df["Minimum Standard"] = pd.Categorical(long_df["Minimum Standard"], categories=["Yes", "No"], ordered=True)
    result = pd.crosstab(index=[long_df["Month"], long_df["Minimum Standard"]],columns=long_df["PriorityArea"], normalize="index").mul(100).round(0).astype(int)
    result = result.reindex(columns=[0, 1, 2, 3], fill_value=0)
    result.columns = [f"Priority_{c}" for c in result.columns]
    result.columns.name = None
    return result
 
def plot_stnd_priority_heatmap(df):
    result = stndwisepriority(df).reset_index()
    priority_cols = ["Priority_0", "Priority_1","Priority_2", "Priority_3"]
    available_months = [month for month in MONTH_ORDER if month in result["Month"].astype(str).unique()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax, status in zip(axes, ["Yes", "No"]):
        subset = result[result["Minimum Standard"] == status].copy()
        heatmap_data = (subset.set_index("Month")[priority_cols].reindex(available_months))
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt=".1f",
                    cmap="YlGnBu",
                    vmin=0,
                    vmax=100,
                    linewidths=0.5,
                    linecolor="white",
                    cbar_kws={"label": "Percentage (%)"},
                    ax=ax)
        ax.set_title(f"Minimum Standards: {status}")
        ax.set_xlabel("Teacher's Priority Area")
        ax.set_ylabel("Month" if status == "Yes" else "")
        ax.set_xticklabels(["Priority 0", "Priority 1","Priority 2", "Priority 3"],rotation=30,ha="right")
    fig.suptitle("Monthly Distribution of Teacher Priority Areas ""by Minimum Standards Status",fontsize=14)
    fig.tight_layout()
    return fig
 
 
def priority_teacher_table(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "Field Office", "District", "RtR Staff Name", "School Name",
        "Teacher Name", "Year of Support", "Grade",
    ]
    missing = [col for col in id_cols if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns for priority analysis: " + ", ".join(missing))
 
    priority_month_cols = [
        col for col in df.columns if col.startswith(PRIORITY_STUB + "_")
    ]
    if not priority_month_cols:
        return pd.DataFrame(columns=[f"Priority_{i}" for i in range(4)])
 
    long_df = pd.wide_to_long(
        df.copy(), stubnames=PRIORITY_STUB, i=id_cols,
        j="Month", sep="_", suffix=r"\w+"
    ).reset_index()
    long_df["Month"] = pd.Categorical(
        long_df["Month"], categories=MONTH_ORDER, ordered=True
    )
    long_df[PRIORITY_STUB] = pd.to_numeric(long_df[PRIORITY_STUB], errors="coerce")
    long_df = long_df[long_df[PRIORITY_STUB].isin([0, 1, 2, 3])]
 
    table = pd.crosstab(
        long_df["Month"], long_df[PRIORITY_STUB], normalize="index"
    ).mul(100).round(0)
    table = table.reindex(index=MONTH_ORDER, columns=[0, 1, 2, 3], fill_value=0)
    table.columns = [f"Priority_{int(c)}" for c in table.columns]
    table = table.loc[table.sum(axis=1) > 0]
    table = table.map(lambda value: f"{value:.0f}%")
    table.columns.name = None
    return table
 
def create_priority_chart(priority_table: pd.DataFrame):
    """Create an interactive Plotly priority chart."""
    chart_df = priority_table.copy()
    priority_cols = [f"Priority_{i}" for i in range(4)]

    for col in priority_cols:
        if col not in chart_df.columns:
            chart_df[col] = 0

        chart_df[col] = pd.to_numeric(
            chart_df[col].astype(str).str.replace("%", "", regex=False),
            errors="coerce"
        ).fillna(0)

    chart_df.index.name = "Month"
    chart_df = chart_df.reset_index()

    if "Month" not in chart_df.columns:
        chart_df = chart_df.rename(columns={chart_df.columns[0]: "Month"})

    chart_df["Month"] = chart_df["Month"].astype(str)
    chart_df["_month_order"] = chart_df["Month"].map(
        {month: i for i, month in enumerate(MONTH_ORDER)}
    )
    chart_df = chart_df.sort_values("_month_order").drop(
        columns="_month_order"
    )

    chart_df = chart_df.melt(
        id_vars="Month",
        value_vars=priority_cols,
        var_name="Priority",
        value_name="Percentage"
    )

    labels = {
        "Priority_0": "Priority 0 — No priority achieved",
        "Priority_1": "Priority 1 — Instructional routine",
        "Priority_2": "Priority 2 — Basic skills",
        "Priority_3": "Priority 3 — Advanced skills",
    }
    chart_df["Priority"] = chart_df["Priority"].map(labels)

    fig = px.line(
        chart_df,
        x="Month",
        y="Percentage",
        color="Priority",
        markers=True,
        title="Teacher Priority Areas by Month",
        template="plotly_white",
        category_orders={"Month": MONTH_ORDER},
        color_discrete_map={
            "Priority 0 — No priority achieved": "#64748B",
            "Priority 1 — Instructional routine": "#0EA5E9",
            "Priority 2 — Basic skills": "#8B5CF6",
            "Priority 3 — Advanced skills": "#F59E0B",
        },
        custom_data=["Priority"]
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Month: %{x}<br>"
            "Teachers: %{y:.1f}%"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        height=440,
        autosize=True,
        margin=dict(l=20, r=20, t=75, b=35),
        title=dict(x=0.03),
        xaxis=dict(
            title="Month",
            categoryorder="array",
            categoryarray=MONTH_ORDER
        ),
        yaxis=dict(
            title="Share of teachers (%)",
            range=[0, 100],
            ticksuffix="%",
            gridcolor="#E2E8F0"
        ),
        legend=dict(
            title="Priority area",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        font=dict(family="Arial", color="#0F172A"),
        hovermode="x unified"
    )

    return fig
 
def priority_staff_table(long_df: pd.DataFrame) -> pd.DataFrame:
    col = PRIORITY_STUB
    table = pd.pivot_table(long_df, 
                           index="RtR Staff Name", 
                           values="Teacher Name", 
                           columns=col,
        aggfunc="nunique", fill_value=0
    ).reindex(columns=[0, 1, 2, 3], fill_value=0)
    table = table.div(table.sum(axis=1).replace(0, 1), axis=0).mul(100)
    table.columns = [f"Priority_{int(c)}" for c in table.columns]
    table = table.map(lambda value: f"{value:.0f}%")
    table.columns.name = None
    return table
 
def priority_school_g1(df: pd.DataFrame,long_df: pd.DataFrame) -> pd.DataFrame:
    col = PRIORITY_STUB
    grade1 = long_df[long_df["Grade"] == 1]
    table = pd.pivot_table(grade1,
                           index="Month",
                           values="Teacher Name",
                           columns=col,
                           aggfunc="count",
                           fill_value=0,
                           observed=True).reindex(columns=[0, 1, 2, 3], fill_value=0)
    table = (table.div(table.sum(axis=1).replace(0, 1), axis=0).mul(100).round(0).astype(int))
    table = table.reindex([m for m in MONTH_ORDER if m in table.index])
    table.columns = [f"Priority_{int(c)}" for c in table.columns]
    table = table.map(lambda value: f"{value:.0f}%")
    table.columns.name = None
    table.index.name = "Month"
    return table
 
def priority_school_g2(df: pd.DataFrame,long_df: pd.DataFrame) -> pd.DataFrame:
    col = PRIORITY_STUB
    grade2 = long_df[long_df["Grade"] == 2]
    table = pd.pivot_table(grade2,
                           index="Month",
                           values="Teacher Name",
                           columns=col,
                           aggfunc="count",
                           fill_value=0,
                           observed=True).reindex(columns=[0, 1, 2, 3], fill_value=0)
    table = (table.div(table.sum(axis=1).replace(0, 1), axis=0).mul(100).round(0).astype(int))
    table = table.reindex([m for m in MONTH_ORDER if m in table.index])
    table.columns = [f"Priority_{int(c)}" for c in table.columns]
    table = table.map(lambda value: f"{value:.0f}%")
    table.columns.name = None
    table.index.name = "Month"
    return table
 
 
def teacher_priority_heatmap(df: pd.DataFrame, staff_name=None):
    months = MONTH_ORDER
    priority_prefix = PRIORITY_STUB + "_"
    priority_cols = {month: f"{priority_prefix}{month}"for month in months if f"{priority_prefix}{month}" in df.columns}
    if not priority_cols:
        return None, None
    df_temp = df.copy()
 
    if staff_name is not None:
        df_temp = df_temp[df_temp["RtR Staff Name"].eq(staff_name)].copy()
    if df_temp.empty:
        return None, None
    id_cols = [col for col in ["RtR Staff Name","Project ID","School Name","Teacher Name","Grade","Class",] if col in df_temp.columns]
 
    heatmap_data = df_temp[id_cols + list(priority_cols.values())].copy()
    heatmap_data = heatmap_data.rename(columns={col: month for month, col in priority_cols.items()})
    for month in priority_cols:
        heatmap_data[month] = pd.to_numeric(heatmap_data[month], errors="coerce")
        heatmap_data.loc[~heatmap_data[month].isin([0, 1, 2, 3]),month] = float("nan")
 
    available_months = [month for month in months if month in heatmap_data.columns and heatmap_data[month].notna().any()]
 
    if not available_months:
        return None, None
 
    heatmap_data = heatmap_data.dropna(subset=available_months, how="all")
 
    if heatmap_data.empty:
        return None, None
 
    sort_cols = [col for col in ["Teacher Name", "Project ID", "Grade"] if col in heatmap_data.columns]
 
    if sort_cols:
        heatmap_data = heatmap_data.sort_values(sort_cols, na_position="last")
 
    if "Teacher Name" in heatmap_data.columns:
        labels = (heatmap_data["Teacher Name"].fillna("Unknown Teacher").astype(str))
 
        if "School Name" in heatmap_data.columns:
            labels += (" | " + heatmap_data["School Name"].fillna("").astype(str))
        if "Grade" in heatmap_data.columns:
            labels += (" | Grade " + heatmap_data["Grade"].fillna("").astype(str))
        if "Class" in heatmap_data.columns:
            labels += (" | Class "+ heatmap_data["Class"].fillna("").astype(str))
    else:
        labels = pd.Series([f"Teacher {i + 1}" for i in range(len(heatmap_data))],index=heatmap_data.index,)
 
    plot_data = heatmap_data[available_months].copy()
    plot_data.index = labels
    colors = ["#d9eaf7","#78b7c5","#f4c36a","#d95f59",]
    cmap = ListedColormap(colors)
    cmap.set_bad("#eeeeee")
 
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5],cmap.N,)
    fig_height = max(5, len(plot_data) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))
 
    sns.heatmap(
        plot_data,
        cmap=cmap,
        norm=norm,
        mask=plot_data.isna(),
        annot=True,
        fmt=".0f",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"ticks": [0, 1, 2, 3],"label": "Priority Area",},
        ax=ax,)
 
    title = (f"Teacher Priority Trends — {staff_name}" if staff_name is not None else "Teacher Priority — All Staff")
 
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Teacher / School / Grade / Class")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig, plot_data
 
 
# ============================================================
# 6. STREAMLIT UI FUNCTIONS
# ============================================================
def apply_page_css():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {background-color: #0F172A;}
    [data-testid="stSidebar"] * {color: white !important;}
    [data-testid="stSidebar"] label {color: white !important;}
    [data-testid="stSidebar"] button, .stButton > button, .stDownloadButton > button {border-radius: 8px; font-weight: 600;}
    [data-testid="stDataFrame"] {border-radius: 10px;}
    .stAlert {border-radius: 10px;}
    .main .block-container {max-width: none; padding: 2rem;}
    </style>
    """, unsafe_allow_html=True)
 
 
def login_page():
    st.markdown("""
    <style>
    .stApp {background:linear-gradient(135deg,#0F172A 0%,#172554 50%,#0F172A 100%) !important;}
    .main .block-container {min-height:100vh;box-sizing:border-box;display:flex;flex-direction:column;justify-content:center;padding-top:30px!important;padding-bottom:30px!important;}
    .pim-title {text-align:center;color:white;font-size:42px;font-weight:700;margin-bottom:5px;}
    .pim-subtitle {text-align:center;color:#CBD5E1;font-size:17px;margin-bottom:30px;}
    [data-testid="stForm"] {background:rgba(15,23,42,.95);padding:30px!important;border-radius:18px;border:1px solid rgba(255,255,255,.18);box-shadow:0 10px 40px rgba(0,0,0,.45);}
    [data-testid="stForm"] label {color:#E5E7EB!important;font-weight:500;}
    [data-testid="stForm"] input {background-color:rgba(255,255,255,.08)!important;color:black!important;border:1px solid rgba(255,255,255,.20)!important;border-radius:8px!important;}
    [data-testid="stFormSubmitButton"] button {width:100%;background:#2563EB;color:white;border:0;border-radius:8px;padding:10px;font-size:16px;font-weight:600;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="pim-title">PIM Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="pim-subtitle">Monitor &nbsp;•&nbsp; Analyze &nbsp;•&nbsp; Improve</div>', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        st.markdown('<div style="text-align:center;color:#FFFFFF;font-size:24px;font-weight:600;margin-bottom:8px;">Welcome Back!</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:#CBD5E1;font-size:14px;margin-bottom:15px;">Sign in to access the PIM Dashboard</div>', unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("🔐 Login")
        if submitted:
            # Demo credentials from the original code. Use a secure auth system in production.
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.session_state.page = "upload"
                st.rerun()
            else:
                st.error("Incorrect username or password.")
 
def _logout_pim():
   st.session_state.logged_in = False
   st.session_state.page = "login"
   st.session_state.df = None
   st.session_state.selected_sheet = None
   for key in ["pim_navigation", "pim_district", "pim_staff_filter", "pim_year"]:
            st.session_state.pop(key, None) 


def upload_page():
    st.markdown("""
    <style>
    .stApp {background:linear-gradient(135deg,#F8FAFC 0%,#EEF2FF 50%,#F8FAFC 100%)!important;}
    .main .block-container {max-width:900px!important;margin:auto;padding-top:5rem!important;}
    .upload-title {text-align:center;color:#0F172A;font-size:36px;font-weight:700;margin-bottom:5px;}
    .upload-subtitle {text-align:center;color:#64748B;font-size:16px;margin-bottom:35px;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="upload-title">📂 Data Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-subtitle">Upload your PIM Excel file and select the worksheet you want to analyze.</div>', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 2, 1])
    with center:
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"], help="Upload your PIM Excel dataset.")
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                selected_sheet = st.selectbox("Select Worksheet", sheet_names)
                if st.button("Continue to Dashboard →"):
                    try:
                        raw_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, skiprows=17, header=None)
                        if len(raw_df) < 2:
                            st.error("The selected worksheet does not contain enough rows to create the required headers.")
                        else:
                            cleaned = data_clean(raw_df)
                            st.session_state.df = raw_df
                            st.session_state.selected_sheet = selected_sheet
                            st.session_state.page = "dashboard"
                            st.rerun()
                    except Exception as exc:
                        st.error("Unable to load or clean the selected worksheet.")
                        st.exception(exc)
            except Exception as exc:
                st.error("Unable to read the Excel file.")
                st.exception(exc)
        with st.container():
             st.markdown("""<style>div.stButton > button[key="pim_logout"] {background-color: #DC2626 !important;color: white !important;border-radius: 8px !important;}</style>""", unsafe_allow_html=True)
             st.button("🚪 Logout",use_container_width=True,key="pim_logout",on_click=_logout_pim)
 
def dashboard_page(df: pd.DataFrame, long_df: pd.DataFrame, metrics: dict):
    """Interactive dashboard with stable chart dimensions and global filters."""
    st.markdown("""
    <style>
      /* Keep main-page text readable even when Streamlit's app theme is dark. */
      .stApp, [data-testid="stAppViewContainer"] {
        background: #F4F7FB !important;
        color: #0F172A !important;
      }
      .main, .main .block-container,
      .main p, .main label, .main li, .main span,
      .main h1, .main h2, .main h3, .main h4,
      .main [data-testid="stMarkdownContainer"] {
        color: #0F172A;
      }
      .main .block-container { max-width: 1600px; padding: 1.25rem 1.5rem 2rem; }
      .main [data-testid="stCaptionContainer"] { color: #475569 !important; }
      .main [data-testid="stWidgetLabel"] p { color: #0F172A !important; }
      .main input, .main textarea,
      .main [data-baseweb="select"] > div {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
      }
      .main [data-testid="stCheckbox"] label,
      .main [data-testid="stCheckbox"] label p { color: #0F172A !important; }
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, #0F172A, #1E293B);
        padding: 14px 12px; border-radius: 14px;
        border: 1px solid #243247; box-shadow: 0 4px 14px rgba(15,23,42,.10);
        min-height: 112px; height: 100%; overflow-wrap: anywhere;
      }
      [data-testid="stMetricLabel"] p { font-size: .78rem !important; }
      [data-testid="stMetricValue"] { font-size: clamp(1.05rem, 1.5vw, 1.8rem) !important; }
      @media (max-width: 1100px) {
        [data-testid="stMetricLabel"] p { font-size: .68rem !important; }
        [data-testid="stMetricValue"] { font-size: 1rem !important; }
      }
      [data-testid="stMetricLabel"] { color: #CBD5E1 !important; }
      [data-testid="stMetricValue"] { color: #FFFFFF !important; }
      [data-testid="stMetricDelta"] { color: #A7F3D0 !important; }
      [data-testid="stSidebar"] { background: #0F172A; }
      [data-testid="stSidebar"] * { color: #F8FAFC; }
      div[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
      div[data-testid="stDataFrame"] { border-radius: 10px; }
      .pim-section { font-size: 1.08rem; font-weight: 700; color: #0F172A !important; margin: .4rem 0 .8rem; }
      @media (max-width: 900px) {
        .main .block-container { padding: .8rem .65rem 1.5rem; }
      }
    </style>
    """, unsafe_allow_html=True)
 
    # Widget callbacks run before Streamlit redraws the page, so reset/logout can
    # safely clear widget state without causing a StreamlitAPIException.
    def _reset_pim_filters():
        st.session_state["pim_district"] = "All districts"
        st.session_state["pim_staff_filter"] = "All staff"
        st.session_state["pim_year"] = "All years"
 

 
    # Navigation and filters live in the sidebar. Widget keys preserve selection
    # across Streamlit reruns and avoid accidentally resetting the page.
    with st.sidebar:
        st.markdown("<div style='font-size:25px;font-weight:800; padding:4px 0 12px'>PIM Dashboard</div>", unsafe_allow_html=True)
        st.caption(f"Workbook: {st.session_state.get('selected_sheet') or 'Uploaded data'}")
        page = st.radio(
            "Navigation", ["Home", "Visits", "Standards", "Priority", "Reports"],
            key="pim_navigation", label_visibility="visible"
        )
        st.divider()
        st.markdown("**Dashboard filters**")
        filtered_df = df.copy()
        if "District" in filtered_df.columns:
            districts = sorted(filtered_df["District"].dropna().astype(str).unique().tolist())
            selected_district = st.selectbox("District", ["All districts"] + districts, key="pim_district")
            if selected_district != "All districts":
                filtered_df = filtered_df[filtered_df["District"].astype(str) == selected_district]
        if "RtR Staff Name" in filtered_df.columns:
            staff_names = sorted(filtered_df["RtR Staff Name"].dropna().astype(str).unique().tolist())
            selected_staff_filter = st.selectbox("Staff member", ["All staff"] + staff_names, key="pim_staff_filter")
            if selected_staff_filter != "All staff":
                filtered_df = filtered_df[filtered_df["RtR Staff Name"].astype(str) == selected_staff_filter]
        if "Year of Support" in filtered_df.columns:
            years = sorted(filtered_df["Year of Support"].dropna().unique().tolist())
            if years:
                selected_year = st.selectbox("Support year", ["All years"] + years, key="pim_year")
                if selected_year != "All years":
                    filtered_df = filtered_df[filtered_df["Year of Support"] == selected_year]
        st.caption(f"Showing {len(filtered_df):,} records")
        st.divider()
        if st.button("📂 Change Data", use_container_width=True, key="pim_change_data"):
            st.session_state.page = "upload"
            st.rerun()
        st.button("↻ Reset filters", use_container_width=True, key="pim_reset_filters", on_click=_reset_pim_filters)
        st.button("🚪 Logout", use_container_width=True, key="pim_logout", on_click=_logout_pim)
 
    if filtered_df.empty:
        st.warning("No records match the selected filters. Reset the filters or choose another option.")
        return
 
    # Recompute everything from the filtered records so the whole dashboard
    # responds consistently to filter changes.
    metrics = calculate_dashboard_metrics(filtered_df)
    try:
        filtered_long_df = df_long(filtered_df)
    except Exception:
        filtered_long_df = long_df.copy()
#=================================================================================================================

 
    if page == "Home":
        st.markdown("""<style>.block-container {padding-top: 1em;}</style>""",unsafe_allow_html=True)

    # Dashboard metrics
        metric_values = [("Total Staff", f"{metrics['total_staff']:,}"),
                         ("Total Schools", f"{metrics['total_schools']:,}"),
                         ("Total Teachers", f"{metrics['total_teachers']:,}"),
                         ("Total School Visits", f"{metrics['total_visits']:,.0f}"),
                         ("standard Met", f"{metrics['standard_rate']:.1f}%"),
                         ("Average Priority", f"{metrics['avg_priority']:.2f}"),]
    
        metric_cols = st.columns(6, gap="small")
    
        for col, (label, value) in zip(metric_cols, metric_values):
            with col:
                st.metric(label, value)
        # Shared chart value control
        show_monthly_values = st.checkbox("Show values on charts",value=True,key="home_show_chart_values")
    
        # Graph 1 and Graph 2
        chart_left, chart_right = st.columns(2, gap="large")
    
        with chart_left:
            render_monthly_visits_chart(filtered_df,
                                        key_prefix="home",
                                        show_values=show_monthly_values)
    
        with chart_right:
            render_standards_chart(filtered_df,
                                   key_prefix="home",
                                   show_values=show_monthly_values)
        
        # Graph 3
        st.divider()
        render_priority_chart(filtered_df,key_prefix="home")

 #=================================================================================================================

    elif page == "Visits":
        st.markdown("""<style>.block-container {padding-top: 0rem;}</style>""", unsafe_allow_html=True)
        st.title("School Visits Summary")
        try:
            visit_table = table_visit(filtered_df)
            actual_total = metrics["total_visits"]
            target_total = visit_table.loc[visit_table.index != "Total", "Target_Visit"].sum()
            gap_total = visit_table.loc[visit_table.index != "Total", "Gap of Visit"].sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total School Visits", f"{actual_total:,.0f}")
            c2.metric("Target Visits", f"{target_total:,.0f}")
            c3.metric("Visit Gap", f"{gap_total:,.0f}")
            chart_left, chart_right = st.columns(2)
            with chart_left:
                fig = create_visit_gap_chart(visit_table)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with chart_right:
                fig = create_visit_distribution_chart(filtered_long_df)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            st.dataframe(visit_table, use_container_width=True)
            st.download_button("Download visits summary CSV", visit_table.to_csv().encode("utf-8-sig"),
                               "pim_visits_summary.csv", "text/csv")
        except Exception as exc:
            st.error(f"Unable to build the visit summary: {exc}")


 
 #=================================================================================================================

 
    elif page == "Standards":
        st.markdown("""<style>
                       .block-container {padding-top: 0rem;}
                       .block-container h1,
                       .block-container h2,
                       .block-container h3,
                       .block-container p,
                       .block-container label,
                       .block-container span,
                       .block-container div {color: black !important;}
                       </style>""", unsafe_allow_html=True)

        st.title("Minimum Standards")

        left, right = st.columns(2)
        with left:
            st.dataframe(standard_month(filtered_df), height = "content" , width = "content")
        with right:
            fig = stnadard_LF_graph(filtered_df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
 #=================================================================================================================

 
    elif page == "Priority":
        st.markdown("""<style>
                       .block-container {padding-top: 0rem;}
                       .block-container h1,
                       .block-container h2,
                       .block-container h3,
                       .block-container p,
                       .block-container label,
                       .block-container span,
                       .block-container div {color: black !important;}
                       </style>""", unsafe_allow_html=True)

        st.title("Teacher Priority Areas")
        tab1, tab3, tab4, tab5 = st.tabs([
            "By standard", "By school", "By staff", "Teacher detail"
        ])
        with tab1:
            left, right = st.columns([1, 2])
            with left:
                st.dataframe(stndwisepriority(filtered_df), use_container_width=True)
            with right:
                fig = plot_stnd_priority_heatmap(filtered_df)
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("No priority-by-standard data is available for these filters.")

        with tab3:
            p_long = priority_long(filtered_df)
            left, right = st.columns(2)
            with left:
                st.markdown("**Grade 1 — priority areas by month**")
                st.dataframe(priority_school_g1(filtered_df, p_long), use_container_width=True)
            with right:
                st.markdown("**Grade 2 — priority areas by month**")
                st.dataframe(priority_school_g2(filtered_df, p_long), use_container_width=True)
        with tab4:
            p_long = priority_long(filtered_df)
            st.dataframe(priority_staff_table(p_long), use_container_width=True)
        with tab5:
            staff_options = sorted(filtered_df["RtR Staff Name"].dropna().unique().tolist()) if "RtR Staff Name" in filtered_df else []
            if staff_options:
                chosen_staff = st.selectbox("Select staff member", staff_options, key="priority_teacher_detail_staff")
                fig, plot_data = teacher_priority_heatmap(filtered_df, staff_name=chosen_staff)
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    with st.expander("View heatmap values"):
                        st.dataframe(plot_data, use_container_width=True)
                else:
                    st.info("No monthly priority data is available for this staff member.")
            else:
                st.info("No staff records are available for the selected filters.")
             
 #=================================================================================================================

 
    elif page == "Reports":
        st.markdown("""<style>.block-container {padding-top: 0rem;}</style>""", unsafe_allow_html=True)
        st.title("Reports & Exports")
        st.caption("Export the records currently selected by the sidebar filters.")
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            filtered_df.to_excel(writer, index=False, sheet_name="PIM Cleaned")
            filtered_long_df.to_excel(writer, index=False, sheet_name="PIM Monthly")
            try:
                table_visit(filtered_df).to_excel(writer, sheet_name="Visits Summary")
            except Exception:
                pass
            try:
                standard_grade_lf(filtered_df).to_excel(writer, sheet_name="Standards Summary")
            except Exception:
                pass
        st.download_button(label="📥 Download Excel Report", data=output.getvalue(),
                           file_name="PIM_Report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True,)
     
        st.markdown("**Preview of included records**")
        st.dataframe(filtered_df.head(100), use_container_width=True, hide_index=True)
 
 
def initialize_session_state():
    defaults = {"logged_in": False, "page": "login", "df": None, "selected_sheet": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
 
 
def main():
    st.set_page_config(page_title="PIM Dashboard",
                       page_icon="🎯",
                       layout="wide",
                       initial_sidebar_state="expanded",
                       menu_items={
                           "About": "PIM Dashboard"})
    initialize_session_state()
 
    if not st.session_state.logged_in or st.session_state.page == "login":
        login_page()
        return
    if st.session_state.page == "upload":
        upload_page()
        return
    if st.session_state.page == "dashboard":
        if st.session_state.df is None:
            st.warning("Please upload a workbook first.")
            st.session_state.page = "upload"
            st.rerun()
        try:
            cleaned_df = data_clean(st.session_state.df)
            monthly_df = df_long(cleaned_df)
            metrics = calculate_dashboard_metrics(cleaned_df)
            apply_page_css()
            dashboard_page(cleaned_df, monthly_df, metrics)
        except Exception as exc:
            st.error("The workbook could not be processed. Please check that it uses the expected PIM template.")
            st.exception(exc)
            if st.button("Back to Upload"):
                st.session_state.page = "upload"
                st.rerun()
 
 
if __name__ == "__main__":
    main()
