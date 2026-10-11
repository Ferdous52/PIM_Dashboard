# ============================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================
import re
from io import BytesIO
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    ("Jhalakathi", 2, 1): 2, ("Jhalakathi", 2, 2): 2,
    ("Jhalakathi", 3, 1): 2, ("Jhalakathi", 3, 2): 2,
    ("Jhalakathi", 4, 1): 2, ("Jhalakathi", 4, 2): 2,
    ("Jamalpur", 1, 1): 1, ("Jamalpur", 1, 2): 1,
    ("Jamalpur", 2, 1): 1, ("Jamalpur", 2, 2): 1,
    ("Jamalpur", 3, 1): 1, ("Jamalpur", 3, 2): 1,
    ("Jamalpur", 4, 1): 1, ("Jamalpur", 4, 2): 1,
    ("Moulvibazar", 1, 1): 2,
    ("Moulvibazar", 2, 1): 2, ("Moulvibazar", 2, 2): 2,
    ("Moulvibazar", 3, 1): 1, ("Moulvibazar", 3, 2): 2,
    ("Moulvibazar", 4, 1): 1, ("Moulvibazar", 4, 2): 1,
    ("Narail", 1, 1): 1, ("Narail", 1, 2): 1,
    ("Narail", 2, 1): 1, ("Narail", 2, 2): 1, ("Narail", 2, 3): 1,
    ("Narail", 3, 1): 1, ("Narail", 3, 2): 1, ("Narail", 3, 3): 1,
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
    df_long["Meeting Minimum Standards By Grade?"] = np.where(df_long[visit_column] >= 2,1,0)
    return df_long
 
 
def calculate_dashboard_metrics(df: pd.DataFrame) -> dict:
    visit_cols = [c for c in df.columns if c.startswith(VISIT_PREFIX)]
    standard_cols = [c for c in df.columns if c.startswith(STANDARD_PREFIX)]
    priority_cols = [c for c in df.columns if c.startswith(PRIORITY_PREFIX)]
    return {
        "total_staff": df["RtR Staff Name"].nunique() if "RtR Staff Name" in df else 0,
        "total_schools": df["Project ID"].nunique() if "School Name" in df else 0,
        "total_teachers": df["Teacher Name"].nunique() if "Teacher Name" in df else 0,
        "total_visits": float(df[visit_cols].sum().sum()) if visit_cols else 0,
        "standard_rate": float(df[standard_cols].mean().mean() * 100) if standard_cols else 0,
        "avg_priority": float(df[priority_cols].mean().mean()) if priority_cols else 0,
    }
 
 
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
 
# ============================================================
# 3. VISITS ANALYSIS FUNCTIONS
# ============================================================
def table_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Create staff-level visit target, actual visits, gap, and school/teacher totals."""
    visit_cols = [c for c in df.columns if c.startswith(VISIT_PREFIX)]
    if not visit_cols:
        raise ValueError("No monthly visit columns were found.")
 
    staff_summary = df.pivot_table(index="RtR Staff Name", 
                                   values=["Project ID", "Teacher Name"],
                                   aggfunc={"Project ID": "nunique", "Teacher Name": "count"}, fill_value=0).rename(columns={"Project ID": "Total Number of Schools","Teacher Name": "Total Number of Teachers",})
    target_df = df.copy()
    target_df["target"] = [
        VISIT_TARGET_RULES.get((district, year, grade), 0)
        for district, year, grade in zip(target_df["District"], target_df["Year of Support"], target_df["Grade"])
    ]
    target_df["Target_Visit"] = target_df["target"] * len(visit_cols)
    targets = target_df.groupby("RtR Staff Name")["Target_Visit"].sum()
    visited = target_df.groupby("RtR Staff Name")[visit_cols].sum().sum(axis=1).rename("Total_visited")
    result = pd.concat([staff_summary, targets, visited], axis=1).fillna(0)
    result["Gap of Visit"] = result["Target_Visit"] - result["Total_visited"]
 
    for grade in [1, 2]:
        grade_visits = target_df[target_df["Grade"] == grade].groupby("RtR Staff Name")[visit_cols].sum().sum(axis=1)
        result[f"Visit Grade_{grade}"] = grade_visits
 
    result = result.fillna(0)
    result = result[[
        "Total Number of Schools", "Total Number of Teachers", "Target_Visit", "Total_visited",
        "Gap of Visit", "Visit Grade_1", "Visit Grade_2",
    ]]
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
    bars = ax.bar(stnd_LF.index.astype(str),stnd_LF["Total Standard Meet"], label = "Total Standard Meet", width = 0.55,  color="seagreen")
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
    stnd_month = pd.pivot_table(stnd_month,
                                index="Month",
                                values="Project ID",
                                columns="Meeting Minimum Standards By Grade?",
                                aggfunc="count",
                                margins=True,
                                margins_name="TOTAL")
    stnd_month = stnd_month.fillna(0)
    stnd_month = stnd_month.rename(columns={0: "Does Not Meet",1: "Meets Standard"})
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
    ax.set_title("Meeting Minimum Standards by Grade")
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
    result = pd.crosstab(index=[long_df["Month"], long_df["Minimum Standard"]],columns=long_df["PriorityArea"], normalize="index").mul(100).round(1)
    result = result.reindex(columns=[0, 1, 2, 3], fill_value=0)
    result.columns = [f"Priority_{c}" for c in result.columns]
    result.columns.name = None
    return result
 
 
def priority_teacher_table(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["Field Office", "District", "RtR Staff Name", "School Name", "Teacher Name","Year of Support", "Grade",]
    long_df = pd.wide_to_long(df.copy(), stubnames=PRIORITY_STUB, i=id_cols, j="Month", sep="_", suffix=r"\w+").reset_index()
    long_df["Month"] = pd.Categorical(long_df["Month"], categories=MONTH_ORDER, ordered=True)
    table = pd.pivot_table(long_df, 
                           index="Month", 
                           values="Teacher Name", 
                           columns=PRIORITY_STUB,
                           aggfunc="count", 
                           observed=False).reindex(columns=[0, 1, 2, 3], fill_value=0)
    denominator = max(df["Teacher Name"].count(), 1)
    table = (table / denominator * 100).round(0)
    table.columns = [f"Priority_{int(c)}" for c in table.columns]
    table = table.loc[table.sum(axis=1) > 0]
    table = table.map(lambda value: f"{value:.0f}%")
    table.columns.name = None
    return table
 
def create_priority_chart(priority_table: pd.DataFrame):
    chart_df = priority_table.copy()
    priority_cols = [f"Priority_{i}" for i in range(4)]

    for col in priority_cols:
        if col not in chart_df:
            chart_df[col] = 0

        chart_df[col] = pd.to_numeric(chart_df[col].astype(str).str.replace("%", "", regex=False),errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1F2937")
    colors = ["#38BDF8", "#A78BFA", "#34D399", "#FBBF24"]

    for col, color in zip(priority_cols, colors):
        ax.plot(
            chart_df["Month"] if "Month" in chart_df.columns
            else chart_df.index,
            chart_df[col],
            label=col.replace("_", " "),
            color=color,
            marker="o",
            linewidth=2.8,
            markersize=7,
            markerfacecolor="#FFFFFF",
            markeredgecolor=color,
            markeredgewidth=1.5
        )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.set_xlabel("Month", color="#D1D5DB", labelpad=10)
    ax.set_ylabel("Teachers (%)", color="#D1D5DB", labelpad=10)

    ax.set_title("Priority Areas by Month",
                 color="white",
                 fontsize=16,
                 fontweight="bold",pad=20
    )

    ax.set_ylim(0, 100)
    ax.grid(axis="y",
            linestyle="--",color="#374151",alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors="#D1D5DB")
    ax.tick_params(axis="y", colors="#D1D5DB")

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(title="Priority Area",
                       facecolor="#1F2937",
                       edgecolor="#374151",
                       labelcolor="white")
    legend.get_title().set_color("white")
    fig.tight_layout()
    return fig
 
def priority_staff_table(long_df: pd.DataFrame) -> pd.DataFrame:
    col = PRIORITY_STUB
    table = pd.pivot_table(
        long_df, index="RtR Staff Name", values="School Name", columns=col,
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
                           values="Project ID",
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
                           values="Project ID",
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
    [data-testid="stForm"] input {background-color:rgba(255,255,255,.08)!important;color:white!important;border:1px solid rgba(255,255,255,.20)!important;border-radius:8px!important;}
    [data-testid="stFormSubmitButton"] button {width:100%;background:#2563EB;color:white;border:0;border-radius:8px;padding:10px;font-size:16px;font-weight:600;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="pim-title">📊 PIM Dashboard</div>', unsafe_allow_html=True)
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
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                selected_sheet = st.selectbox("Select Worksheet", sheet_names)
                if st.button("Continue to Dashboard →", use_container_width=True):
                    try:
                        raw_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, skiprows=17, header=None)
                        if len(raw_df) < 2:
                            st.error("The selected worksheet does not contain enough rows to create the required headers.")
                        else:
                            # Validate now so header issues are shown on the upload page.
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
 
 
def dashboard_page(df: pd.DataFrame, long_df: pd.DataFrame, metrics: dict):
    st.markdown("""
    <style>
    .stApp {background:#F4F9F5!important;}
    [data-testid="stMetric"] {background:#0F172A;padding:18px;border-radius:12px;border:1px solid #E2E8F0;box-shadow:0 2px 8px rgba(0,0,0,.08);}
    [data-testid="stMetricLabel"] {color:#CBD5E1!important;}
    [data-testid="stMetricValue"] {color:#FFFFFF!important;}
    </style>
    """, unsafe_allow_html=True)
    #st.markdown('<div style="color:#0F172A;font-size:32px;font-weight:700;">PIM Dashboard</div>', unsafe_allow_html=True)
    #st.caption(f"Worksheet: {st.session_state.selected_sheet}")
    with st.sidebar:
        st.markdown('<div style="font-size:24px;font-weight:700;margin-bottom:20px;">PIM Dashboard</div>', unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio("Navigation", ["Home", "Visits", "Standards", "Priority", "Reports"])
        st.markdown("---")
        st.caption(f"Worksheet: {st.session_state.selected_sheet}")
        st.markdown("---")
        if st.button("📂 Change Data", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.session_state.df = None
            st.session_state.selected_sheet = None
            st.rerun()

#==============================================================================================================================================================================================
    st.markdown("""<style>h1, h2, h3 {color: black !important;}</style>""", unsafe_allow_html=True)
    
    if page == "Home":
        st.markdown("""<style>.block-container {padding-top: .5rem;}</style>""", unsafe_allow_html=True)
        cols = st.columns(6)
        labels = [
            ("Total Staff", metrics["total_staff"]),
            ("Total Schools", metrics["total_schools"]),
            ("Total Teachers", metrics["total_teachers"]),
            ("Total School Visits", f'{metrics["total_visits"]:,.0f}'),
            ("Standards Met", f'{metrics["standard_rate"]:.1f}%'),
            ("Average Priority Score", f'{metrics["avg_priority"]:.2f}'),
        ]
        for col, (label, value) in zip(cols, labels):
            col.metric(label, value)
         
        col1, col2, col3= st.columns(3)
        
        with col1:
         fig = create_monthly_visit_chart(df)
         st.pyplot(fig, use_container_width=True)
         plt.close(fig)
        
        with col2:
         fig = standard_graph(df)
         st.pyplot(fig, use_container_width=True)
         plt.close(fig)
         
        with col3:
         teacher_table = priority_teacher_table(df)
         fig = create_priority_chart(teacher_table)
         st.pyplot(fig, use_container_width=True)
         plt.close(fig)
         
#==========================================================================================================================================================================================

    elif page == "Visits":
        st.markdown("""<style>.block-container {padding-top: 0rem;}</style>""", unsafe_allow_html=True)
        st.header("School Visits Summary")
        visit_table = table_visit(df)
        actual_total = metrics["total_visits"]
        target_total = visit_table.loc[visit_table.index != "Total", "Target_Visit"].sum()
        gap_total = visit_table.loc[visit_table.index != "Total", "Gap of Visit"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total School Visits", f"{actual_total:,.0f}")
        c2.metric("Target Visits", f"{target_total:,.0f}")
        c3.metric("Visit Gap", f"{gap_total:,.0f}")
        left, right = st.columns(2)
        with left:
            fig = create_visit_gap_chart(visit_table)
            st.pyplot(fig)
            plt.close(fig)
        with right:
            fig = create_visit_distribution_chart(long_df)
            st.pyplot(fig)
            plt.close(fig)
        st.dataframe(visit_table, use_container_width=True)

#============================================================================================================================================================================================
 
    elif page == "Standards":
        st.markdown("""<style>.block-container {padding-top: 0rem;}</style>""", unsafe_allow_html=True)
        st.subheader("Standards Meet")
        tab1, tab2 = st.tabs(["Standard Meet Monthly", "Standard Meet by Staff"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(standard_month(df),width="stretch",height = "content")
            with col2:
                st.pyplot(standard_month_gph(df))
        with tab2:
            standards = standard_grade_lf(df)
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(standards, width = "content", height = "content")
            with col2:
                st.pyplot(stnadard_LF_graph(df))

#==============================================================================================================================================================================================
 
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

        st.subheader("Teacher's Priority Area")
        tab1, tab2, tab3 = st.tabs(["Priority by Standard", "Priority by Teacher Monthly", "Priority by School and Staff"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(stndwisepriority(df),width="content",height="content")
            with col2:
                pass
        with tab2:
            left, right = st.columns(2)
            teacher_table = priority_teacher_table(df)
            with left:
                st.dataframe(teacher_table)
            with right:
                fig = create_priority_chart(teacher_table)
                st.pyplot(fig)
                plt.close(fig)
        with tab3:
            p_long = priority_long(df)
            left, right = st.columns(2)
            with left:
                st.markdown("**Percentage of schools in each teacher priority area by month : Grade-1**")
                st.dataframe(priority_school_g1(df, p_long))
             
            with right:
                st.markdown("**Percentage of schools in each teacher priority area by month : Grade-2**")
                st.dataframe(priority_school_g2(df, p_long))
     
            st.markdown("**Percentage of schools handled by each RtR staff member in each priority area**")
            st.dataframe(priority_staff_table(p_long), use_container_width=True)
#============================================================================================================================================================================================= 
 
    elif page == "Reports":
        st.markdown("""<style>.block-container {padding-top: 0rem;}</style>""", unsafe_allow_html=True)
        st.subheader("Reports")
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="PIM Cleaned")
            long_df.to_excel(writer, index=False, sheet_name="PIM Monthly")
            table_visit(df).to_excel(writer, sheet_name="Visits Summary")
            standard_grade_lf(df).to_excel(writer, index=False, sheet_name="Standards Summary")
        st.download_button(
            label="Download Excel Report", data=output.getvalue(), file_name="PIM_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
 
# ============================================================
# 7. MAIN APPLICATION AND PAGE ROUTING
# ============================================================
def initialize_session_state():
    defaults = {"logged_in": False, "page": "login", "df": None, "selected_sheet": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
 
 
def main():
    st.set_page_config(page_title="PIM Dashboard",
                       page_icon="🎯",
                       layout="wide",
                       initial_sidebar_state="collapsed",
                       menu_items={
                           "About": "PIM Dashboard — Project Implementation Monitoring"})
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
