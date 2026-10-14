import re
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISIT_PREFIX = "Total Number of Visits Per Month"
STANDARD_PREFIX = "Meeting Minimum Standards By Grade?"
PRIORITY_PREFIX = "Teacher's Priority Area (0, 1, 2, or 3)"
PRIORITY_STUB = PRIORITY_PREFIX

VISIT_TARGET_RULES = {
    ("Jhalakathi", 1, 1): 2, ("Jhalakathi", 2, 1): 2,
    ("Jhalakathi", 2, 2): 2, ("Jhalakathi", 3, 1): 2,
    ("Jhalakathi", 3, 2): 2, ("Jhalakathi", 4, 1): 2,
    ("Jhalakathi", 4, 2): 2,
    ("Jamalpur", 1, 1): 1, ("Jamalpur", 1, 2): 1,
    ("Jamalpur", 2, 1): 1, ("Jamalpur", 2, 2): 1,
    ("Jamalpur", 3, 1): 1, ("Jamalpur", 3, 2): 1,
    ("Jamalpur", 4, 1): 1, ("Jamalpur", 4, 2): 1,
    ("Moulvibazar", 1, 1): 2, ("Moulvibazar", 2, 1): 2,
    ("Moulvibazar", 2, 2): 2, ("Moulvibazar", 3, 1): 1,
    ("Moulvibazar", 3, 2): 2, ("Moulvibazar", 4, 1): 1,
    ("Moulvibazar", 4, 2): 1,
    ("Narail", 1, 1): 1, ("Narail", 1, 2): 1,
    ("Narail", 2, 1): 1, ("Narail", 2, 2): 1,
    ("Narail", 2, 3): 1, ("Narail", 3, 1): 1,
    ("Narail", 3, 2): 1, ("Narail", 3, 3): 1,
}

ID_COLUMNS = [
    "Field Office", "District", "RtR Staff Name", "Project ID",
    "School Name", "Teacher Name", "Year of Support", "Grade", "Class",
]


def _month_name(value: Any) -> str | None:
    """Normalize Excel dates or month strings to Jan, Feb, etc."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    # Month names already present in the workbook.
    short = text[:3].title()
    if short in MONTH_ORDER:
        return short
    # Excel may store the second header row as a date or date-like string.
    try:
        parsed = pd.to_datetime(value, errors="raise")
        return parsed.strftime("%b")
    except (ValueError, TypeError, OverflowError):
        return None


def data_clean(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build column names from the two header rows and normalize values."""
    if raw_df is None or raw_df.empty or len(raw_df) < 2:
        raise ValueError("The worksheet must contain two header rows.")

    df = raw_df.copy()
    main_header = df.iloc[0].ffill()
    month_header = df.iloc[1]
    columns: list[str] = []

    for main, month in zip(main_header, month_header):
        main_text = str(main).strip() if pd.notna(main) else "Unnamed"
        month_name = _month_name(month)
        if month_name:
            columns.append(f"{main_text}_{month_name}")
        elif pd.notna(month) and str(month).strip():
            # Keep non-month suffixes rather than silently changing them.
            columns.append(f"{main_text}_{str(month).strip()}")
        else:
            columns.append(main_text)

    df.columns = columns
    df = df.iloc[2:].reset_index(drop=True)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")

    # Normalize month suffixes in columns that already have month names/dates.
    normalized = []
    for col in df.columns:
        text = str(col).strip()
        if "_" in text:
            prefix, suffix = text.rsplit("_", 1)
            month = _month_name(suffix)
            if month:
                text = f"{prefix}_{month}"
        normalized.append(text)
    df.columns = normalized

    if "School Name" not in df.columns:
        raise ValueError(
            "Could not find 'School Name'. Check the worksheet and the header-row offset."
        )
    df = df.dropna(subset=["School Name"]).copy()
    if "S/N" in df.columns:
        df = df.drop(columns=["S/N"])

    standard_cols = [c for c in df.columns if c.startswith(STANDARD_PREFIX)]
    priority_cols = [c for c in df.columns if c.startswith(PRIORITY_PREFIX)]
    visit_cols = [c for c in df.columns if c.startswith(VISIT_PREFIX)]

    yes_no = {"No": 0, "Yes": 1, "NO": 0, "YES": 1, 0: 0, 1: 1}
    priority_map = {
        "0: No Priority Areas Achieved": 0,
        "1: Mastered Instructional Routine": 1,
        "2: Mastered Basic Skills": 2,
        "3: Mastered Advanced Skills": 3,
        0: 0, 1: 1, 2: 2, 3: 3,
    }
    for col in standard_cols:
        df[col] = df[col].map(
            lambda x: yes_no.get(str(x).strip(), yes_no.get(x)) if pd.notna(x) else np.nan
        )
    for col in priority_cols:
        df[col] = df[col].map(
            lambda x: priority_map.get(str(x).strip(), priority_map.get(x))
            if pd.notna(x) else np.nan
        )
    for col in visit_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["Year of Support", "Grade"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def _monthly_columns(df: pd.DataFrame, stub: str) -> list[str]:
    return [c for c in df.columns if c.startswith(stub + "_")]


def to_monthly_long(df: pd.DataFrame, require_visits: bool = True) -> pd.DataFrame:
    """Convert monthly wide columns to long format without requiring unique IDs.

    A temporary row ID is used so duplicate teacher/school identifier combinations
    do not cause wide_to_long to fail or merge records unexpectedly.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=ID_COLUMNS + ["Month"])

    work = df.copy().reset_index(drop=True)
    work["_pim_row_id"] = np.arange(len(work))
    stubs = [
        stub for stub in [STANDARD_PREFIX, VISIT_PREFIX, PRIORITY_STUB]
        if _monthly_columns(work, stub)
    ]
    if not stubs:
        raise ValueError("No monthly visit, standards, or priority columns were found.")

    long_df = pd.wide_to_long(
        work,
        stubnames=stubs,
        i="_pim_row_id",
        j="Month",
        sep="_",
        suffix=r"[A-Za-z]+",
    ).reset_index()

    # Reattach non-month identifier columns after reshaping.
    original = work.set_index("_pim_row_id")
    for col in work.columns:
        if col == "_pim_row_id" or col in stubs:
            continue
        if not any(col.startswith(stub + "_") for stub in stubs):
            long_df[col] = long_df["_pim_row_id"].map(original[col])

    if VISIT_PREFIX not in long_df.columns:
        if require_visits:
            raise ValueError(f"Expected monthly visit columns beginning '{VISIT_PREFIX}_'.")
        long_df[VISIT_PREFIX] = np.nan

    if STANDARD_PREFIX not in long_df.columns:
        long_df[STANDARD_PREFIX] = np.where(
            pd.to_numeric(long_df[VISIT_PREFIX], errors="coerce").fillna(0) >= 2, 1, 0
        )
    else:
        long_df[STANDARD_PREFIX] = pd.to_numeric(
            long_df[STANDARD_PREFIX], errors="coerce"
        )

    long_df[VISIT_PREFIX] = pd.to_numeric(long_df[VISIT_PREFIX], errors="coerce").fillna(0)
    if PRIORITY_STUB in long_df.columns:
        long_df[PRIORITY_STUB] = pd.to_numeric(long_df[PRIORITY_STUB], errors="coerce")
    long_df["Month"] = pd.Categorical(long_df["Month"], categories=MONTH_ORDER, ordered=True)
    return long_df.drop(columns=["_pim_row_id"], errors="ignore").reset_index(drop=True)


def calculate_dashboard_metrics(df: pd.DataFrame) -> dict[str, float]:
    visit_cols = _monthly_columns(df, VISIT_PREFIX)
    standard_cols = _monthly_columns(df, STANDARD_PREFIX)
    priority_cols = _monthly_columns(df, PRIORITY_PREFIX)
    standard_values = df[standard_cols].apply(pd.to_numeric, errors="coerce") if standard_cols else pd.DataFrame()
    priority_values = df[priority_cols].apply(pd.to_numeric, errors="coerce") if priority_cols else pd.DataFrame()
    return {
        "total_staff": int(df["RtR Staff Name"].nunique()) if "RtR Staff Name" in df else 0,
        "total_schools": int(df["School Name"].nunique()) if "School Name" in df else 0,
        "total_teachers": int(df["Teacher Name"].nunique()) if "Teacher Name" in df else 0,
        "total_visits": float(df[visit_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum()) if visit_cols else 0.0,
        "standard_rate": float(standard_values.mean().mean() * 100) if not standard_values.empty else 0.0,
        "avg_priority": float(priority_values.mean().mean()) if not priority_values.empty else 0.0,
    }


def monthly_visit_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = _monthly_columns(df, VISIT_PREFIX)
    if not cols:
        return pd.DataFrame({"Month": pd.Categorical([], categories=MONTH_ORDER, ordered=True),
                             "Total_Visit": []})
    summary = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().rename_axis("Column").reset_index(name="Total_Visit")
    summary["Month"] = summary["Column"].str.replace(
        rf"^{re.escape(VISIT_PREFIX)}_", "", regex=True
    )
    summary = summary[summary["Month"].isin(MONTH_ORDER)].copy()
    summary["Month"] = pd.Categorical(summary["Month"], categories=MONTH_ORDER, ordered=True)
    return summary[["Month", "Total_Visit"]].sort_values("Month").reset_index(drop=True)


def table_visit(df: pd.DataFrame) -> pd.DataFrame:
    visit_cols = _monthly_columns(df, VISIT_PREFIX)
    if not visit_cols:
        raise ValueError("No monthly visit columns were found.")
    required = ["RtR Staff Name", "District", "Year of Support", "Grade"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns for visit targets: " + ", ".join(missing))

    school_count = df.groupby("RtR Staff Name")["Project ID"].nunique() if "Project ID" in df else pd.Series(dtype=float)
    teacher_count = df.groupby("RtR Staff Name")["Teacher Name"].count() if "Teacher Name" in df else pd.Series(dtype=float)
    result = pd.concat([
        school_count.rename("Total Number of Schools"),
        teacher_count.rename("Total Number of Teachers"),
    ], axis=1).fillna(0)

    target_df = df.copy()
    target_df["target"] = [
        VISIT_TARGET_RULES.get((district, year, grade), 0)
        for district, year, grade in zip(
            target_df["District"], target_df["Year of Support"], target_df["Grade"]
        )
    ]
    target_df["Target_Visit"] = target_df["target"] * len(visit_cols)
    targets = target_df.groupby("RtR Staff Name")["Target_Visit"].sum()
    actuals = target_df.groupby("RtR Staff Name")[visit_cols].sum().sum(axis=1)
    result = result.join(targets).join(actuals.rename("Total_visited")).fillna(0)
    result["Gap of Visit"] = result["Target_Visit"] - result["Total_visited"]
    for grade in [1, 2]:
        grade_df = target_df[target_df["Grade"] == grade]
        result[f"Visit Grade_{grade}"] = grade_df.groupby("RtR Staff Name")[visit_cols].sum().sum(axis=1)
    result = result.fillna(0)
    columns = ["Total Number of Schools", "Total Number of Teachers", "Target_Visit",
               "Total_visited", "Gap of Visit", "Visit Grade_1", "Visit Grade_2"]
    result = result.reindex(columns=columns, fill_value=0)
    total_row = result.sum(numeric_only=True).to_frame().T
    total_row.index = ["Total"]
    result = pd.concat([result, total_row])
    result.index.name = "RtR Staff Name"
    return result


def standards_monthly(df: pd.DataFrame) -> pd.DataFrame:
    long_df = to_monthly_long(df, require_visits=False)
    valid = long_df[long_df[STANDARD_PREFIX].isin([0, 1])].copy()
    if valid.empty:
        return pd.DataFrame(columns=["Meets Standard", "Does Not Meet"])
    counts = pd.crosstab(valid["Month"], valid[STANDARD_PREFIX])
    counts = counts.reindex(index=MONTH_ORDER, columns=[0, 1], fill_value=0)
    totals = counts.sum(axis=1).replace(0, np.nan)
    result = pd.DataFrame({
        "Does Not Meet": (counts[0] / totals * 100).fillna(0).round(0),
        "Meets Standard": (counts[1] / totals * 100).fillna(0).round(0),
    })
    return result.loc[result.sum(axis=1) > 0]


def priority_monthly(df: pd.DataFrame) -> pd.DataFrame:
    long_df = to_monthly_long(df, require_visits=False)
    if PRIORITY_STUB not in long_df.columns:
        return pd.DataFrame(columns=[f"Priority_{i}" for i in range(4)])
    valid = long_df[long_df[PRIORITY_STUB].isin([0, 1, 2, 3])].copy()
    if valid.empty:
        return pd.DataFrame(columns=[f"Priority_{i}" for i in range(4)])
    counts = pd.crosstab(valid["Month"], valid[PRIORITY_STUB], normalize="index") * 100
    counts = counts.reindex(index=MONTH_ORDER, columns=[0, 1, 2, 3], fill_value=0).round(0)
    counts.columns = [f"Priority_{int(c)}" for c in counts.columns]
    return counts.loc[counts.sum(axis=1) > 0]


def apply_page_css() -> None:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {background:#0F172A;}
    [data-testid="stSidebar"] * {color:#F8FAFC;}
    .main .block-container {max-width:1600px;padding:1.25rem 1.5rem 2rem;}
    [data-testid="stMetric"] {background:linear-gradient(145deg,#0F172A,#1E293B);
      padding:14px;border-radius:14px;border:1px solid #243247;min-height:100px;}
    [data-testid="stMetricLabel"] {color:#CBD5E1!important;}
    [data-testid="stMetricValue"] {color:#FFFFFF!important;}
    </style>
    """, unsafe_allow_html=True)


def login_page() -> None:
    st.markdown("""
    <style>
    .stApp {background:linear-gradient(135deg,#0F172A,#172554,#0F172A);}
    .pim-title {text-align:center;color:white;font-size:40px;font-weight:700;}
    .pim-subtitle {text-align:center;color:#CBD5E1;margin-bottom:25px;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="pim-title">📊 PIM Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="pim-subtitle">Monitor • Analyze • Improve</div>', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        st.subheader("Welcome Back!")
        st.caption("Sign in to access the PIM Dashboard")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("🔐 Login", use_container_width=True)
        if submitted:
            # Demo-only credentials. Do not use this for production authentication.
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.session_state.page = "upload"
                st.rerun()
            st.error("Incorrect username or password.")


def upload_page() -> None:
    st.title("📂 Data Upload")
    st.caption("Upload the PIM Excel file and select a worksheet to analyze.")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file is None:
        return
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("Select worksheet", excel_file.sheet_names)
        if st.button("Continue to Dashboard →", type="primary"):
            raw = pd.read_excel(uploaded_file, sheet_name=sheet, skiprows=17, header=None)
            cleaned = data_clean(raw)  # Validate before changing application state.
            st.session_state.df = cleaned
            st.session_state.selected_sheet = sheet
            st.session_state.page = "dashboard"
            st.rerun()
    except Exception as exc:
        st.error(f"Unable to read or clean the workbook: {exc}")


def dashboard_page(df: pd.DataFrame) -> None:
    apply_page_css()

    def reset_filters() -> None:
        for key, default in [("pim_district", "All districts"),
                             ("pim_staff", "All staff"), ("pim_year", "All years")]:
            st.session_state[key] = default

    with st.sidebar:
        st.markdown("## 📊 PIM Dashboard")
        st.caption(f"Worksheet: {st.session_state.get('selected_sheet', 'Uploaded data')}")
        page = st.radio("Navigation", ["Home", "Visits", "Standards", "Priority", "Reports"], key="pim_navigation")
        st.divider()
        filtered = df.copy()
        if "District" in filtered:
            options = sorted(filtered["District"].dropna().astype(str).unique())
            district = st.selectbox("District", ["All districts"] + options, key="pim_district")
            if district != "All districts":
                filtered = filtered[filtered["District"].astype(str) == district]
        if "RtR Staff Name" in filtered:
            options = sorted(filtered["RtR Staff Name"].dropna().astype(str).unique())
            staff = st.selectbox("Staff member", ["All staff"] + options, key="pim_staff")
            if staff != "All staff":
                filtered = filtered[filtered["RtR Staff Name"].astype(str) == staff]
        if "Year of Support" in filtered:
            years = sorted(filtered["Year of Support"].dropna().unique().tolist())
            year = st.selectbox("Support year", ["All years"] + years, key="pim_year")
            if year != "All years":
                filtered = filtered[filtered["Year of Support"] == year]
        st.caption(f"Showing {len(filtered):,} records")
        st.button("↻ Reset filters", on_click=reset_filters, use_container_width=True)
        if st.button("📂 Change data", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

    if filtered.empty:
        st.warning("No records match the selected filters.")
        return

    metrics = calculate_dashboard_metrics(filtered)
    if page == "Home":
        st.title("PIM Dashboard Overview")
        metric_data = [
            ("Total Staff", metrics["total_staff"]),
            ("Total Schools", metrics["total_schools"]),
            ("Total Teachers", metrics["total_teachers"]),
            ("Total School Visits", f'{metrics["total_visits"]:,.0f}'),
            ("Standards Met", f'{metrics["standard_rate"]:.1f}%'),
            ("Average Priority", f'{metrics["avg_priority"]:.2f}'),
        ]
        cols = st.columns(6)
        for col, (label, value) in zip(cols, metric_data):
            col.metric(label, value)

        left, right = st.columns(2)
        monthly = monthly_visit_summary(filtered)
        with left:
            st.subheader("Monthly School Visits")
            if not monthly.empty:
                fig = px.line(monthly, x="Month", y="Total_Visit", markers=True,
                              category_orders={"Month": MONTH_ORDER}, template="plotly_white")
                fig.update_layout(yaxis_rangemode="tozero", height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No monthly visit columns found.")
        with right:
            st.subheader("Minimum Standards by Month")
            standard_data = standards_monthly(filtered)
            if not standard_data.empty:
                fig = px.bar(standard_data.reset_index().melt(
                    id_vars="Month", var_name="Status", value_name="Percentage"),
                    x="Month", y="Percentage", color="Status", barmode="group",
                    category_orders={"Month": MONTH_ORDER}, template="plotly_white",
                    color_discrete_map={"Meets Standard": "#059669", "Does Not Meet": "#F97316"})
                fig.update_layout(yaxis_range=[0, 105], height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No standards data found.")

        st.subheader("Teacher Priority Insights")
        priority = priority_monthly(filtered)
        if priority.empty:
            st.info("No valid monthly priority values found.")
        else:
            priority_chart = priority.reset_index().melt(
                id_vars="Month", var_name="Priority", value_name="Percentage")
            fig = px.line(priority_chart, x="Month", y="Percentage", color="Priority",
                          markers=True, category_orders={"Month": MONTH_ORDER},
                          template="plotly_white")
            fig.update_layout(yaxis_range=[0, 100], yaxis_title="Share of teachers (%)", height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(priority, use_container_width=True)

    elif page == "Visits":
        st.title("School Visits Summary")
        try:
            visits = table_visit(filtered)
            non_total = visits.drop(index="Total", errors="ignore")
            c1, c2, c3 = st.columns(3)
            c1.metric("Actual visits", f'{non_total["Total_visited"].sum():,.0f}')
            c2.metric("Target visits", f'{non_total["Target_Visit"].sum():,.0f}')
            c3.metric("Visit gap", f'{non_total["Gap of Visit"].sum():,.0f}')
            left, right = st.columns(2)
            with left:
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.bar(non_total.index.astype(str), non_total["Gap of Visit"])
                ax.set_title("School Visit Gap by Staff")
                ax.set_ylabel("Target visits minus actual visits")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with right:
                long = to_monthly_long(filtered)
                counts = long[VISIT_PREFIX].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(9, 5))
                bars = ax.bar(counts.index.astype(str), counts.values)
                ax.bar_label(bars, padding=3)
                ax.set_title("Monthly Visit Distribution")
                ax.set_xlabel("Visits recorded per month")
                ax.set_ylabel("Number of observations")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            st.dataframe(visits, use_container_width=True)
            st.download_button("Download visits summary CSV",
                               visits.to_csv().encode("utf-8-sig"),
                               "pim_visits_summary.csv", "text/csv")
        except Exception as exc:
            st.error(f"Unable to build visit summary: {exc}")

    elif page == "Standards":
        st.title("Minimum Standards")
        monthly = standards_monthly(filtered)
        tab1, tab2 = st.tabs(["Monthly standards", "Standards by staff"])
        with tab1:
            left, right = st.columns(2)
            with left:
                st.dataframe(monthly, use_container_width=True)
            with right:
                if not monthly.empty:
                    fig = px.bar(monthly.reset_index().melt(
                        id_vars="Month", var_name="Status", value_name="Percentage"),
                        x="Month", y="Percentage", color="Status", barmode="group",
                        category_orders={"Month": MONTH_ORDER}, template="plotly_white")
                    fig.update_layout(yaxis_range=[0, 105])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No standards data found.")
        with tab2:
            long = to_monthly_long(filtered, require_visits=False)
            valid = long[long[STANDARD_PREFIX].isin([0, 1])]
            if "RtR Staff Name" in valid:
                summary = valid.pivot_table(index="RtR Staff Name",
                                            values=STANDARD_PREFIX,
                                            aggfunc="sum", observed=False)
                summary = summary.rename(columns={STANDARD_PREFIX: "Standards met"})
                st.dataframe(summary, use_container_width=True)
                fig = px.bar(summary.reset_index(), x="RtR Staff Name", y="Standards met",
                             template="plotly_white")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Staff or standards data is unavailable.")

    elif page == "Priority":
        st.title("Teacher Priority Areas")
        monthly = priority_monthly(filtered)
        tab1, tab2, tab3 = st.tabs(["Monthly trends", "By staff", "Teacher detail"])
        with tab1:
            if monthly.empty:
                st.info("No valid monthly priority values found.")
            else:
                left, right = st.columns([1, 2])
                with left:
                    st.dataframe(monthly, use_container_width=True)
                with right:
                    fig = px.line(monthly.reset_index().melt(
                        id_vars="Month", var_name="Priority", value_name="Percentage"),
                        x="Month", y="Percentage", color="Priority", markers=True,
                        category_orders={"Month": MONTH_ORDER}, template="plotly_white")
                    fig.update_layout(yaxis_range=[0, 100], yaxis_title="Share (%)")
                    st.plotly_chart(fig, use_container_width=True)
        with tab2:
            long = to_monthly_long(filtered, require_visits=False)
            if PRIORITY_STUB in long:
                valid = long[long[PRIORITY_STUB].isin([0, 1, 2, 3])]
                if not valid.empty:
                    summary = pd.crosstab(valid["RtR Staff Name"], valid[PRIORITY_STUB], normalize="index") * 100
                    summary = summary.reindex(columns=[0, 1, 2, 3], fill_value=0).round(0)
                    summary.columns = [f"Priority {int(c)} (%)" for c in summary.columns]
                    st.dataframe(summary, use_container_width=True)
                else:
                    st.info("No valid priority values found.")
            else:
                st.info("Priority data is unavailable.")
        with tab3:
            priority_cols = _monthly_columns(filtered, PRIORITY_STUB)
            if not priority_cols:
                st.info("No monthly priority columns found.")
            else:
                teacher_rows = filtered.copy()
                display_cols = [c for c in ["RtR Staff Name", "School Name", "Teacher Name",
                                            "Grade", "Class"] if c in teacher_rows]
                teacher_rows = teacher_rows[display_cols + priority_cols].copy()
                teacher_rows = teacher_rows.rename(columns={
                    c: c.replace(PRIORITY_STUB + "_", "") for c in priority_cols
                })
                for month in MONTH_ORDER:
                    if month in teacher_rows:
                        teacher_rows[month] = pd.to_numeric(teacher_rows[month], errors="coerce")
                st.dataframe(teacher_rows, use_container_width=True, hide_index=True)

    elif page == "Reports":
        st.title("Reports & Exports")
        st.caption("Exports use the records currently selected by the sidebar filters.")
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            filtered.to_excel(writer, index=False, sheet_name="PIM Cleaned")
            try:
                to_monthly_long(filtered, require_visits=False).to_excel(
                    writer, index=False, sheet_name="PIM Monthly")
            except Exception:
                pass
            try:
                table_visit(filtered).to_excel(writer, sheet_name="Visits Summary")
            except Exception:
                pass
            try:
                standards_monthly(filtered).to_excel(writer, sheet_name="Standards Summary")
            except Exception:
                pass
            try:
                priority_monthly(filtered).to_excel(writer, sheet_name="Priority Summary")
            except Exception:
                pass
        st.download_button(
            "📥 Download Excel Report", data=output.getvalue(),
            file_name="PIM_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.subheader("Preview of included records")
        st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)


def initialize_session_state() -> None:
    defaults = {"logged_in": False, "page": "login", "df": None, "selected_sheet": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(
        page_title="PIM Dashboard", page_icon="🎯", layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "PIM Dashboard — Project Implementation Monitoring"},
    )
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
            dashboard_page(st.session_state.df)
        except Exception as exc:
            st.error("The workbook could not be processed. Check the PIM template and column names.")
            st.exception(exc)
            if st.button("Back to Upload"):
                st.session_state.page = "upload"
                st.rerun()


if __name__ == "__main__":
    main()
