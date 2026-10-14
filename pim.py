                    st.info("No priority-by-standard data is available for these filters.")
        with tab2:
            teacher_table = priority_teacher_table(filtered_df)
            left, right = st.columns(2)
            with left:
                st.dataframe(teacher_table, use_container_width=True)
            with right:
                fig = create_priority_chart(teacher_table)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
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
        st.download_button(
            label="📥 Download Excel Report", data=output.getvalue(),
            file_name="PIM_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
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
