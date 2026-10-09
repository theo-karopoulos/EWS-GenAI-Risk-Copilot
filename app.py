import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv

from src.mrm_generator import generate_mrm_executive_report
from src.rag_engine import build_or_load_vectorstore, query_rag_system

load_dotenv()

st.set_page_config(
    page_title="GenAI Risk & EWS Copilot | Enterprise Risk Advisory",
    page_icon="🛡️",
    layout="wide"
)

# Sidebar Header
st.sidebar.title("🛡️ GenAI Risk Copilot")
st.sidebar.caption("Financial Risk Advisory & MRM Solutions")
st.sidebar.markdown("---")

# Main Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Analytics & Validation", 
    "📝 Executive MRM Report Generator", 
    "💬 Regulatory RAG Copilot",
    "🎯 Live Borrower EWS Simulator"
])

# ---------------------------------------------------------
# TAB 1: Model Analytics
# ---------------------------------------------------------
with tab1:
    st.header("Credit Risk Model Performance & Risk Drivers")
    st.markdown("Overview of the XGBoost Early Warning System (EWS) probability-of-default model.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model ROC-AUC", "0.868")
    col2.metric("Gini Coefficient", "0.736")
    col3.metric("Training Samples", "120,000")
    col4.metric("Test Samples", "30,000")

    st.markdown("---")
    st.subheader("Top Risk Drivers (SHAP Feature Importance)")
    
    feature_df = pd.DataFrame({
        "Risk Factor": [
            "Revolving Utilization of Unsecured Lines",
            "Past Due 30-59 Days Frequency",
            "Debt Ratio",
            "Monthly Income",
            "Age"
        ],
        "Relative SHAP Impact": [0.42, 0.28, 0.15, 0.09, 0.06]
    }).sort_values(by="Relative SHAP Impact", ascending=True)

    fig = px.bar(
        feature_df,
        x="Relative SHAP Impact",
        y="Risk Factor",
        orientation='h',
        text="Relative SHAP Impact",
        color="Relative SHAP Impact",
        color_continuous_scale="Blues"
    )
    
    fig.update_traces(
        texttemplate='%{text:.2f}', 
        textposition='outside',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.9
    )
    
    fig.update_layout(
        xaxis_title="Relative SHAP Importance Score",
        yaxis_title="",
        height=380,
        margin=dict(l=20, r=40, t=20, b=20),
        coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Executive MRM Report Generator
# ---------------------------------------------------------
with tab2:
    st.header("Executive MRM Validation Report Generator")
    st.markdown("Generate formal Model Risk Management (MRM) validation reports following institutional risk advisory standards.")

    if st.button("🚀 Generate Executive Validation Report", type="primary"):
        with st.spinner("Analyzing model artifacts and compiling executive report via GPT-4o-mini..."):
            sample_metrics = {
                "train_samples": 120000,
                "test_samples": 30000,
                "roc_auc": 0.868,
                "gini": 0.736
            }
            sample_features = [
                "RevolvingUtilizationOfUnsecuredLines",
                "NumberOfTime30-59DaysPastDueNotWorse",
                "DebtRatio",
                "MonthlyIncome",
                "Age"
            ]
            
            try:
                report_md = generate_mrm_executive_report(sample_metrics, sample_features)
                st.success("Executive Validation Report generated successfully!")
                st.markdown("---")
                st.markdown(report_md)
                
                st.download_button(
                    label="📥 Download Report (.md)",
                    data=report_md,
                    file_name="MRM_Executive_Validation_Report.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Error generating report: {e}")

# ---------------------------------------------------------
# TAB 3: Regulatory RAG Copilot
# ---------------------------------------------------------
with tab3:
    st.header("Regulatory & Policy Document RAG Search")
    st.markdown("Ask natural language queries over uploaded credit policy guidelines, research papers, and supervisory frameworks.")

    @st.cache_resource
    def load_rag():
        return build_or_load_vectorstore()

    vstore = load_rag()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask about Early Warning Systems, default indicators, or policy guidelines...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document vector database..."):
                response = query_rag_system(user_query, vstore)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# ---------------------------------------------------------
# TAB 4: Live Borrower EWS Simulator
# ---------------------------------------------------------
with tab4:
    st.header("Real-Time Borrower Credit Risk & EWS Scoring")
    st.markdown("Adjust borrower risk drivers below to compute real-time Probability of Default (PD) and trigger automated early warning alerts.")

    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        revol_util = st.slider("Revolving Utilization of Unsecured Lines (%)", 0.0, 150.0, 35.0, step=1.0) / 100.0
        age = st.slider("Borrower Age", 18, 95, 45)
        past_due = st.number_input("30-59 Days Past Due (Last 2 Years)", min_value=0, max_value=10, value=0)

    with col_input2:
        debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 3.0, 0.40, step=0.05)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=6500)

    # Risk evaluation model simulation (logistic scoring function)
    logit = -2.8 + (3.5 * revol_util) + (0.6 * past_due) + (0.8 * debt_ratio) - (0.012 * (age - 40)) - (0.00004 * monthly_income)
    pd_score = 1.0 / (1.0 + np.exp(-logit))

    st.markdown("---")
    st.subheader("EWS Risk Assessment Results")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    res_col1.metric("Predicted Probability of Default (PD)", f"{pd_score:.1%}")
    
    if pd_score < 0.15:
        res_col2.metric("EWS Status", "🟢 Normal Monitoring")
        res_col3.metric("Action Required", "Standard Operations")
        st.success("Borrower profile aligns with target credit quality limits. No elevated warning flags detected.")
    elif pd_score < 0.35:
        res_col2.metric("EWS Status", "🟡 Amber Watchlist")
        res_col3.metric("Action Required", "Enhanced Monitoring")
        st.warning("EWS Alert: Borrower exceeds standard risk thresholds. Recommended for review at next Credit Risk Committee.")
    else:
        res_col2.metric("EWS Status", "🔴 Critical EWS Alert")
        res_col3.metric("Action Required", "Immediate Exposure Freeze / Restructure")
        st.error("Critical EWS Flag: High likelihood of 2-year delinquency. Proactively initiate credit restructuring or exposure limits reduction.")