"""
Streamlit UI for the AI-Powered Financial Fraud Detection & Risk Scoring
System. Talks to the FastAPI backend only via HTTP — it holds no model or
DB logic of its own, matching the architecture in the project PPT.
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
if not API_URL.startswith(("http://", "https://")):
    API_URL = "http://" + API_URL  # e.g. Render private-network "host:port"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="🛡️")

RISK_COLORS = {"LOW": "#22C55E", "MEDIUM": "#F59E0B", "HIGH": "#EF4444"}

st.title("🛡️ AI-Powered Financial Fraud Detection & Risk Scoring")

tab_check, tab_dashboard = st.tabs(["🔍 Check Transaction", "📊 Monitoring Dashboard"])

# ----------------------------------------------------------------------
# TAB 1 — Transaction Prediction Screen
# ----------------------------------------------------------------------
with tab_check:
    st.subheader("Check a Transaction")
    col_form, col_result = st.columns([1, 1])

    with col_form:
        with st.form("transaction_form"):
            txn_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
            amount = st.number_input("Amount (₹)", min_value=0.0, value=85000.0, step=100.0)
            step = st.number_input("Time Step", min_value=0, value=521, step=1)
            old_org = st.number_input("Sender balance before (₹)", min_value=0.0, value=90000.0, step=100.0)
            new_org = st.number_input("Sender balance after (₹)", min_value=0.0, value=5000.0, step=100.0)
            old_dest = st.number_input("Receiver balance before (₹)", min_value=0.0, value=0.0, step=100.0)
            new_dest = st.number_input("Receiver balance after (₹)", min_value=0.0, value=0.0, step=100.0)
            submitted = st.form_submit_button("Check Transaction", use_container_width=True)

    with col_result:
        st.markdown("#### Model Result")
        if submitted:
            payload = {
                "type": txn_type, "amount": amount, "step": int(step),
                "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
                "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
            }
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                level = result["risk_level"]
                color = RISK_COLORS[level]

                m1, m2, m3 = st.columns(3)
                m1.metric("Fraud Probability", f"{result['fraud_probability'] * 100:.1f}%")
                m2.metric("Risk Score", f"{result['risk_score']:.0f} / 100")
                m3.markdown(
                    f"<div style='padding:0.6rem;border-radius:8px;background:{color}22;"
                    f"border:1px solid {color};text-align:center;font-weight:700;color:{color}'>"
                    f"{level} RISK</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Prediction:** {result['prediction']}")
                st.caption(
                    f"Transaction ID: {result['transaction_id']} • "
                    f"Model: {result['model_version']}"
                )
                if level == "HIGH":
                    st.error("⚠️ Flag for review according to configured business policy.")
                elif level == "MEDIUM":
                    st.warning("Additional verification or monitoring recommended.")
                else:
                    st.success("Approve / normal processing.")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not reach the API at {API_URL}. Is the backend running?")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")
        else:
            st.info("Fill in the transaction details and click **Check Transaction**.")

# ----------------------------------------------------------------------
# TAB 2 — Streamlit Monitoring Dashboard
# ----------------------------------------------------------------------
with tab_dashboard:
    st.subheader("Monitoring Dashboard")
    st.caption("A visual layer for operators and project demonstration")

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        stats = requests.get(f"{API_URL}/stats", timeout=10).json()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f"{stats['total_transactions']:,}")
        c2.metric("Fraud", f"{stats['fraud_count']:,}")
        c3.metric("Fraud Rate", f"{stats['fraud_rate_pct']}%")
        c4.metric("High Risk", f"{stats['high_risk_count']:,}")

        col_chart, col_alerts = st.columns([1, 1])

        with col_chart:
            st.markdown("**Fraud by Transaction Type**")
            if stats["fraud_by_type"]:
                chart_df = pd.DataFrame(
                    list(stats["fraud_by_type"].items()), columns=["Type", "Fraud Count"]
                ).set_index("Type")
                st.bar_chart(chart_df)
            else:
                st.info("No fraud recorded yet — check some transactions first.")

        with col_alerts:
            st.markdown("**Recent High-Risk Alerts**")
            if stats["recent_high_risk"]:
                alerts_df = pd.DataFrame(stats["recent_high_risk"])[
                    ["transaction_id", "risk_score", "risk_level"]
                ]
                st.dataframe(alerts_df, use_container_width=True, hide_index=True)
            else:
                st.info("No high-risk transactions yet.")

    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the API at {API_URL}. Is the backend running?")
