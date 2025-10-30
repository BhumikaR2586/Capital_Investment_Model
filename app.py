import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
from datetime import datetime
from ui import load_custom_ui  

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Investment Decision Predictor",
    page_icon="💰",
    layout="wide"
)

# Load UI
load_custom_ui()

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        with open('models/cash_flow_regressor.pkl', 'rb') as file:
            regressor_model = pickle.load(file)
        classifier_model = joblib.load('models/decision_classifier.pkl')
        return regressor_model, classifier_model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.error("Please ensure 'cash_flow_regressor.pkl' and 'decision_classifier.pkl' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

regressor_model, classifier_model = load_models()

# --- Financial Metrics Function ---
def calculate_financial_metrics(initial_cost, discount_rate, cash_flows):
    """Calculate NPV, IRR, PI, and Payback Period with error handling"""
    try:
        cash_flows = np.array(cash_flows)
        npv = np.sum(cash_flows / (1 + discount_rate/100) ** np.arange(1, len(cash_flows) + 1)) - abs(initial_cost)
        try:
            all_cash_flows = np.concatenate([[initial_cost], cash_flows])
            irr = np.irr(all_cash_flows) * 100
        except:
            irr = (np.sum(cash_flows) / abs(initial_cost) - 1) * 100 / len(cash_flows)
        pi = (npv + abs(initial_cost)) / abs(initial_cost) if initial_cost != 0 else 0

        cumulative_cf = 0
        payback_period = None
        for i, cf in enumerate(cash_flows, 1):
            cumulative_cf += cf
            if cumulative_cf >= abs(initial_cost):
                if i > 1:
                    prev_cumulative = cumulative_cf - cf
                    remaining = abs(initial_cost) - prev_cumulative
                    fraction = remaining / cf if cf != 0 else 0
                    payback_period = i - 1 + fraction
                else:
                    payback_period = i
                break
        if payback_period is None:
            payback_period = len(cash_flows)
        return npv, irr, pi, payback_period
    except Exception as e:
        st.error(f"Error calculating financial metrics: {e}")
        return 0, 0, 0, 0

# --- Sidebar Inputs ---
st.sidebar.markdown('<h2 class="sub-header">Project Parameters</h2>', unsafe_allow_html=True)

initial_cost = st.sidebar.number_input("Initial Cost ($)", min_value=-1000000, max_value=0, value=-100000, step=1000, help="Initial investment cost (negative value)")
discount_rate = st.sidebar.number_input("Discount Rate (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1, help="Annual discount rate for NPV calculation")
duration_years = st.sidebar.slider("Project Duration (Years)", min_value=1, max_value=10, value=5, help="Expected duration of the project in years")
risk_rating = st.sidebar.selectbox("Risk Rating", options=["Low", "Medium", "High"], index=1)
project_type = st.sidebar.selectbox("Project Type", options=["Retail", "Tech", "Healthcare", "Energy", "Infra"], index=0)
market_condition = st.sidebar.selectbox("Market Condition", options=["Stable", "Unstable", "Volatile"], index=0)

# --- Main Analysis ---
if st.sidebar.button("🔍 Analyze Investment", type="primary"):
    if regressor_model is None or classifier_model is None:
        st.error("Models could not be loaded. Please check the model files and try again.")
    else:
        try:
            years = list(range(1, duration_years + 1))
            prediction_data = pd.DataFrame({
                'Year': years,
                'Risk_Rating': [risk_rating] * duration_years,
                'Project_Type': [project_type] * duration_years,
                'Market_Condition': [market_condition] * duration_years
            })
            prediction_data = prediction_data.astype(str)

            predicted_cash_flows = regressor_model.predict(prediction_data)
            predicted_cash_flows = np.maximum(predicted_cash_flows, 0)

            npv, irr, pi, payback_period = calculate_financial_metrics(initial_cost, discount_rate, predicted_cash_flows)
            total_inflows = np.sum(predicted_cash_flows)
            avg_cash_flow = np.mean(predicted_cash_flows)
            cf_volatility = np.std(predicted_cash_flows)

            classifier_data = pd.DataFrame({
                'Initial_Cost': [initial_cost],
                'Discount_Rate_%': [discount_rate],
                'Risk_Rating': [risk_rating],
                'Project_Type': [project_type],
                'Market_Condition': [market_condition],
                'Duration_Years': [duration_years],
                'Total_Cash_Inflows': [total_inflows],
                'Avg_Cash_Flow': [avg_cash_flow],
                'CF_Volatility': [cf_volatility]
            }).astype(str)

            decision = classifier_model.predict(classifier_data)[0]
            decision_proba = classifier_model.predict_proba(classifier_data)[0]
            confidence = max(decision_proba) * 100

            st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
            decision_color = "positive" if decision == "accept" else "negative"
            st.markdown(f"""
            <div class="result-box">
                <h3>Investment Decision: <span class="{decision_color}">{decision.upper()}</span></h3>
                <p>Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                npv_color = "positive" if npv > 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Net Present Value</h4>
                    <p class="{npv_color}">${npv:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                irr_color = "positive" if irr > discount_rate else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Internal Rate of Return</h4>
                    <p class="{irr_color}">{irr:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                pi_color = "positive" if pi > 1 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Profitability Index</h4>
                    <p class="{pi_color}">{pi:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                payback_color = "positive" if payback_period <= duration_years * 0.5 else "warning" if payback_period <= duration_years else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Payback Period</h4>
                    <p class="{payback_color}">{payback_period:.1f} years</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<h2 class="sub-header">📈 Predicted Cash Flows</h2>', unsafe_allow_html=True)
            cf_df = pd.DataFrame({
                'Year': years,
                'Predicted Cash Flow': predicted_cash_flows,
                'Cumulative Cash Flow': np.cumsum(predicted_cash_flows)
            })
            st.line_chart(cf_df.set_index('Year'))

            st.markdown('<h3>Cash Flow Details</h3>', unsafe_allow_html=True)
            cf_table = pd.DataFrame({
                'Year': years,
                'Cash Flow': [f"${cf:,.2f}" for cf in predicted_cash_flows],
                'Cumulative': [f"${cf:,.2f}" for cf in np.cumsum(predicted_cash_flows)]
            })
            st.table(cf_table)

            st.markdown('<h2 class="sub-header">💡 Investment Insights</h2>', unsafe_allow_html=True)
            insights = []
            if npv > 0:
                insights.append("✅ Positive NPV indicates the project is expected to generate value")
            else:
                insights.append("❌ Negative NPV suggests the project may not be profitable")
            if irr > discount_rate:
                insights.append("✅ IRR exceeds the discount rate, indicating good return potential")
            else:
                insights.append("❌ IRR is below the discount rate, consider alternative investments")
            if pi > 1:
                insights.append("✅ PI > 1 means the project creates value for each dollar invested")
            else:
                insights.append("❌ PI < 1 indicates the project destroys value")
            if payback_period <= duration_years * 0.5:
                insights.append("✅ Quick payback period reduces investment risk")
            elif payback_period <= duration_years:
                insights.append("⚠️ Moderate payback period")
            else:
                insights.append("❌ Payback period exceeds project duration")
            if risk_rating == "High":
                insights.append("⚠️ High risk project requires careful consideration")
            elif risk_rating == "Low":
                insights.append("✅ Low risk project with stable returns expected")
            for insight in insights:
                st.markdown(f"<p>{insight}</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.error("Please check your input values and try again.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Investment Decision Predictor | Powered by Machine Learning</p>
    <p>This tool provides predictions based on historical data and should not be the sole basis for investment decisions.</p>
</div>
""", unsafe_allow_html=True)
