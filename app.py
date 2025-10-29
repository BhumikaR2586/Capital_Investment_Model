import streamlit as st
import pandas as pd
import numpy as np
import joblib 

# --- 1. Load Model ---
@st.cache_resource
def load_model():
    try:
        # Load the model using joblib
        return joblib.load('investment_decision_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'investment_decision_model.pkl' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Ensure all required libraries (scikit-learn, joblib) are installed.")
        return None

# --- 2. Feature Engineering Function ---
def parse_cash_flows(cf_str):
    """Parses cash flow string and computes the three engineered features."""
    try:
        # Extract and clean cash flow values
        cash_flows = [float(x.strip()) for x in str(cf_str).strip('[]').split(',')]
        
        # Compute features matching the training data
        return {
            'Total_Cash_Inflows': sum(cash_flows),
            'Avg_Cash_Flow': np.mean(cash_flows),
            'CF_Volatility': np.std(cash_flows)
        }
    except ValueError:
        return None # Return None on parsing failure
    except Exception:
        return None

# --- 3. Main App Function ---
def main():
    st.set_page_config(page_title="Investment Decision Predictor", layout="wide")
    st.title("🤖 AI Investment Decision Predictor")
    
    # Load model and check for success
    model = load_model()
    if model is None:
        st.stop()
    
    # --- Input Section ---
    st.header("Project Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Note: Initial Cost is entered as positive, but saved as negative for the model
        initial_cost = st.number_input("Initial Cost ($)", min_value=1000, value=100000, step=1000) * -1
        discount_rate = st.number_input("Discount Rate (%)", min_value=1.0, max_value=25.0, value=10.0, step=0.1)
        duration = st.number_input("Duration (Years)", min_value=1, max_value=10, value=5)
    
    with col2:
        risk_rating = st.selectbox("Risk Rating", ['Low', 'Medium', 'High'])
        project_type = st.selectbox("Project Type", ['Tech', 'Infra', 'Retail', 'Energy', 'Healthcare'])
        market_condition = st.selectbox("Market Condition", ['Stable', 'Unstable'])
    
    # --- Cash flows input ---
    st.subheader("Cash Flows")
    st.info("Enter cash flows as a comma-separated list of values, optionally enclosed in brackets. e.g., 50000, 60000, 70000")
    cash_flows_str = st.text_input("Cash Flows (C1, C2, C3, ...)", value="[50000, 60000, 70000]")
    
    # --- Prediction button ---
    if st.button("Predict Decision", type="primary"):
        # Parse cash flows and calculate engineered features
        cf_stats = parse_cash_flows(cash_flows_str)
        
        if cf_stats is None:
            st.error("❌ Invalid cash flow format. Please use simple comma-separated numbers.")
            return
        
        # Create input dataframe (order must match model training features)
        input_data = pd.DataFrame({
            'Initial_Cost': [initial_cost],
            'Discount_Rate_%': [discount_rate],
            'Risk_Rating': [risk_rating],
            'Project_Type': [project_type],
            'Market_Condition': [market_condition],
            'Duration_Years': [duration],
            **cf_stats # Unpack the engineered features
        })
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display result
            if prediction == 'accept':
                proba = prediction_proba[model.classes_.tolist().index('accept')]
                st.success(f"## ✅ Decision: ACCEPT")
                st.write(f"Confidence: **{proba:.2%}**")
            else:
                proba = prediction_proba[model.classes_.tolist().index('reject')]
                st.error(f"## ❌ Decision: REJECT")
                st.write(f"Confidence: **{proba:.2%}**")
            
            # Show input summary
            with st.expander("View Input Summary"):
                st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
                
        except Exception as e:
            st.error(f"Prediction error. This may indicate a mismatch between the input features and the trained model pipeline. Error: {str(e)}")

if __name__ == "__main__":
    main()