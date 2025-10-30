import streamlit as st

def load_custom_ui():
    """Load custom CSS and page headers with a light blue gradient theme."""
    
    # --- Custom CSS Styling ---
    st.markdown("""
    <style>
        /* === Global Background === */
        .stApp {
            background: linear-gradient(180deg, #e8f1fa 0%, #f6fbff 100%);
            background-attachment: fixed;
        }

        /* === Main Header === */
        .main-header {
            font-size: 2.5rem;
            color: #1e6091;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 700;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }

        /* === Sub Header === */
        .sub-header {
            font-size: 1.5rem;
            color: #168aad;
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        /* === Result Box === */
        .result-box {
            background: linear-gradient(135deg, #d7ecff 0%, #eaf6ff 100%);
            padding: 1.2rem;
            border-radius: 0.75rem;
            margin: 1rem 0;
            border: 1px solid #cce0ff;
            box-shadow: 0 2px 6px rgba(0, 85, 170, 0.1);
        }

        /* === Metric Cards === */
        .metric-card {
            background: #ffffff;
            padding: 1rem;
            border-radius: 0.75rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.08);
            margin: 0.5rem;
            border-left: 5px solid #1e6091;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 10px rgba(0,0,0,0.15);
        }

        /* === Text Colors for Results === */
        .positive {
            color: #2a9d8f;
            font-weight: 700;
        }

        .negative {
            color: #d62828;
            font-weight: 700;
        }

        .warning {
            color: #f4a261;
            font-weight: 700;
        }

        /* === Streamlit Sidebar Styling === */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #d8ebff 0%, #f0f7ff 100%);
            border-right: 1px solid #cce0ff;
        }

        /* Sidebar Titles */
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label {
            color: #144b75 !important;
        }

        /* === Streamlit Buttons === */
        div.stButton > button {
            background: linear-gradient(90deg, #1e6091 0%, #1a759f 100%);
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            background: linear-gradient(90deg, #168aad 0%, #1a759f 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        /* === Tables === */
        .stTable {
            background: white !important;
            border-radius: 0.5rem !important;
            border: 1px solid #cce0ff !important;
            box-shadow: 0 2px 6px rgba(0, 85, 170, 0.08);
        }

        /* === Footer === */
        footer {
            visibility: hidden;
        }

    </style>
    """, unsafe_allow_html=True)

    # --- Page Header ---
    st.markdown('<h1 class="main-header">💰 Investment Decision Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    This application uses **machine learning** to predict cash flows and guide investment decisions.  
    Enter your project parameters in the sidebar to get a detailed financial and risk analysis.
    """)

