# Capital Investment Model 💰

A sophisticated machine learning-powered application for predicting investment decisions and analyzing capital investment opportunities. This project combines financial metrics with ML models to provide data-driven investment recommendations.

## What it is
This repository contains a Streamlit web application that uses machine learning models to:
- Predict future cash flows for investment projects
- Calculate key financial metrics (NPV, IRR, PI, Payback Period)
- Provide investment recommendations with confidence scores
- Visualize projected cash flows and financial insights

## Features
- 🤖 ML-powered cash flow prediction
- 📊 Comprehensive financial analysis including:
  - Net Present Value (NPV) calculation
  - Internal Rate of Return (IRR)
  - Profitability Index (PI)
  - Payback Period estimation
- 📈 Interactive data visualization
- 🎯 Risk-adjusted investment recommendations
- 💡 Automated insights generation
- 🎨 Clean, modern UI with Streamlit

## Models
The project uses two main machine learning models:
- `cash_flow_regressor.pkl`: Predicts yearly cash flows based on project parameters
- `decision_classifier.pkl`: Makes final investment recommendations based on financial metrics and project characteristics

## Requirements
- Python 3.8+ (recommended)
- scikit-learn == 1.6.1 (required)
- Other dependencies listed in `requirements.txt`

## Setup (Windows PowerShell)
1. Clone the repository and navigate to the project directory

2. Create and activate a virtual environment:
   ```powershell
   # Create new environment
   python -m venv inv_env1

   # Activate the environment
   .\inv_env1\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   Note: Make sure scikit-learn version 1.6.1 is installed correctly

## Usage
1. Start the Streamlit application:
   ```powershell
   streamlit run app.py
   ```

2. Enter project parameters in the sidebar:
   - Initial Cost ($): Investment required (negative value)
   - Discount Rate (%): Annual discount rate for NPV
   - Project Duration: Expected length in years (1-10)
   - Risk Rating: Low/Medium/High
   - Project Type: Retail/Tech/Healthcare/Energy/Infra
   - Market Condition: Stable/Unstable/Volatile

3. Click "Analyze Investment" to get:
   - Investment recommendation with confidence score
   - Financial metrics dashboard
   - Cash flow projections and visualizations
   - Risk-adjusted insights

## Files
- `app.py` — Streamlit application with UI and analysis logic
- `requirements.txt` — Python package dependencies
- `Investment_Dataset.xlsx` - Training dataset (synthetic but realistic)
- `cash_flow_regressor.pkl` - Cash flow prediction model
- `decision_classifier.pkl` - Investment decision model

## Note
While the dataset used is synthetic, it's designed to reflect realistic investment scenarios and market conditions. The models are intended to demonstrate ML-powered investment analysis but should not be the sole basis for actual investment decisions.

## Contributing
Feel free to open issues or submit pull requests if you have suggestions for improvements or find any bugs. 