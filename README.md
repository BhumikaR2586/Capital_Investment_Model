# Capital Investment Model

A small, easy-to-follow project for exploring AI-powered capital investment predictions and simple investment analysis.

## What it is
This repository contains a compact Python application that demonstrates a capital investment prediction model. It includes the model code, required dependencies, and a simple entry point (`app.py`) to run the project locally.

## Features
- Basic investment prediction workflow
- Example scripts and a runnable `app.py`
- Uses standard Python tooling and a requirements file for dependencies

## Requirements
- Python 3.8+ (recommended)
- See `requirements.txt` for exact Python packages used in this project

## Setup (Windows PowerShell)
1. Create or activate a virtual environment (this repo includes `inv_env1`):

	 # If you want to create a new environment
	 python -m venv venv

	 # Activate the environment 
	 \venv\Scripts\Activate

2. Install dependencies:

	 pip install -r requirements.txt
	 - make sure the scikit-learn version is 1.6.1

## Run
- The project uses Streamlit (a streamlit CLI is included in the environment):

	streamlit run app.py

## Files of note
- `app.py` — application entry point
- `requirements.txt` — Python dependencies
- `Investment_Dataset.xlsx` - dataset used to train the model
- `investment_decision_model.pkl` - main model
